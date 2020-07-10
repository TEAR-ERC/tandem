#ifndef GLOBALSIMPLEXMESH_H
#define GLOBALSIMPLEXMESH_H

#include "LocalFaces.h"
#include "LocalSimplexMesh.h"
#include "MeshData.h"
#include "Simplex.h"

#include "parallel/CommPattern.h"
#include "parallel/DistributedCSR.h"
#include "parallel/MPITraits.h"
#include "parallel/MetisPartitioner.h"
#include "parallel/SortedDistribution.h"
#include "util/Algorithm.h"
#include "util/Utility.h"

#include <mpi.h>

#include <array>
#include <cstddef>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tndm {

/**
 * @brief Class that holds a distributed D-simplex mesh.
 *
 * Global means that vertices and elements (an element is a D-simplex) exist only once in the
 * distributed memory space. Moreover, vertices and elements are distributed independently, that is,
 * the vertices required by an element may not reside on the same rank.
 *
 * One may attach vertex data and element data to a global mesh. If you need facet data or edge
 * data, then you have to add a boundary mesh. (The element data on the boundary mesh is then going
 * to be the edge or facet data.)
 *
 * @tparam D simplex dimension
 */
template <std::size_t D> class GlobalSimplexMesh {
public:
    using simplex_t = Simplex<D>;
    static_assert(sizeof(simplex_t) == (D + 1) * sizeof(uint64_t));
    template <std::size_t DD> using global_mesh_ptr = std::unique_ptr<GlobalSimplexMesh<DD>>;

    GlobalSimplexMesh(std::vector<simplex_t>&& elements,
                      std::unique_ptr<MeshData> vertexDat = nullptr,
                      std::unique_ptr<MeshData> elementDat = nullptr,
                      MPI_Comm comm = MPI_COMM_WORLD)
        : elems_(std::move(elements)), vertexData(std::move(vertexDat)),
          elementData(std::move(elementDat)), comm(comm), isPartitionedByHash(false) {
        if (vertexData) {
            vtxdist = makeSortedDistribution(vertexData->size());
        }
    }

    auto const& getElements() const { return elems_; }
    std::size_t numElements() const { return elems_.size(); }

    template <std::size_t DD> void setBoundaryMesh(global_mesh_ptr<DD> boundaryMesh) {
        static_assert(0 < DD && DD < D);
        std::get<DD>(boundaryMeshes) = std::move(boundaryMesh);
    }

    /**
     * @brief Mesh topology for partitioning.
     *
     * @tparam OutIntT Integer type of distributed CSR.
     *
     * @return Returns mesh in distributed CSR format as required by ParMETIS.
     */
    template <typename OutIntT> DistributedCSR<OutIntT> distributedCSR() const {
        DistributedCSR<OutIntT> csr;

        auto elmdist = makeSortedDistribution(numElements());
        csr.dist.resize(elmdist.size());
        std::copy(elmdist.begin(), elmdist.end(), csr.dist.begin());

        auto numElems = numElements();
        csr.rowPtr.resize(numElems + 1);
        csr.colInd.resize(numElems * (D + 1));

        OutIntT ind = 0;
        OutIntT ptr = 0;
        for (auto& e : elems_) {
            csr.rowPtr[ptr++] = ind;
            for (auto& p : e) {
                csr.colInd[ind++] = p;
            }
        }
        csr.rowPtr.back() = ind;

        return csr;
    }

    /**
     * @brief Use ParMETIS to optimise mesh partitioning.
     */
    void repartition() {
        auto distCSR = distributedCSR<idx_t>();
        auto partition = MetisPartitioner::partition(distCSR, D);

        doPartition(partition);
        isPartitionedByHash = false;
    }

    /**
     * @brief Partition elements by their hash value (SimplexHash).
     *
     * Should only be used for efficient element data queries. Otherwise use repartition().
     */
    void repartitionByHash() {
        if (isPartitionedByHash) {
            return;
        }
        std::vector<idx_t> partition;
        partition.reserve(numElements());
        auto plex2rank = getPlex2Rank<D>();
        for (auto& e : elems_) {
            partition.emplace_back(plex2rank(e));
        }

        doPartition(partition);
        isPartitionedByHash = true;
    }

    /**
     * @brief Local mesh construction with ghost entities.
     *
     * @param overlap Number of elements the partitions shall overlap
     *
     * @return
     */
    std::unique_ptr<LocalSimplexMesh<D>> getLocalMesh(unsigned overlap = 0) const {
        auto localFaces = getAllLocalFaces(overlap, std::make_index_sequence<D>{});

        return std::make_unique<LocalSimplexMesh<D>>(std::move(localFaces));
    }

private:
    template <std::size_t DD> friend class GlobalSimplexMesh;

    auto makeG2LMap() const {
        std::unordered_map<Simplex<D>, std::size_t, SimplexHash<D>> map;
        std::size_t local = 0;
        for (auto& e : elems_) {
            map[e] = local++;
        }
        return map;
    }

    void doPartition(std::vector<idx_t> const& partition) {
        int procs, rank;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &procs);
        assert(partition.size() == numElements());

        std::vector<std::size_t> enumeration(partition.size());
        std::iota(enumeration.begin(), enumeration.end(), std::size_t(0));

        std::sort(
            enumeration.begin(), enumeration.end(),
            [&partition](std::size_t a, std::size_t b) { return partition[a] < partition[b]; });

        std::vector<int> sendcounts(procs, 0);
        std::vector<simplex_t> elemsToSend;
        elemsToSend.reserve(numElements());
        auto eIt = enumeration.begin();
        for (int p = 0; p < procs; ++p) {
            while (eIt != enumeration.end() && partition[*eIt] <= p) {
                ++sendcounts[p];
                elemsToSend.emplace_back(elems_[*eIt]);
                ++eIt;
            }
        }
        assert(eIt == enumeration.end());

        AllToAllV a2a(std::move(sendcounts), comm);
        mpi_array_type<uint64_t> mpi_simplex_t(D + 1);
        elems_ = a2a.exchange(elemsToSend, mpi_simplex_t.get());

        if (elementData) {
            elementData = elementData->redistributed(enumeration, a2a);
        }
    }

    template <std::size_t... Is>
    auto getAllLocalFaces(unsigned overlap, std::index_sequence<Is...>) const {
        auto localElems = getGhostElements(elems_, overlap);
        return std::make_tuple(getFaces<Is>(localElems.faces())..., std::move(localElems));
    }

    LocalFaces<D> getGhostElements(std::vector<Simplex<D>> elems, unsigned overlap) const {
        auto myComm = comm;
        auto const getElementContiguousGIDs = [&myComm](std::size_t numElems) {
            std::size_t ownedSize = numElems;
            std::size_t gidOffset;
            MPI_Scan(&ownedSize, &gidOffset, 1, mpi_type_t<std::size_t>(), MPI_SUM, myComm);
            gidOffset -= ownedSize;
            std::vector<std::size_t> cGIDs(numElems);
            std::iota(cGIDs.begin(), cGIDs.end(), gidOffset);
            return cGIDs;
        };
        auto cGIDs = getElementContiguousGIDs(numElements());

        if (overlap == 0) {
            return LocalFaces<D>(std::move(elems), std::move(cGIDs));
        }

        int rank, procs;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &procs);
        mpi_array_type<uint64_t> mpi_facet_t(D);
        mpi_array_type<uint64_t> mpi_elem_t(D + 1u);
        auto plex2rank = getPlex2Rank<D - 1u>();

        auto const makeDistributedUpwardMap = [&procs, &myComm, &mpi_facet_t, &mpi_elem_t,
                                               &plex2rank](auto&& elems,
                                                           std::vector<std::size_t> const& cGIDs) {
            std::unordered_multimap<Simplex<D - 1u>, std::size_t, SimplexHash<D - 1u>> up;
            for (std::size_t elNo = 0; elNo < elems.size(); ++elNo) {
                auto downward = elems[elNo].template downward<D - 1u>();
                for (auto& s : downward) {
                    up.emplace(s, elNo);
                }
            }
            std::vector<int> counts(procs, 0);
            for (auto&& u : up) {
                ++counts[plex2rank(u.first)];
            }
            std::vector<std::ptrdiff_t> offsets(procs + 1);
            offsets[0] = 0;
            std::partial_sum(counts.begin(), counts.end(), offsets.begin() + 1);

            std::vector<Simplex<D - 1u>> upFaces(offsets.back());
            std::vector<Simplex<D>> upElems(offsets.back());
            std::vector<std::size_t> upCGIDs(offsets.back());
            for (auto&& u : up) {
                auto rank = plex2rank(u.first);
                upFaces[offsets[rank]] = u.first;
                upElems[offsets[rank]] = elems[u.second];
                upCGIDs[offsets[rank]] = cGIDs[u.second];
                ++offsets[rank];
            }

            AllToAllV a2a(std::move(counts), myComm);
            auto receivedUpFaces = a2a.exchange(upFaces, mpi_facet_t.get());
            auto receivedUpElems = a2a.exchange(upElems, mpi_elem_t.get());
            auto receivedUpCGIDs = a2a.exchange(upCGIDs);

            std::unordered_multimap<Simplex<D - 1u>, std::pair<std::size_t, Simplex<D>>,
                                    SimplexHash<D - 1u>>
                distUp;
            for (std::size_t i = 0; i < receivedUpFaces.size(); ++i) {
                distUp.emplace(receivedUpFaces[i],
                               std::make_pair(receivedUpCGIDs[i], receivedUpElems[i]));
            }
            return distUp;
        };

        auto distUp = makeDistributedUpwardMap(elems, cGIDs);

        for (int level = 0; level < static_cast<int>(overlap); ++level) {
            auto up = getBoundaryFaces(elems);

            auto const getRequestedFacesAndMissingVertices = [&procs, &myComm, &mpi_facet_t,
                                                              &plex2rank](auto&& elems, auto&& up) {
                std::vector<int> counts(procs, 0);
                for (auto&& u : up) {
                    ++counts[plex2rank(u.first)];
                }
                std::vector<std::ptrdiff_t> offsets(procs + 1);
                offsets[0] = 0;
                std::partial_sum(counts.begin(), counts.end(), offsets.begin() + 1);

                std::vector<Simplex<D - 1u>> boundaryFaces(offsets.back());
                std::vector<uint64_t> missingVertex(offsets.back());
                for (auto&& u : up) {
                    // Find missing vertex i.e. elem \ facet
                    auto& elem = elems[u.second];
                    auto f = u.first.begin();
                    auto e = elem.begin();
                    while (e != elem.end() && f != u.first.end() && *e == *f) {
                        ++f;
                        ++e;
                    }
                    assert(e != elem.end());

                    auto rank = plex2rank(u.first);
                    boundaryFaces[offsets[rank]] = u.first;
                    missingVertex[offsets[rank]] = *e;
                    ++offsets[rank];
                }

                AllToAllV a2a(std::move(counts), myComm);
                auto requestedBoundaryFaces = a2a.exchange(boundaryFaces, mpi_facet_t.get());
                auto requestedMissingVertices = a2a.exchange(missingVertex);
                return std::make_tuple(requestedBoundaryFaces, requestedMissingVertices, a2a);
            };

            auto [requestedBoundaryFaces, requestedMissingVertices, a2a] =
                getRequestedFacesAndMissingVertices(elems, up);

            std::vector<Simplex<D>> requestedBoundaryElems(requestedBoundaryFaces.size());
            std::vector<std::size_t> requestedCGIDs(requestedBoundaryFaces.size());
            for (std::size_t i = 0; i < requestedBoundaryFaces.size(); ++i) {
                auto [it, end] = distUp.equal_range(requestedBoundaryFaces[i]);
                while (it != end &&
                       std::find(it->second.second.begin(), it->second.second.end(),
                                 requestedMissingVertices[i]) != it->second.second.end()) {
                    ++it;
                }
                // Neighbour could not be found (i.e. domain boundary)
                if (it == end) {
                    requestedCGIDs[i] = std::numeric_limits<std::size_t>::max();
                    requestedBoundaryElems[i] = Simplex<D>::invalidSimplex();
                } else {
                    requestedCGIDs[i] = it->second.first;
                    requestedBoundaryElems[i] = it->second.second;
                }
            }
            a2a.swap();
            auto boundaryElems = a2a.exchange(requestedBoundaryElems, mpi_elem_t.get());
            auto boundaryCGIDs = a2a.exchange(requestedCGIDs);

            // Remove duplicates
            std::vector<std::size_t> enumeration(boundaryElems.size());
            std::iota(enumeration.begin(), enumeration.end(), std::size_t(0));
            std::sort(enumeration.begin(), enumeration.end(),
                      [&boundaryElems](std::size_t a, std::size_t b) {
                          return boundaryElems[a] < boundaryElems[b];
                      });
            apply_permutation(boundaryElems, enumeration);
            apply_permutation(boundaryCGIDs, enumeration);
            auto firstInvalid =
                std::find(boundaryElems.begin(), boundaryElems.end(), Simplex<D>::invalidSimplex());
            boundaryElems.erase(firstInvalid, boundaryElems.end());
            boundaryCGIDs.erase(boundaryCGIDs.begin() +
                                    std::distance(boundaryElems.begin(), firstInvalid),
                                boundaryCGIDs.end());
            boundaryElems.erase(std::unique(boundaryElems.begin(), boundaryElems.end()),
                                boundaryElems.end());
            boundaryCGIDs.erase(std::unique(boundaryCGIDs.begin(), boundaryCGIDs.end()),
                                boundaryCGIDs.end());

            assert(boundaryElems.size() == boundaryCGIDs.size());

            elems.reserve(elems.size() + boundaryElems.size());
            elems.insert(elems.end(), boundaryElems.begin(), boundaryElems.end());
            cGIDs.reserve(cGIDs.size() + boundaryCGIDs.size());
            cGIDs.insert(cGIDs.end(), boundaryCGIDs.begin(), boundaryCGIDs.end());
        }
        return LocalFaces<D>(std::move(elems), std::move(cGIDs));
    }

    auto getBoundaryFaces(std::vector<Simplex<D>> const& elems) const {
        // Construct upward map from faces to local element ids
        std::unordered_multimap<Simplex<D - 1u>, std::size_t, SimplexHash<D - 1u>> up;
        for (std::size_t elNo = 0; elNo < elems.size(); ++elNo) {
            auto downward = elems[elNo].template downward<D - 1u>();
            for (auto& s : downward) {
                up.emplace(s, elNo);
            }
        }
        // Delete all internal faces and count number of faces per rank
        auto const deleteInternalFaces = [](auto& up) {
            for (auto it = up.begin(); it != up.end();) {
                auto count = up.count(it->first);
                assert(count <= 2);
                if (count > 1) {
                    auto range = up.equal_range(it->first);
                    it = up.erase(range.first, range.second);
                } else {
                    ++it;
                }
            }
        };
        deleteInternalFaces(up);
        return up;
    }

    template <std::size_t DD> auto getPlex2Rank() const {
        int procs;
        MPI_Comm_size(comm, &procs);
        if constexpr (DD == 0) {
            if (vtxdist.size() > 0) {
                SortedDistributionToRank v2r(vtxdist);
                return std::function([v2r](Simplex<0> const& plex) -> int { return v2r(plex[0]); });
            }
            return std::function(
                [procs](Simplex<0> const& plex) -> int { return plex[0] % procs; });
        } else {
            return [procs](Simplex<DD> const& plex) { return SimplexHash<DD>()(plex) % procs; };
        }
    }

    std::size_t getVertexLID(Simplex<0> const& plex) const {
        int rank;
        MPI_Comm_rank(comm, &rank);
        assert(plex[0] >= vtxdist[rank] && plex[0] < vtxdist[rank + 1]);
        return plex[0] - vtxdist[rank];
    }

    template <std::size_t DD> auto getFaces(std::vector<Simplex<D>> const& elems) const {
        auto plex2rank = getPlex2Rank<DD>();

        int procs;
        MPI_Comm_size(comm, &procs);

        std::vector<std::set<Simplex<DD>>> requiredFaces(procs);
        for (auto& elem : elems) {
            auto downward = elem.template downward<DD>();
            for (auto& s : downward) {
                requiredFaces[plex2rank(s)].insert(s);
            }
        }
        std::vector<int> counts(procs, 0);
        std::size_t total = 0;
        for (int p = 0; p < procs; ++p) {
            auto size = requiredFaces[p].size();
            counts[p] = size;
            total += size;
        }

        std::vector<Simplex<DD>> faces;
        faces.reserve(total);
        for (auto& perRank : requiredFaces) {
            std::copy(perRank.begin(), perRank.end(), std::back_inserter(faces));
        }

        // Exchange data
        AllToAllV a2a(counts, comm);
        mpi_array_type<uint64_t> mpi_plex_t(DD + 1);
        auto requestedFaces = a2a.exchange(faces, mpi_plex_t.get());
        a2a.swap();

        auto lf = LocalFaces<DD>(std::move(faces), getContiguousGIDs(requestedFaces, a2a));
        if constexpr (DD == 0) {
            if (vertexData) {
                std::vector<std::size_t> lids;
                lids.reserve(requestedFaces.size());
                for (auto& face : requestedFaces) {
                    lids.emplace_back(getVertexLID(face));
                }
                lf.setMeshData(vertexData->redistributed(lids, a2a));
            }
        } else if constexpr (0 < DD && DD < D) {
            auto& boundaryMesh = std::get<DD>(boundaryMeshes);
            if (boundaryMesh && boundaryMesh->elementData) {
                boundaryMesh->repartitionByHash();
                auto map = boundaryMesh->makeG2LMap();
                std::vector<std::size_t> lids;
                lids.reserve(requestedFaces.size());
                for (auto& face : requestedFaces) {
                    auto it = map.find(face);
                    if (it == map.end()) {
                        lids.emplace_back(std::numeric_limits<std::size_t>::max());
                    } else {
                        lids.emplace_back(it->second);
                    }
                }
                lf.setMeshData(boundaryMesh->elementData->redistributed(lids, a2a));
            }
        }

        getSharedRanks(lf, requestedFaces, a2a);
        return lf;
    }

    template <std::size_t DD>
    std::vector<std::size_t> getContiguousGIDs(std::vector<Simplex<DD>> const& requestedFaces,
                                               AllToAllV const& a2a) const {
        std::vector<std::size_t> cGIDs;
        cGIDs.reserve(requestedFaces.size());
        if constexpr (DD == 0) {
            for (auto& face : requestedFaces) {
                cGIDs.emplace_back(face[0]);
            }
        } else {
            int rank;
            MPI_Comm_rank(comm, &rank);

            std::map<Simplex<DD>, std::size_t> ownedFacesToCGID;
            for (auto& face : requestedFaces) {
                ownedFacesToCGID[face] = -1;
            }

            std::size_t ownedSize = ownedFacesToCGID.size();
            std::size_t gidOffset;
            MPI_Scan(&ownedSize, &gidOffset, 1, mpi_type_t<std::size_t>(), MPI_SUM, comm);
            gidOffset -= ownedSize;

            std::size_t cGID = gidOffset;
            for (auto& faceToGID : ownedFacesToCGID) {
                faceToGID.second = cGID++;
            }

            for (auto& face : requestedFaces) {
                cGIDs.emplace_back(ownedFacesToCGID[face]);
            }
        }
        assert(requestedFaces.size() == cGIDs.size());

        return a2a.exchange(cGIDs);
    }

    template <std::size_t DD>
    void getSharedRanks(LocalFaces<DD>& lf, std::vector<Simplex<DD>> const& requestedFaces,
                        AllToAllV const& a2a) const {
        int procs;
        MPI_Comm_size(comm, &procs);

        std::unordered_map<Simplex<DD>, std::vector<int>, SimplexHash<DD>> sharedRanksInfo;
        for (auto [p, i] : a2a.getSDispls()) {
            sharedRanksInfo[requestedFaces[i]].emplace_back(p);
        }

        std::vector<int> sharedRanksSendCount;
        sharedRanksSendCount.reserve(requestedFaces.size());
        std::size_t totalSharedRanksSendCount = 0;
        for (auto& face : requestedFaces) {
            sharedRanksSendCount.emplace_back(sharedRanksInfo[face].size());
            totalSharedRanksSendCount += sharedRanksInfo[face].size();
        }

        auto sharedRanksRecvCount = a2a.exchange(sharedRanksSendCount);

        std::vector<int> requestedSharedRanks;
        requestedSharedRanks.reserve(totalSharedRanksSendCount);
        for (auto& face : requestedFaces) {
            std::copy(sharedRanksInfo[face].begin(), sharedRanksInfo[face].end(),
                      std::back_inserter(requestedSharedRanks));
        }
        std::vector<int> sendcounts(procs, 0);
        std::vector<int> recvcounts(procs, 0);
        for (auto [p, i] : a2a.getSDispls()) {
            sendcounts[p] += sharedRanksSendCount[i];
        }
        for (auto [p, i] : a2a.getRDispls()) {
            recvcounts[p] += sharedRanksRecvCount[i];
        }

        AllToAllV a2aSharedRanks(std::move(sendcounts), std::move(recvcounts));
        auto sharedRanks = a2aSharedRanks.exchange(requestedSharedRanks);
        Displacements sharedRanksDispls(sharedRanksRecvCount);

        lf.setSharedRanks(std::move(sharedRanks), std::move(sharedRanksDispls));
    }

    std::vector<simplex_t> elems_;
    std::unique_ptr<MeshData> vertexData;
    std::unique_ptr<MeshData> elementData;
    MPI_Comm comm;
    bool isPartitionedByHash = false;
    std::vector<std::size_t> vtxdist;
    ntuple_t<global_mesh_ptr, D> boundaryMeshes;
};
} // namespace tndm

#endif // GLOBALSIMPLEXMESH_H
