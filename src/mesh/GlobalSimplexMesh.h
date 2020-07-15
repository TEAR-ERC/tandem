#ifndef GLOBALSIMPLEXMESH_H
#define GLOBALSIMPLEXMESH_H

#include "LocalFaces.h"
#include "LocalSimplexMesh.h"
#include "MeshData.h"
#include "Simplex.h"

#include "mneme/displacements.hpp"
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
            vtxdist = makeSortedDistribution(vertexData->size(), comm);
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

        auto elmdist = makeSortedDistribution(numElements(), comm);
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
    void repartition();

    /**
     * @brief Partition elements by their hash value (SimplexHash).
     *
     * Should only be used for efficient element data queries. Otherwise use repartition().
     */
    void repartitionByHash();

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

    void doPartition(std::vector<idx_t> const& partition);

    template <std::size_t... Is>
    auto getAllLocalFaces(unsigned overlap, std::index_sequence<Is...>) const {
        auto elemDist = makeSortedDistribution(elems_.size(), comm);
        auto localElems = getGhostElements(elems_, overlap, elemDist);
        return std::make_tuple(getFaces<Is>(localElems.faces(), localElems.localSize())...,
                               std::move(localElems));
    }

    LocalFaces<D> getGhostElements(std::vector<Simplex<D>> elems, unsigned overlap,
                                   std::vector<std::size_t> const& elemDist) const;
    void setSharedRanksAndElementData(LocalFaces<D>& elems,
                                      std::vector<std::size_t> const& elemDist) const;

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

    template <std::size_t DD>
    auto getFaces(std::vector<Simplex<D>> const& elems, std::size_t elemsLocalSize) const {
        auto plex2rank = getPlex2Rank<DD>();

        int rank, procs;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &procs);

        std::vector<std::set<Simplex<DD>>> requiredFaces(procs);
        std::unordered_map<Simplex<DD>, std::size_t, SimplexHash<DD>> faceOrder;
        std::size_t faceNo = 0;
        std::size_t localSize = 0;
        for (std::size_t elNo = 0; elNo < elems.size(); ++elNo) {
            auto downward = elems[elNo].template downward<DD>();
            for (auto& s : downward) {
                requiredFaces[plex2rank(s)].insert(s);
                if (faceOrder.find(s) == faceOrder.end()) {
                    faceOrder[s] = faceNo++;
                }
            }
            if (elNo < elemsLocalSize) {
                localSize = faceNo;
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
        std::vector<int> owner;
        faces.reserve(total);
        owner.reserve(total);
        for (int p = 0; p < procs; ++p) {
            std::copy(requiredFaces[p].begin(), requiredFaces[p].end(), std::back_inserter(faces));
            std::fill_n(std::back_inserter(owner), requiredFaces[p].size(), p);
        }

        // Exchange data
        AllToAllV a2a(std::move(counts), comm);
        mpi_array_type<uint64_t> mpi_plex_t(DD + 1);
        auto requestedFaces = a2a.exchange(faces, mpi_plex_t.get());
        a2a.swap();

        // Create local faces
        auto contiguousGIDs = getContiguousGIDs(requestedFaces, a2a);
        auto lf = LocalFaces<DD>(std::move(faces), std::move(owner), std::move(contiguousGIDs),
                                 localSize);

        if constexpr (DD == 0) {
            if (vertexData) {
                std::vector<std::size_t> lids;
                lids.reserve(requestedFaces.size());
                for (auto& face : requestedFaces) {
                    lids.emplace_back(getVertexLID(face));
                }
                auto meshData = vertexData->redistributed(lids, a2a);
                lf.setMeshData(std::move(meshData));
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
                auto meshData = boundaryMesh->elementData->redistributed(lids, a2a);
                lf.setMeshData(std::move(meshData));
            }
        }

        auto [sharedRanks, sharedRanksDispls] = getSharedRanks(requestedFaces, a2a);
        lf.setSharedRanks(std::move(sharedRanks), std::move(sharedRanksDispls));

        // Sort faces
        std::vector<std::size_t> permutation(lf.faces().size());
        std::iota(permutation.begin(), permutation.end(), std::size_t(0));
        std::sort(permutation.begin(), permutation.end(),
                  [&lf, &faceOrder](std::size_t a, std::size_t b) {
                      return faceOrder[lf.faces()[a]] < faceOrder[lf.faces()[b]];
                  });
        lf.permute(permutation);

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
    auto getSharedRanks(std::vector<Simplex<DD>> const& requestedFaces,
                        AllToAllV const& a2a) const {
        int procs;
        MPI_Comm_size(comm, &procs);

        std::unordered_map<Simplex<DD>, std::vector<int>, SimplexHash<DD>> sharedRanksInfo;
        for (int p = 0; p < procs; ++p) {
            for (auto&& i : a2a.sendRange(p)) {
                sharedRanksInfo[requestedFaces[i]].emplace_back(p);
            }
        }

        std::vector<int> sharedRanksSendCount;
        sharedRanksSendCount.reserve(requestedFaces.size());
        std::size_t totalSharedRanksSendCount = 0;
        for (auto& face : requestedFaces) {
            auto size = sharedRanksInfo[face].size();
            assert(size > 0);
            size -= 1;
            sharedRanksSendCount.emplace_back(size);
            totalSharedRanksSendCount += size;
        }

        auto sharedRanksRecvCount = a2a.exchange(sharedRanksSendCount);

        std::vector<int> requestedSharedRanks;
        requestedSharedRanks.reserve(totalSharedRanksSendCount);
        for (int p = 0; p < procs; ++p) {
            for (auto&& i : a2a.sendRange(p)) {
                for (auto&& shRk : sharedRanksInfo[requestedFaces[i]]) {
                    if (shRk != p) {
                        requestedSharedRanks.push_back(shRk);
                    }
                }
            }
        }
        std::vector<int> sendcounts(procs, 0);
        std::vector<int> recvcounts(procs, 0);
        for (int p = 0; p < procs; ++p) {
            for (auto&& i : a2a.sendRange(p)) {
                sendcounts[p] += sharedRanksSendCount[i];
            }
        }
        for (int p = 0; p < procs; ++p) {
            for (auto&& i : a2a.recvRange(p)) {
                recvcounts[p] += sharedRanksRecvCount[i];
            }
        }

        AllToAllV a2aSharedRanks(std::move(sendcounts), std::move(recvcounts));
        auto sharedRanks = a2aSharedRanks.exchange(requestedSharedRanks);
        Displacements sharedRanksDispls(sharedRanksRecvCount);

        return std::make_pair(std::move(sharedRanks), std::move(sharedRanksDispls));
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
