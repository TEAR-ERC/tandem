#ifndef GLOBALSIMPLEXMESH_H
#define GLOBALSIMPLEXMESH_H

#include "LocalFaces.h"
#include "LocalSimplexMesh.h"
#include "MeshData.h"
#include "Simplex.h"

#include "parallel/CommPattern.h"
#include "parallel/DistributedCSR.h"
#include "parallel/MPITraits.h"
#include "parallel/SortedDistribution.h"
#include "util/Range.h"
#include "util/Utility.h"
#include "mesh/MultiplyBoundaryTags.h"
#include <metis.h>
#include <mpi.h>
#include <cstring> // For std::memcpy

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
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

    void setBoundaryTags(std::vector<globalBoundaryTag<D>> boundaryTagsSent){
        boundaryTags=std::move(boundaryTagsSent);
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


        
        auto mesh=std::make_unique<LocalSimplexMesh<D>>(std::move(localFaces));
        auto tempMesh=mesh.get();
        const auto& facets = tempMesh->facets();
        //auto gl2=facets.g2l();
        auto faultTags=returnFaultTypeBoundaries(boundaryTags);  //only treat faults - later can add direclhet 
        auto faultsTagsInLocalMesh=getLocalBoundaryTagFromGlobal( faultTags , facets); // here is where I do the mapping between global and local
        mesh->saveFaultTags(faultsTagsInLocalMesh);
        return mesh;
    }



void AllGatherBoundaryTags() {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Serialize BoundaryTags
    std::vector<char> serializedTags;
    serializeBoundaryTags(boundaryTags, serializedTags);

    // Determine the size of serialized data from each process
    int localSize = serializedTags.size();
    std::vector<int> allSizes(size);
    MPI_Allgather(&localSize, 1, MPI_INT, allSizes.data(), 1, MPI_INT, comm);

    // Determine the displacements for the gathered data
    std::vector<int> displs(size, 0);
    for (int i = 1; i < size; ++i) {
        displs[i] = displs[i - 1] + allSizes[i - 1];
    }

    // Allocate buffer for all gathered data
    std::vector<char> allSerializedTags(displs.back() + allSizes.back());
    MPI_Allgatherv(serializedTags.data(), localSize, MPI_BYTE, allSerializedTags.data(),
                   allSizes.data(), displs.data(), MPI_BYTE, comm);

    // Deserialize gathered data
    boundaryTags.clear();
    deserializeBoundaryTags(allSerializedTags, boundaryTags);
}

private:



void serializeBoundaryTags(const std::vector<globalBoundaryTag<D>>& tags, std::vector<char>& buffer) {
    size_t totalSize = 0;
    for (const auto& tag : tags) {
        totalSize += sizeof(tag.tagID);
        totalSize += sizeof(tag.dimension);
        totalSize += sizeof(tag.boundaryType);
        totalSize += sizeof(size_t) + tag.tagLabel.size();
        totalSize += sizeof(size_t) + tag.elementBoundary.size() * sizeof(Simplex<D - 1>);
    }
    buffer.resize(totalSize);

    char* ptr = buffer.data();
    for (const auto& tag : tags) {
        std::memcpy(ptr, &tag.tagID, sizeof(tag.tagID));
        ptr += sizeof(tag.tagID);

        std::memcpy(ptr, &tag.dimension, sizeof(tag.dimension));
        ptr += sizeof(tag.dimension);

        std::memcpy(ptr, &tag.boundaryType, sizeof(tag.boundaryType));
        ptr += sizeof(tag.boundaryType);

        size_t labelSize = tag.tagLabel.size();
        std::memcpy(ptr, &labelSize, sizeof(size_t));
        ptr += sizeof(size_t);
        std::memcpy(ptr, tag.tagLabel.data(), labelSize);
        ptr += labelSize;

        size_t elementBoundarySize = tag.elementBoundary.size();
        std::memcpy(ptr, &elementBoundarySize, sizeof(size_t));
        ptr += sizeof(size_t);
        std::memcpy(ptr, tag.elementBoundary.data(), elementBoundarySize * sizeof(Simplex<D - 1>));
        ptr += elementBoundarySize * sizeof(Simplex<D - 1>);
    }
}



void deserializeBoundaryTags(const std::vector<char>& buffer, std::vector<globalBoundaryTag<D>>& tags) {
    const char* ptr = buffer.data();
    while (ptr < buffer.data() + buffer.size()) {
        std::string tagLabel;
        long tagID;
        int dimension;
        BC boundaryType;

        std::memcpy(&tagID, ptr, sizeof(tagID));
        ptr += sizeof(tagID);

        std::memcpy(&dimension, ptr, sizeof(dimension));
        ptr += sizeof(dimension);

        std::memcpy(&boundaryType, ptr, sizeof(boundaryType));
        ptr += sizeof(boundaryType);

        size_t labelSize;
        std::memcpy(&labelSize, ptr, sizeof(size_t));
        ptr += sizeof(size_t);
        tagLabel.assign(ptr, labelSize);
        ptr += labelSize;

        size_t elementBoundarySize;
        std::memcpy(&elementBoundarySize, ptr, sizeof(size_t));
        ptr += sizeof(size_t);
        std::vector<Simplex<D - 1>> elementBoundary(elementBoundarySize);
        std::memcpy(elementBoundary.data(), ptr, elementBoundarySize * sizeof(Simplex<D - 1>));
        ptr += elementBoundarySize * sizeof(Simplex<D - 1>);

        globalBoundaryTag<D> tag(tagLabel, tagID, dimension, boundaryType);
        tag.elementBoundary = std::move(elementBoundary);
        tags.push_back(std::move(tag));
    }
}




    template <std::size_t DD> friend class GlobalSimplexMesh;
    using upward_map_t = std::unordered_multimap<Simplex<D - 1u>, std::size_t, SimplexHash<D - 1u>>;
    using facet_set_t = std::unordered_set<Simplex<D - 1u>, SimplexHash<D - 1u>>;

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

    auto makeUpwardMap(std::vector<Simplex<D>> const& elems) const {
        // Construct upward map from faces to local element ids
        upward_map_t up;
        for (std::size_t elNo = 0; elNo < elems.size(); ++elNo) {
            auto downward = elems[elNo].template downward<D - 1u>();
            for (auto& s : downward) {
                up.emplace(s, elNo);
            }
        }
        return up;
    }
    auto getBoundaryFaces(upward_map_t const& up) const {
        facet_set_t boundaryFaces;
        for (auto&& it : up) {
            auto count = up.count(it.first);
            assert(1 <= count && count <= 2);
            if (count == 1) {
                boundaryFaces.insert(it.first);
            }
        }
        return boundaryFaces;
    }
    void deleteDomainBoundaryFaces(facet_set_t& boundaryFaces) const;

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
                assert(plex2rank(s) < procs);
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
    auto getSharedRanks(std::vector<Simplex<DD>> const& requestedFaces, AllToAllV const& a2a,
                        std::vector<int> const* rankPermutation = nullptr) const {
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
            int pp = rankPermutation ? (*rankPermutation)[p] : p;
            for (auto&& i : a2a.sendRange(pp)) {
                for (auto&& shRk : sharedRanksInfo[requestedFaces[i]]) {
                    if (shRk != pp) {
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
        if (rankPermutation) {
            a2aSharedRanks.setRankPermutation(*rankPermutation);
        }
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
    std::vector<globalBoundaryTag<D>> boundaryTags;
};
} // namespace tndm

#endif // GLOBALSIMPLEXMESH_H
