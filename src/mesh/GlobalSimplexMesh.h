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
#include "util/Utility.h"

#include <mpi.h>

#include <array>
#include <cstddef>
#include <functional>
#include <iterator>
#include <memory>
#include <numeric>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tndm {

template <std::size_t D> class GlobalSimplexMesh {
public:
    using simplex_t = Simplex<D>;
    static_assert(sizeof(simplex_t) == (D + 1) * sizeof(uint64_t));

    GlobalSimplexMesh(std::vector<simplex_t>&& elements, std::unique_ptr<MeshData> vertexDat,
                      std::unique_ptr<MeshData> elementDat = nullptr,
                      MPI_Comm comm = MPI_COMM_WORLD)
        : elems(std::move(elements)), vertexData(std::move(vertexDat)),
          elementData(std::move(elementDat)), comm(comm) {
        assert(vertexData != nullptr);
        vtxdist = makeSortedDistribution(vertexData->size());
    }

    std::size_t numElements() const { return elems.size(); }

    template<typename OutIntT>
    DistributedCSR<OutIntT> distributedCSR() const {
        DistributedCSR<OutIntT> csr;

        auto elmdist = makeSortedDistribution(numElements());
        csr.dist.reserve(elmdist.size());
        std::copy(elmdist.begin(), elmdist.end(), csr.dist.begin());

        auto numElems = numElements();
        csr.rowPtr.resize(numElems+1);
        csr.colInd.resize(numElems*(D+1));

        OutIntT ind = 0;
        OutIntT ptr = 0;
        for (auto& e : elems) {
            csr.rowPtr[ptr++] = ind;
            for (auto& p : e) {
                csr.colInd[ind++] = p;
            }
        }
        csr.rowPtr.back() = ind;

        return csr;
    }

    void repartition() {
        int procs, rank;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &procs);

        auto distCSR = distributedCSR<idx_t>();
        auto partition = MetisPartitioner::partition(distCSR, D);
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
                elemsToSend.emplace_back(elems[*eIt]);
                ++eIt;
            }
        }
        assert(eIt == enumeration.end());

        AllToAllV a2a(std::move(sendcounts), comm);
        mpi_array_type<uint64_t> mpi_simplex_t(D + 1);
        elems = a2a.exchange(elemsToSend, mpi_simplex_t.get());

        if (elementData) {
            elementData = elementData->redistributed(enumeration, a2a);
        }
    }

    LocalSimplexMesh<D> getLocalMesh() const {
        auto localFaces = getAllLocalFaces(std::make_index_sequence<D>{});

        return LocalSimplexMesh<D>(std::move(localFaces));
    }

private:
    template <std::size_t... Is> auto getAllLocalFaces(std::index_sequence<Is...>) const {
        std::vector<Simplex<D>> elemsCopy(elems);
        return std::make_tuple(getFaces<Is>()..., LocalFaces<D>(std::move(elemsCopy)));
    }

    template <std::size_t DD> auto getPlex2Rank() const {
        if constexpr (DD == 0) {
            SortedDistributionToRank v2r(vtxdist);
            return [v2r](Simplex<0> const& plex) { return v2r(plex[0]); };
        } else {
            int procs;
            MPI_Comm_size(comm, &procs);
            return [procs](Simplex<DD> const& plex) { return SimplexHash<DD>()(plex) % procs; };
        }
    }

    std::size_t getVertexLID(Simplex<0> const& plex) const {
        int rank;
        MPI_Comm_rank(comm, &rank);
        assert(plex[0] >= vtxdist[rank] && plex[0] < vtxdist[rank + 1]);
        return plex[0] - vtxdist[rank];
    }

    template <std::size_t DD> auto getFaces() const {
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

        auto lf = LocalFaces<DD>(std::move(faces));
        if constexpr (DD == 0) {
            if (vertexData) {
                std::vector<std::size_t> lids;
                lids.reserve(requestedFaces.size());
                for (auto& face : requestedFaces) {
                    lids.emplace_back(getVertexLID(face));
                }
                lf.setMeshData(vertexData->redistributed(lids, a2a));
            }
        }

        getSharedRanks(lf, requestedFaces, a2a);
        return lf;
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

    std::vector<simplex_t> elems;
    std::unique_ptr<MeshData> vertexData;
    std::unique_ptr<MeshData> elementData;
    MPI_Comm comm;
    std::vector<std::size_t> vtxdist;
};
}


#endif // GLOBALSIMPLEXMESH_H
