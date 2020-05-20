#ifndef GLOBALSIMPLEXMESH_H
#define GLOBALSIMPLEXMESH_H

#include "LocalSimplexMesh.h"
#include "Simplex.h"

#include "parallel/CommPattern.h"
#include "parallel/DistributedCSR.h"
#include "parallel/MPITraits.h"
#include "parallel/MetisPartitioner.h"
#include "parallel/SortedDistribution.h"

#include <mpi.h>

#include <array>
#include <cstddef>
#include <functional>
#include <iterator>
#include <numeric>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tndm {

template<std::size_t D, typename RealT>
class GlobalSimplexMesh {
public:
    using vertex_t = std::array<RealT,D>;
    using simplex_t = Simplex<D>;

    static_assert(sizeof(vertex_t) == D*sizeof(RealT));
    static_assert(sizeof(simplex_t) == (D+1)*sizeof(int));

    GlobalSimplexMesh(std::vector<vertex_t>&& vertices, std::vector<simplex_t>&& elements)
        : verts(std::move(vertices)), elems(std::move(elements)), comm(MPI_COMM_WORLD) {
        int procs;
        MPI_Comm_size(comm, &procs);

        vtxdist = makeSortedDistribution(numVertices());
    }

    std::size_t numVertices() const { return verts.size(); }
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

        std::vector<idx_t> enumeration(partition.size());
        std::iota(enumeration.begin(), enumeration.end(), 0);

        std::sort(enumeration.begin(), enumeration.end(), [&partition](auto a, auto b) {
            return partition[a] < partition[b];
        });

        std::vector<int> sendcounts(procs, 0);
        std::vector<simplex_t> elemsToSend;
        elemsToSend.reserve(numElements());
        auto eIt = enumeration.begin();
        for (int p = 0; p < procs; ++p) {
            while (partition[*eIt] <= p && eIt != enumeration.end()) {
                ++sendcounts[p];
                elemsToSend.emplace_back(elems[*eIt]);
                ++eIt;
            }
        }
        assert(eIt == enumeration.end());

        AllToAllV a2a(std::move(sendcounts));
        mpi_array_type<int> mpi_simplex_t(D+1);
        elems = a2a.exchange(elemsToSend, mpi_simplex_t.get());
    }

    LocalSimplexMesh<D> getLocalMesh() const {
        auto localFaces = getAllLocalFaces(std::make_index_sequence<D>{});

        /*auto g2l = makeG2LMap(GIDs);
        auto localVertices = getLocalVertices(requestedGIDs, a2a);
        std::vector<loc_vertex_t> locVerts(localVertices.size());
        auto locVertsIt = locVerts.begin();
        for (auto& vert : localVertices) {
            for (std::size_t d = 0; d < D; ++d) {
                (*locVertsIt)[d] = vert[d];
            }
            ++locVertsIt;
        }

        std::vector<loc_simplex_t> locElems;
        locElems.reserve(numElements());
        for (auto& elem : elems) {
            loc_simplex_t localPlex;
            for (std::size_t d = 0; d < D+1; ++d) {
                assert(g2l.find(elem[d]) != g2l.end());
                assert(g2l[ elem[d] ] < locVerts.size());
                localPlex[d] = g2l[ elem[d] ];
            }
            locElems.emplace_back(std::move(localPlex));
        }*/

        return LocalSimplexMesh<D>(std::move(localFaces));
    }

private:
    template <std::size_t... Is> auto getAllLocalFaces(std::index_sequence<Is...>) const {
        std::vector<Simplex<D>> elemsCopy(elems);
        return std::make_tuple(getLocalFaces<Is>()..., LocalFaces(std::move(elemsCopy)));
    }

    template <std::size_t DD> auto getPlex2Rank() const {
        if constexpr (DD == 0) {
            SortedDistributionToRank v2r(vtxdist);
            return [v2r](Simplex<0> const& plex) { return v2r(plex[0]); };
        } else {
            return SimplexHash<DD>();
        }
    }

    template <std::size_t DD> auto getLocalFaces() const {
        auto plex2rank = getPlex2Rank<DD>();

        auto [faces, a2a] = getFaces<DD>(plex2rank);

        mpi_array_type<int> mpi_vertex_plex_t(1);
        auto requestedFaces = a2a.exchange(faces, mpi_vertex_plex_t.get());
        a2a.swap();

        auto [sharedRanks, sharedRanksLayout] = getSharedRanks<DD>(requestedFaces, a2a);

        LocalFaces<DD> lf(std::move(faces));
        lf.setSharedRanks(std::move(sharedRanks), std::move(sharedRanksLayout));
        return lf;
    }

    template <std::size_t DD>
    auto getFaces(std::function<int(Simplex<DD> const&)> const& plex2rank) const {
        int procs;
        MPI_Comm_size(comm, &procs);

        std::vector<std::set<Simplex<DD>>> requiredSimplices(procs);
        for (auto& elem : elems) {
            for (auto&& s : elem.template downward<DD>()) {
                requiredSimplices[plex2rank(s)].insert(s);
            }
        }
        std::vector<int> requestcounts(procs, 0);
        std::size_t total = 0;
        for (int p = 0; p < procs; ++p) {
            auto size = requiredSimplices[p].size();
            requestcounts[p] = size;
            total += size;
        }

        std::vector<Simplex<DD>> GIDs;
        GIDs.reserve(total);
        for (auto& perRank : requiredSimplices) {
            std::copy(perRank.begin(), perRank.end(), std::back_inserter(GIDs));
        }

        AllToAllV a2a(std::move(requestcounts));
        return std::make_pair(std::move(GIDs), std::move(a2a));
    }

    auto getLocalVertices(std::vector<Simplex<0>> const& requestedGIDs, AllToAllV const& a2a) const {
        int rank;
        MPI_Comm_rank(comm, &rank);

        std::vector<vertex_t> requestedVertices;
        requestedVertices.reserve(requestedGIDs.size());
        for (auto& vGID : requestedGIDs) {
            if (!(vGID[0] >= vtxdist[rank] && vGID[0] < vtxdist[rank+1]))
            std::cout << rank << " " << vGID[0] << " " << vtxdist[rank] << " " << vtxdist[rank+1] << std::endl;
            assert(vGID[0] >= vtxdist[rank] && vGID[0] < vtxdist[rank+1]);
            requestedVertices.emplace_back(verts[vGID[0]-vtxdist[rank]]);
        }

        mpi_array_type<RealT> mpi_vertex_t(D);
        return a2a.exchange(requestedVertices, mpi_vertex_t.get());
    }

    template <std::size_t DD>
    auto getSharedRanks(std::vector<Simplex<DD>> const& requestedGIDs, AllToAllV const& a2a) const {
        int procs, rank;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &procs);

        std::unordered_map<Simplex<DD>, std::vector<int>, SimplexHash<DD>> sharedRanksInfo;
        for (auto [p,i] : a2a.getSDispls()) {
            sharedRanksInfo[requestedGIDs[i]].emplace_back(p);
        }

        std::vector<int> sharedRanksSendCount;
        sharedRanksSendCount.reserve(requestedGIDs.size());
        std::size_t totalSharedRanksSendCount = 0;
        for (auto& vGID : requestedGIDs) {
            sharedRanksSendCount.emplace_back(sharedRanksInfo[vGID].size());
            totalSharedRanksSendCount += sharedRanksInfo[vGID].size();
        }

        auto sharedRanksRecvCount = a2a.exchange(sharedRanksSendCount);

        std::vector<int> requestedSharedRanks;
        requestedSharedRanks.reserve(totalSharedRanksSendCount);
        for (auto& vGID : requestedGIDs) {
            std::copy(sharedRanksInfo[vGID].begin(), sharedRanksInfo[vGID].end(),
                      std::back_inserter(requestedSharedRanks));
        }
        std::vector<int> sendcounts(procs, 0);
        std::vector<int> recvcounts(procs, 0);
        for (auto [p,i] : a2a.getSDispls()) {
            sendcounts[p] += sharedRanksSendCount[i];
        }
        for (auto [p,i] : a2a.getRDispls()) {
            recvcounts[p] += sharedRanksRecvCount[i];
        }

        AllToAllV a2aSharedRanks(std::move(sendcounts), std::move(recvcounts));
        auto sharedRanks = a2aSharedRanks.exchange(requestedSharedRanks);
        Displacements sharedRanksDispls(sharedRanksRecvCount);

        return std::make_pair(std::move(sharedRanks), std::move(sharedRanksDispls));
    }

    std::vector<vertex_t> verts;
    std::vector<simplex_t> elems;
    MPI_Comm comm;
    std::vector<std::size_t> vtxdist;
};

}


#endif // GLOBALSIMPLEXMESH_H
