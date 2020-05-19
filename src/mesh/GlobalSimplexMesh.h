#ifndef GLOBALSIMPLEXMESH_H
#define GLOBALSIMPLEXMESH_H

#include <cstddef>
#include <vector>
#include <array>
#include <utility>
#include <set>
#include <numeric>

#include <mpi.h>

#include "parallel/DistributedCSR.h"
#include "parallel/MPITraits.h"
#include "parallel/MetisPartitioner.h"
#include "parallel/CommPattern.h"
#include "LocalSimplexMesh.h"

namespace tndm {

template<std::size_t D, typename RealT, typename IntT>
class GlobalSimplexMesh {
public:
    using vertex_t = std::array<RealT,D>;
    using simplex_t = std::array<IntT,D+1>;

    static_assert(sizeof(vertex_t) == D*sizeof(RealT));
    static_assert(sizeof(simplex_t) == (D+1)*sizeof(IntT));

    GlobalSimplexMesh(std::vector<vertex_t>&& vertices, std::vector<simplex_t>&& elements)
        : verts(std::move(vertices)), elems(std::move(elements)), comm(MPI_COMM_WORLD) {
        int procs;
        MPI_Comm_size(comm, &procs);

        vtxdist = makeDist(numVertices());

    }

    std::size_t numVertices() const { return verts.size(); }
    std::size_t numElements() const { return elems.size(); }

    template<typename OutIntT>
    DistributedCSR<OutIntT> distributedCSR() const {
        DistributedCSR<OutIntT> csr;

        auto elmdist = makeDist(numElements());
        csr.dist.reserve(elmdist.size());
        std::copy(elmdist.begin(), elmdist.end(), csr.dist.begin());

        IntT numElems = numElements();
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
        mpi_array_type<IntT> mpi_simplex_t(D+1);
        elems = a2a.exchange(elemsToSend, mpi_simplex_t.get());
    }

    LocalSimplexMesh<D> getLocalMesh() const {
        using loc_vertex_t = typename LocalSimplexMesh<D>::vertex_t;
        using loc_simplex_t = typename LocalSimplexMesh<D>::simplex_t;

        auto [GIDs, a2a] = getGIDs();
        auto requestedGIDs = a2a.exchange(GIDs);
        a2a.swap();

        auto localVertices = getLocalVertices(requestedGIDs, a2a);
        auto [sharedRanks,sharedRanksLayout] = getSharedRanks(requestedGIDs, a2a);

        auto g2l = makeG2LMap(GIDs);

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
            simplex_t localPlex;
            for (std::size_t d = 0; d < D+1; ++d) {
                assert(g2l.find(elem[d]) != g2l.end());
                assert(g2l[ elem[d] ] < locVerts.size());
                localPlex[d] = g2l[ elem[d] ];
            }
            locElems.emplace_back(loc_simplex_t(localPlex));
        }

        return LocalSimplexMesh<D>(std::move(locVerts), std::move(locElems));
    }

private:
    auto makeG2LMap(std::vector<IntT> const& vGIDs) const {
        std::unordered_map<IntT, IntT> g2l;
        IntT local = 0;
        for (auto& vGID : vGIDs) {
            g2l[vGID] = local++;
        }
        return g2l;
    }

    auto getGIDs() const {
        int procs;
        MPI_Comm_size(comm, &procs);

        std::set<IntT> requiredVertices;
        for (auto& elem : elems) {
            for (auto& v : elem) {
                requiredVertices.insert(v);
            }
        }

        auto rvIt = requiredVertices.begin();
        std::vector<int> requestcounts(procs, 0);
        std::vector<IntT> GIDs;
        GIDs.reserve(requiredVertices.size());
        for (int p = 0; p < procs; ++p) {
            while (*rvIt < vtxdist[p+1] && rvIt != requiredVertices.end()) {
                ++requestcounts[p];
                GIDs.push_back(*rvIt);
                ++rvIt;
            }
        }
        assert(rvIt == requiredVertices.end());

        AllToAllV a2a(std::move(requestcounts));
        return std::make_pair(std::move(GIDs), std::move(a2a));
    }

    auto getLocalVertices(std::vector<IntT> const& requestedGIDs, AllToAllV const& a2a) const {
        int rank;
        MPI_Comm_rank(comm, &rank);

        std::vector<vertex_t> requestedVertices;
        requestedVertices.reserve(requestedGIDs.size());
        for (auto& vGID : requestedGIDs) {
            assert(vGID >= vtxdist[rank] && vGID < vtxdist[rank+1]);
            requestedVertices.emplace_back(verts[vGID-vtxdist[rank]]);
        }

        mpi_array_type<RealT> mpi_vertex_t(D);
        return a2a.exchange(requestedVertices, mpi_vertex_t.get());
    }

    auto getSharedRanks(std::vector<IntT> const& requestedGIDs, AllToAllV const& a2a) const {
        int procs, rank;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &procs);

        std::vector<std::vector<int>> sharedRanksInfo(numVertices());
        for (auto [p,i] : a2a.getSDispls()) {
            assert(requestedGIDs[i] >= vtxdist[rank] && requestedGIDs[i] < vtxdist[rank+1]);
            sharedRanksInfo[requestedGIDs[i]-vtxdist[rank]].emplace_back(p);
        }

        std::vector<int> sharedRanksSendCount;
        sharedRanksSendCount.reserve(requestedGIDs.size());
        std::size_t totalSharedRanksSendCount = 0;
        for (auto& vGID : requestedGIDs) {
            assert(vGID >= vtxdist[rank] && vGID < vtxdist[rank+1]);
            sharedRanksSendCount.emplace_back(sharedRanksInfo[vGID-vtxdist[rank]].size());
            totalSharedRanksSendCount += sharedRanksInfo[vGID-vtxdist[rank]].size();
        }

        auto sharedRanksRecvCount = a2a.exchange(sharedRanksSendCount);

        std::vector<int> requestedSharedRanks(totalSharedRanksSendCount);
        auto requestedSharedRanksIt = requestedSharedRanks.begin();
        for (auto& vGID : requestedGIDs) {
            requestedSharedRanksIt = std::copy(sharedRanksInfo[vGID-vtxdist[rank]].begin(),
                                          sharedRanksInfo[vGID-vtxdist[rank]].end(),
                                          requestedSharedRanksIt);
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

    std::vector<std::size_t> makeDist(std::size_t num) const {
        int procs;
        MPI_Comm_size(comm, &procs);

        std::vector<std::size_t> dist(procs+1);
        dist[0] = 0;
        MPI_Allgather(&num, 1, mpi_type_t<std::size_t>(), dist.data()+1, 1, mpi_type_t<std::size_t>(), MPI_COMM_WORLD);
        for (int p = 1; p < procs+1; ++p) {
            dist[p] += dist[p-1];
        }
        return dist;
    }

    std::vector<vertex_t> verts;
    std::vector<simplex_t> elems;
    MPI_Comm comm;
    std::vector<std::size_t> vtxdist;
};

}


#endif // GLOBALSIMPLEXMESH_H
