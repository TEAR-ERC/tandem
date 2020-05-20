#ifndef MESHDATA_H
#define MESHDATA_H

#include "GlobalSimplexMesh.h"
#include "LocalFaces.h"
#include "parallel/CommPattern.h"

#include <mpi.h>

#include <array>
#include <cstddef>

namespace tndm {

template <std::size_t DD> class MeshData {
public:
    MeshData(LocalFaces<DD> const& faces, MPI_Comm comm = MPI_COMM_WORLD)
        : fs(faces), comm(comm), a2a(faces.getRankLayout(), comm) {
        mpi_array_type<int> mpi_vertex_plex_t(1);
        requestedFaces = a2a.exchange(faces.getFaces(), mpi_vertex_plex_t.get());
        a2a.swap();
    }

    template <typename T, std::size_t N, std::size_t D>
    auto getLocalData(std::vector<std::array<T, N>> const& globalData,
                      GlobalSimplexMesh<D> const& globalMesh) const {
        int rank;
        MPI_Comm_rank(comm, &rank);

        std::vector<std::array<T, N>> requestedData;
        requestedData.reserve(requestedFaces.size());
        for (auto& face : requestedFaces) {
            requestedData.emplace_back(globalData[globalMesh.template simplex2Location<DD>(face)]);
        }

        mpi_array_type<T> mpi_t(N);
        return a2a.exchange(requestedData, mpi_t.get());
    }

    auto getSharedRanks() const {
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

        return std::make_pair(std::move(sharedRanks), std::move(sharedRanksDispls));
    }

private:
    LocalFaces<DD> const& fs;
    MPI_Comm comm;
    AllToAllV a2a;
    std::vector<Simplex<DD>> requestedFaces;
};

} // namespace tndm

#endif // MESHDATA_H
