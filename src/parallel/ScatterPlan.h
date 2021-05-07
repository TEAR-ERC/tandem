#ifndef SCATTERPLAN_20210325_H
#define SCATTERPLAN_20210325_H

#include "mesh/LocalFaces.h"

#include <mpi.h>

#include <cstddef>
#include <unordered_map>
#include <vector>

namespace tndm {

class ScatterPlan {
public:
    using byte_t = unsigned char;

    struct CommBlock {
        std::size_t offset;
        int count;
        int source_or_dest;
    };

    template <std::size_t D>
    ScatterPlan(LocalFaces<D> const& faces, MPI_Comm comm = MPI_COMM_WORLD) : comm_(comm) {
        int rank;
        MPI_Comm_rank(comm_, &rank);

        std::unordered_map<int, std::vector<std::size_t>> send_map;
        std::unordered_map<int, std::vector<std::size_t>> recv_map;
        for (std::size_t f = 0; f < faces.size(); ++f) {
            auto owner = faces.owner(f);
            if (owner == rank) {
                for (auto&& shRk : faces.getSharedRanks(f)) {
                    send_map[shRk].push_back(f);
                }
            } else {
                recv_map[owner].push_back(f);
            }
        }

        setup(send_map, recv_map);
    }

    ScatterPlan(std::unordered_map<int, std::vector<std::size_t>> const& send_map,
                std::unordered_map<int, std::vector<std::size_t>> const& recv_map,
                MPI_Comm comm = MPI_COMM_WORLD)
        : comm_(comm) {
        setup(send_map, recv_map);
    }

    MPI_Comm comm() const { return comm_; }
    std::vector<std::size_t> const& send_indices() const { return send_indices_; }
    std::vector<std::size_t> const& recv_indices() const { return recv_indices_; }
    std::vector<CommBlock> const& send_blocks() const { return send_blocks_; }
    std::vector<CommBlock> const& recv_blocks() const { return recv_blocks_; }

private:
    void setup(std::unordered_map<int, std::vector<std::size_t>> const& send_map,
               std::unordered_map<int, std::vector<std::size_t>> const& recv_map);

    MPI_Comm comm_;

    std::vector<std::size_t> send_indices_;
    std::vector<std::size_t> recv_indices_;
    std::vector<CommBlock> send_blocks_;
    std::vector<CommBlock> recv_blocks_;
};

} // namespace tndm

#endif // SCATTERPLAN_20210325_H
