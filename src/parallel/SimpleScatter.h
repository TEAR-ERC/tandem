#ifndef SIMPLESCATTER_20210325_H
#define SIMPLESCATTER_20210325_H

#include "parallel/MPITraits.h"
#include "parallel/ScatterPlan.h"

#include <mpi.h>

#include <cassert>
#include <cstddef>
#include <memory>
#include <vector>

namespace tndm {

template <typename T> class SimpleScatter {
public:
    SimpleScatter(std::shared_ptr<ScatterPlan> topo, std::size_t block_size = 1,
                  MPI_Datatype const& mpiType = mpi_type_t<T>())
        : topo_(std::move(topo)), block_size_(block_size), mpi_type_(mpiType) {
        requests_.resize(topo_->recv_blocks().size() + topo_->send_blocks().size());
        send_buffer_.resize(topo_->send_indices().size() * block_size);
        recv_buffer_.resize(topo_->recv_indices().size() * block_size);
    }

    void scatter(T* data) {
        std::size_t requestNo = 0;
        for (auto const& block : topo_->recv_blocks()) {
            int size = block.count * block_size_;
            MPI_Irecv(&recv_buffer_[block.offset * block_size_], size, mpi_type_,
                      block.source_or_dest, 0, topo_->comm(), &requests_[requestNo++]);
        }
        for (std::size_t i = 0; i < topo_->send_indices().size(); ++i) {
            T* begin = data + topo_->send_indices()[i] * block_size_;
            std::copy(begin, begin + block_size_, send_buffer_.begin() + i * block_size_);
        }
        for (auto const& block : topo_->send_blocks()) {
            int size = block.count * block_size_;
            MPI_Isend(&send_buffer_[block.offset * block_size_], size, mpi_type_,
                      block.source_or_dest, 0, topo_->comm(), &requests_[requestNo++]);
        }

        assert(requestNo == requests_.size());

        MPI_Waitall(requests_.size(), requests_.data(), MPI_STATUSES_IGNORE);
        for (std::size_t i = 0; i < topo_->recv_indices().size(); ++i) {
            auto in_begin = recv_buffer_.begin() + i * block_size_;
            T* out_begin = data + topo_->recv_indices()[i] * block_size_;
            std::copy(in_begin, in_begin + block_size_, out_begin);
        }
    }

private:
    std::shared_ptr<ScatterPlan> topo_;
    std::size_t block_size_;
    MPI_Datatype mpi_type_;

    std::vector<MPI_Request> requests_;
    std::vector<T> send_buffer_;
    std::vector<T> recv_buffer_;
};

} // namespace tndm

#endif // SIMPLESCATTER_20210325_H
