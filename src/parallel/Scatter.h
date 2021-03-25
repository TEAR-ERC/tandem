#ifndef SCATTER_20200715_H
#define SCATTER_20200715_H

#include "parallel/MPITraits.h"
#include "parallel/ScatterPlan.h"
#include "parallel/SparseBlockVector.h"
#include "tensor/Tensor.h"

#include <mpi.h>

#include <cassert>
#include <cstddef>
#include <memory>
#include <vector>

namespace tndm {

class Scatter {
public:
    using byte_t = unsigned char;

    Scatter(std::shared_ptr<ScatterPlan> topo) : topo_(std::move(topo)) {
        requests_.resize(topo_->recv_blocks().size() + topo_->send_blocks().size());
    }

    template <typename T>
    auto recv_prototype(std::size_t block_size,
                        std::size_t alignment = SparseBlockVector<T>::DefaultAlignment) const {
        return SparseBlockVector<T>(topo_->recv_indices(), block_size, alignment);
    }

    template <typename BlockVector, typename T>
    void begin_scatter(BlockVector const& x, SparseBlockVector<T>& y) {
        static_assert(std::is_same_v<typename BlockVector::value_type, T>,
                      "Basic type of x and y must match");
        assert(x.block_size() == y.block_size());

        const auto mpiType = mpi_type_t<T>();

        std::size_t bs = x.block_size();
        auto const resizeIfNecessary = [](std::vector<byte_t>& buffer, std::size_t size) {
            std::size_t requiredSize = sizeof(T) * size;
            if (requiredSize > buffer.size()) {
                buffer.resize(requiredSize);
            }
        };
        resizeIfNecessary(send_buffer_, bs * topo_->send_indices().size());
        T* sendBuf = reinterpret_cast<T*>(send_buffer_.data());

        std::size_t requestNo = 0;
        for (auto const& block : topo_->recv_blocks()) {
            int size = block.count * bs;
            MPI_Irecv(&y.data()[block.offset * bs], size, mpiType, block.source_or_dest, 0,
                      topo_->comm(), &requests_[requestNo++]);
        }

        auto x_handle = x.begin_access_readonly();
        for (std::size_t i = 0; i < topo_->send_indices().size(); ++i) {
            auto idx = topo_->send_indices()[i];
            auto block = x.get_block(x_handle, idx);
            memcpy(&sendBuf[i * bs], block.data(), bs * sizeof(T));
        }
        x.end_access_readonly(x_handle);
        for (auto const& block : topo_->send_blocks()) {
            int size = block.count * bs;
            MPI_Isend(&sendBuf[block.offset * bs], size, mpiType, block.source_or_dest, 0,
                      topo_->comm(), &requests_[requestNo++]);
        }

        assert(requestNo == requests_.size());
    }

    void wait_scatter() { MPI_Waitall(requests_.size(), requests_.data(), MPI_STATUSES_IGNORE); }

private:
    std::shared_ptr<ScatterPlan> topo_;

    std::vector<MPI_Request> requests_;
    std::vector<byte_t> send_buffer_;
};

} // namespace tndm

#endif // SCATTER_20200715_H
