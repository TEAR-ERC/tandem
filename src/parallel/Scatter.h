#ifndef SCATTER_20200715_H
#define SCATTER_20200715_H

#include "mesh/LocalFaces.h"
#include "parallel/MPITraits.h"

#include <mpi.h>

#include <cstddef>
#include <unordered_map>
#include <vector>

namespace tndm {

class Scatter {
public:
    using byte_t = unsigned char;

    template <std::size_t D>
    Scatter(LocalFaces<D> const& faces, MPI_Comm comm = MPI_COMM_WORLD) : comm_(comm) {
        int rank;
        MPI_Comm_rank(comm_, &rank);

        for (std::size_t f = 0; f < faces.size(); ++f) {
            auto owner = faces.owner(f);
            if (owner == rank) {
                for (auto&& shRk : faces.getSharedRanks(f)) {
                    sendMap_[shRk].push_back(f);
                }
            } else {
                recvMap_[owner].push_back(f);
            }
        }

        sendSize_ = 0;
        recvSize_ = 0;
        for (auto& [key, value] : sendMap_) {
            sendSize_ += value.size();
        }
        for (auto& [key, value] : recvMap_) {
            recvSize_ += value.size();
        }

        requests_.resize(sendMap_.size() + recvMap_.size());
    }

    template <typename T> void scatter(T* data, MPI_Datatype const& mpiType = mpi_type_t<T>()) {
        auto const resizeIfNecessary = [](std::vector<byte_t>& buffer, std::size_t size) {
            std::size_t requiredSize = sizeof(T) * size;
            if (requiredSize > buffer.size()) {
                buffer.resize(requiredSize);
            }
        };
        resizeIfNecessary(sendBuffer_, sendSize_);
        resizeIfNecessary(recvBuffer_, recvSize_);
        T* sendBuf = reinterpret_cast<T*>(sendBuffer_.data());
        T* recvBuf = reinterpret_cast<T*>(recvBuffer_.data());

        std::size_t offset = 0;
        std::size_t requestNo = 0;
        for (auto& [key, value] : recvMap_) {
            MPI_Irecv(&recvBuf[offset], value.size(), mpiType, key, 0, comm_,
                      &requests_[requestNo++]);
            offset += value.size();
        }
        offset = 0;
        for (auto& [key, value] : sendMap_) {
            for (auto&& v : value) {
                sendBuf[offset++] = data[v];
            }
            MPI_Isend(&sendBuf[offset - value.size()], value.size(), mpiType, key, 0, comm_,
                      &requests_[requestNo++]);
        }
        MPI_Waitall(requestNo, requests_.data(), MPI_STATUSES_IGNORE);
        offset = 0;
        for (auto& [key, value] : recvMap_) {
            for (auto&& v : value) {
                data[v] = recvBuf[offset++];
            }
        }
    }

private:
    MPI_Comm comm_;
    std::unordered_map<int, std::vector<std::size_t>> sendMap_;
    std::unordered_map<int, std::vector<std::size_t>> recvMap_;
    std::size_t sendSize_ = 0, recvSize_ = 0;

    std::vector<MPI_Request> requests_;

    std::vector<byte_t> sendBuffer_;
    std::vector<byte_t> recvBuffer_;
};

} // namespace tndm

#endif // SCATTER_20200715_H
