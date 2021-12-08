#ifndef COMMPATTERN_H
#define COMMPATTERN_H

#include "MPITraits.h"
#include "tensor/Managed.h"
#include "tensor/Tensor.h"
#include "util/Range.h"

#include <functional>
#include <mpi.h>
#include <numeric>
#include <vector>

namespace tndm {

void make_displs(
    std::vector<int> const& counts, std::vector<int>& displs,
    std::function<int(int)> permutation = [](int p) { return p; });

class AllToAllV {
public:
    template <typename Tr>
    using tensor_ret_t =
        Managed<Tensor<typename detail::traits<Tr>::real_t, detail::traits<Tr>::Dim>>;

    AllToAllV(std::vector<int>&& sndcnts, MPI_Comm comm = MPI_COMM_WORLD);
    AllToAllV(std::vector<int>&& sndcnts, std::vector<int>&& recvcnts,
              MPI_Comm comm = MPI_COMM_WORLD);

    void swap();
    void setRankPermutation(std::vector<int> const& permutation);

    auto recvRange(int p) const { return Range(rdispls[p], rdispls[p] + recvcounts[p]); }
    auto sendRange(int p) const { return Range(sdispls[p], sdispls[p] + sendcounts[p]); }

    template <typename T>
    [[nodiscard]] std::vector<T> exchange(std::vector<T> const& dataToSend,
                                          MPI_Datatype const& mpiType = mpi_type_t<T>()) const {
        std::vector<T> recvdData(std::accumulate(recvcounts.begin(), recvcounts.end(), 0));
        MPI_Alltoallv(dataToSend.data(), sendcounts.data(), sdispls.data(), mpiType,
                      recvdData.data(), recvcounts.data(), rdispls.data(), mpiType, comm);
        return recvdData;
    }

    template <typename Tr>
    [[nodiscard]] auto exchange(Tr const& dataToSend) const -> tensor_ret_t<Tr> {
        auto shape = dataToSend.shape();
        shape.back() = std::accumulate(recvcounts.begin(), recvcounts.end(), 0);
        auto recvdData = tensor_ret_t<Tr>(shape);

        std::size_t numReals = 1;
        for (std::size_t i = 0; i < shape.size() - 1; ++i) {
            numReals *= shape[i];
        }

        auto mpiType = mpi_array_type<typename Tr::real_t>(numReals);
        MPI_Alltoallv(dataToSend.data(), sendcounts.data(), sdispls.data(), mpiType.get(),
                      recvdData.data(), recvcounts.data(), rdispls.data(), mpiType.get(), comm);
        return recvdData;
    }

private:
    std::vector<int> sendcounts;
    MPI_Comm comm;
    int procs;
    std::vector<int> recvcounts;
    std::vector<int> sdispls;
    std::vector<int> rdispls;
};

class GatherV {
public:
    GatherV(int sendcount, int root = 0, MPI_Comm comm = MPI_COMM_WORLD);

    template <typename T>
    [[nodiscard]] std::vector<T> exchange(T const* data,
                                          MPI_Datatype const& type = mpi_type_t<T>()) const {
        std::vector<T> recvd(recvcount());
        exchange(data, recvd.data(), type);
        return recvd;
    }

    template <typename T>
    void exchange(T const* data, T* recvd, MPI_Datatype const& type = mpi_type_t<T>()) const {
        MPI_Gatherv(data, sendcount_, type, recvd, recvcounts_.data(), displs_.data(), type, root_,
                    comm_);
    }

    inline std::size_t recvcount() const {
        return std::accumulate(recvcounts_.begin(), recvcounts_.end(), static_cast<std::size_t>(0));
    }

private:
    int sendcount_;
    int root_;
    MPI_Comm comm_;
    std::vector<int> recvcounts_;
    std::vector<int> displs_;
};

} // namespace tndm

#endif // COMMPATTERN_H
