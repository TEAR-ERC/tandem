#ifndef DG_20200615_H
#define DG_20200615_H

#include "form/RefElement.h"
#include "geometry/Curvilinear.h"
#include "mesh/LocalSimplexMesh.h"
#include "parallel/MPITraits.h"
#include "quadrules/AutoRule.h"
#include "tensor/Managed.h"
#include "tensor/Tensor.h"

#include <mneme/storage.hpp>
#include <mneme/view.hpp>

#include <cstddef>
#include <memory>
#include <vector>

namespace tndm {

template <std::size_t D> class DG {
public:
    DG(LocalSimplexMesh<D> const& mesh, Curvilinear<D>& cl,
       std::unique_ptr<RefElement<D>> refElement, unsigned minQuadOrder, MPI_Comm comm);
    virtual ~DG() = default;

    std::size_t numElements() const { return numElems_; }
    std::size_t numLocalElements() const { return numLocalElems_; }
    std::size_t numLocalFacets() const { return numLocalFacets_; }

    RefElement<D> const& refElement() const { return *refElement_; }

    // \todo: Create proper scatter class
    template <typename T> void scatter(T* data) {
        std::size_t sendSize = 0;
        std::size_t recvSize = 0;
        for (auto& [key, value] : sendMap) {
            sendSize += value.size();
        }
        for (auto& [key, value] : recvMap) {
            recvSize += value.size();
        }
        auto sendBuffer = std::vector<T>(sendSize);
        auto recvBuffer = std::vector<T>(recvSize);
        std::vector<MPI_Request> requests;
        requests.reserve(sendMap.size() + recvMap.size());
        std::size_t offset = 0;
        for (auto& [key, value] : recvMap) {
            requests.push_back(MPI_Request());
            MPI_Irecv(&recvBuffer[offset], value.size(), mpi_type_t<T>(), key, 0, comm_,
                      &requests.back());
            offset += value.size();
        }
        offset = 0;
        for (auto& [key, value] : sendMap) {
            for (auto&& v : value) {
                sendBuffer[offset++] = data[v];
            }
            requests.push_back(MPI_Request());
            MPI_Isend(&sendBuffer[offset - value.size()], value.size(), mpi_type_t<T>(), key, 0,
                      comm_, &requests.back());
        }
        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
        offset = 0;
        for (auto& [key, value] : recvMap) {
            for (auto&& v : value) {
                data[v] = recvBuffer[offset++];
            }
        }
    }

protected:
    void facetPrecompute(LocalSimplexMesh<D> const& mesh, Curvilinear<D>& cl);
    void volumePrecompute(LocalSimplexMesh<D> const& mesh, Curvilinear<D>& cl);

    std::unique_ptr<RefElement<D>> refElement_;
    std::size_t numElems_;
    std::size_t numLocalElems_;
    std::size_t numLocalFacets_;
    MPI_Comm comm_;

    std::unordered_map<int, std::vector<std::size_t>> sendMap;
    std::unordered_map<int, std::vector<std::size_t>> recvMap;

    SimplexQuadratureRule<D - 1u> fctRule;
    SimplexQuadratureRule<D> volRule;

    Managed<Matrix<double>> E;
    Managed<Tensor<double, 3u>> D_xi;
    std::vector<Managed<Matrix<double>>> e;
    std::vector<Managed<Tensor<double, 3u>>> d_xi;

    struct AbsDetJ {
        using type = double;
    };
    struct JInv {
        using type = std::array<double, D * D>;
    };
    struct JInvOther {
        using type = std::array<double, D * D>;
    };
    struct Normal {
        using type = std::array<double, D>;
    };
    struct NormalLength {
        using type = double;
    };
    struct Coords {
        using type = std::array<double, D>;
    };

    struct FacetInfo {
        std::array<bool, 2> inside;
        std::array<std::size_t, 2> up;
        std::array<std::size_t, 2> g_up;
        std::array<std::size_t, 2> localNo;
        double area;
    };
    std::vector<FacetInfo> fctInfo;

    struct GID {
        using type = std::size_t;
    };
    struct Penalty {
        using type = double;
    };
    mneme::MultiStorage<mneme::DataLayout::SoA, GID, Penalty> volInfo;

    using fct_t =
        mneme::MultiStorage<mneme::DataLayout::SoA, JInv, JInvOther, Normal, NormalLength, Coords>;
    mneme::StridedView<fct_t> fct;

    using vol_t = mneme::MultiStorage<mneme::DataLayout::SoA, AbsDetJ, JInv, Coords>;
    mneme::StridedView<vol_t> vol;
};

} // namespace tndm

#endif // DG_20200615_H
