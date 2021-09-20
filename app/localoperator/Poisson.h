#ifndef POISSON_20200910_H
#define POISSON_20200910_H

#include "config.h"
#include "localoperator/ModalInterpolation.h"

#include "form/DGCurvilinearCommon.h"
#include "form/FacetInfo.h"
#include "form/FiniteElementFunction.h"
#include "form/RefElement.h"
#include "geometry/Curvilinear.h"
#include "tensor/Managed.h"
#include "tensor/Tensor.h"
#include "tensor/TensorBase.h"
#include "util/LinearAllocator.h"

#include "mneme/allocators.hpp"
#include "mneme/storage.hpp"
#include "mneme/view.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

namespace tndm {

class Poisson : public DGCurvilinearCommon<DomainDimension> {
public:
    using base = DGCurvilinearCommon<DomainDimension>;
    constexpr static std::size_t Dim = DomainDimension;
    constexpr static std::size_t NumQuantities = 1;

    Poisson(std::shared_ptr<Curvilinear<DomainDimension>> cl, functional_t<1> K,
            DGMethod method = DGMethod::BR2);

    constexpr std::size_t alignment() const { return ALIGNMENT; }
    std::size_t block_size() const { return space_.numBasisFunctions(); }
    auto make_interpolation_op() const {
        return std::make_unique<ModalInterpolation<Dim>>(PolynomialDegree, NumQuantities,
                                                         alignment());
    }

    void begin_preparation(std::size_t numElements, std::size_t numLocalElements,
                           std::size_t numLocalFacets);
    void prepare_volume(std::size_t elNo, LinearAllocator<double>& scratch);
    void prepare_skeleton(std::size_t fctNo, FacetInfo const& info,
                          LinearAllocator<double>& scratch);
    void prepare_boundary(std::size_t fctNo, FacetInfo const& info,
                          LinearAllocator<double>& scratch);
    void prepare_volume_post_skeleton(std::size_t elNo, LinearAllocator<double>& scratch);

    bool assemble_volume(std::size_t elNo, Matrix<double>& A00,
                         LinearAllocator<double>& scratch) const;
    bool assemble_skeleton(std::size_t fctNo, FacetInfo const& info, Matrix<double>& A00,
                           Matrix<double>& A01, Matrix<double>& A10, Matrix<double>& A11,
                           LinearAllocator<double>& scratch) const;
    bool assemble_boundary(std::size_t fctNo, FacetInfo const& info, Matrix<double>& A00,
                           LinearAllocator<double>& scratch) const;

    bool rhs_volume(std::size_t elNo, Vector<double>& B, LinearAllocator<double>& scratch) const;
    bool rhs_skeleton(std::size_t fctNo, FacetInfo const& info, Vector<double>& B0,
                      Vector<double>& B1, LinearAllocator<double>& scratch) const;
    bool rhs_boundary(std::size_t fctNo, FacetInfo const& info, Vector<double>& B0,
                      LinearAllocator<double>& scratch) const;

    void apply(std::size_t elNo, mneme::span<SideInfo> info, Vector<double const> const& x_0,
               std::array<Vector<double const>, NumFacets> const& x_n, Vector<double>& y_0) const;

    std::size_t flops_apply(std::size_t elNo, mneme::span<SideInfo> info) const;

    TensorBase<Matrix<double>> tractionResultInfo() const;
    void traction_skeleton(std::size_t fctNo, FacetInfo const& info, Vector<double const>& u0,
                           Vector<double const>& u1, Matrix<double>& result) const;
    void traction_boundary(std::size_t fctNo, FacetInfo const& info, Vector<double const>& u0,
                           Matrix<double>& result) const;

    FiniteElementFunction<DomainDimension> solution_prototype(std::size_t numLocalElements) const {
        return FiniteElementFunction<DomainDimension>(space_.clone(), {"u"}, numLocalElements);
    }

    FiniteElementFunction<DomainDimension>
    coefficients_prototype(std::size_t numLocalElements) const {
        return FiniteElementFunction<DomainDimension>(materialSpace_.clone(), {"K"},
                                                      numLocalElements);
    }
    void coefficients_volume(std::size_t elNo, Matrix<double>& C, LinearAllocator<double>&) const;

    void set_force(functional_t<NumQuantities> fun) {
        fun_force = make_volume_functional(std::move(fun));
    }
    void set_force(volume_functional_t fun) { fun_force = std::move(fun); }
    void set_dirichlet(functional_t<NumQuantities> fun) {
        fun_dirichlet = make_facet_functional(std::move(fun));
    }
    void set_dirichlet(functional_t<NumQuantities> fun,
                       std::array<double, DomainDimension> const& refNormal) {
        fun_dirichlet = make_facet_functional(std::move(fun), refNormal);
    }
    void set_dirichlet(facet_functional_t fun) { fun_dirichlet = std::move(fun); }
    void set_slip(functional_t<NumQuantities> fun,
                  std::array<double, DomainDimension> const& refNormal) {
        fun_slip = make_facet_functional(std::move(fun), refNormal);
    }
    void set_slip(facet_functional_t fun) { fun_slip = std::move(fun); }

private:
    double penalty(std::size_t elNo0, std::size_t elNo1) const {
        if (method_ == DGMethod::BR2) {
            return NumFacets;
        }
        return std::max(base::penalty[elNo0], base::penalty[elNo1]);
    }
    double penalty(FacetInfo const& info) const { return penalty(info.up[0], info.up[1]); }
    void compute_mass_matrix(std::size_t elNo, double* M) const;
    void compute_inverse_mass_matrix(std::size_t elNo, double* Minv) const;
    void compute_K_Dx_q(std::size_t fctNo, FacetInfo const& info,
                        std::array<double*, 2> K_Dx_q) const;
    void compute_K_q(std::size_t fctNo, FacetInfo const& info, std::array<double*, 2> K_q) const;
    bool bc_skeleton(std::size_t fctNo, BC bc, double f_q_raw[]) const;
    bool bc_boundary(std::size_t fctNo, BC bc, double f_q_raw[]) const;

    DGMethod method_;

    // Ref elements
    ModalRefElement<DomainDimension> space_;
    NodalRefElement<DomainDimension> materialSpace_;

    // Matrices
    Managed<Matrix<double>> Minv_;
    Managed<Matrix<double>> E_Q;
    Managed<Matrix<double>> E_Q_T;
    Managed<Matrix<double>> negative_E_Q_T;
    Managed<Tensor<double, 3u>> Dxi_Q;
    std::vector<Managed<Matrix<double>>> E_q;
    std::vector<Managed<Matrix<double>>> E_q_T;
    std::vector<Managed<Matrix<double>>> negative_E_q_T;
    std::vector<Managed<Tensor<double, 3u>>> Dxi_q;
    std::vector<Managed<Tensor<double, 3u>>> Dxi_q_120;

    Managed<Matrix<double>> matE_Q_T;
    Managed<Tensor<double, 3u>> matDxi_Q;
    std::vector<Managed<Matrix<double>>> matE_q_T;

    // Input
    volume_functional_t fun_K;
    volume_functional_t fun_force;
    facet_functional_t fun_dirichlet;
    facet_functional_t fun_slip;

    // Precomputed data
    struct K {
        using type = double;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    struct AbsDetJWK {
        using type = std::array<double, Dim * Dim>;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    struct KJInv0 {
        using type = std::array<double, Dim * Dim>;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    struct KJInv1 {
        using type = std::array<double, Dim * Dim>;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };

    using material_vol_t = mneme::MultiStorage<mneme::DataLayout::SoA, K>;
    mneme::StridedView<material_vol_t> material;

    using vol_pre_t = mneme::MultiStorage<mneme::DataLayout::SoA, AbsDetJWK>;
    mneme::StridedView<vol_pre_t> volPre;

    using fct_pre_t = mneme::MultiStorage<mneme::DataLayout::SoA, KJInv0, KJInv1>;
    mneme::StridedView<fct_pre_t> fctPre;

    // Options
    constexpr static double epsilon = -1.0;
};

} // namespace tndm

#endif // POISSON_20200910_H
