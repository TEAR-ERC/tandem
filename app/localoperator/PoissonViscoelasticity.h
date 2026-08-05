#ifndef PoissonViscoelasticity_20200910_H
#define PoissonViscoelasticity_20200910_H

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

class PoissonViscoelasticity : public DGCurvilinearCommon<DomainDimension> {
public:
    using base = DGCurvilinearCommon<DomainDimension>;
    constexpr static std::size_t Dim = DomainDimension;
    constexpr static std::size_t NumQuantities = 1;

    PoissonViscoelasticity(std::shared_ptr<Curvilinear<DomainDimension>> cl, functional_t<1> mu0,
                           functional_t<1> mu1, functional_t<1> eta, double theta = 0.1,
                           DGMethod method = DGMethod::BR2);

    constexpr std::size_t alignment() const { return ALIGNMENT; }
    std::size_t block_size() const { return space_.numBasisFunctions(); }
    auto make_interpolation_op() const {
        return std::make_unique<ModalInterpolation<Dim>>(PolynomialDegree, NumQuantities,
                                                         alignment());
    }

    //------------ preparation------------

    void begin_preparation(std::size_t numElements, std::size_t numLocalElements,
                           std::size_t numLocalFacets);
    void prepare_volume(std::size_t elNo, LinearAllocator<double>& scratch);
    void prepare_skeleton(std::size_t fctNo, FacetInfo const& info,
                          LinearAllocator<double>& scratch);
    void prepare_boundary(std::size_t fctNo, FacetInfo const& info,
                          LinearAllocator<double>& scratch);
    void prepare_volume_post_skeleton(std::size_t elNo, LinearAllocator<double>& scratch);
    void prepare_penalty(std::size_t fctNo, FacetInfo const& info, LinearAllocator<double>&);

    //------------ assembly------------

    bool assemble_volume(std::size_t elNo, Matrix<double>& A00,
                         LinearAllocator<double>& scratch) const;
    bool assemble_skeleton(std::size_t fctNo, FacetInfo const& info, Matrix<double>& A00,
                           Matrix<double>& A01, Matrix<double>& A10, Matrix<double>& A11,
                           LinearAllocator<double>& scratch) const;
    bool assemble_boundary(std::size_t fctNo, FacetInfo const& info, Matrix<double>& A00,
                           LinearAllocator<double>& scratch) const;

    //------------ right-hand side------------

    bool rhs_volume(std::size_t elNo, Vector<double>& B, LinearAllocator<double>& scratch) const;
    bool rhs_skeleton(std::size_t fctNo, FacetInfo const& info, Vector<double>& B0,
                      Vector<double>& B1, LinearAllocator<double>& scratch) const;
    bool rhs_boundary(std::size_t fctNo, FacetInfo const& info, Vector<double>& B0,
                      LinearAllocator<double>& scratch) const;

    // Viscoelastic history contributions (detected by DGOperator via SFINAE).
    bool rhs_history_volume(std::size_t elNo, Vector<double>& B,
                            LinearAllocator<double>& scratch) const;
    bool rhs_history_skeleton(std::size_t fctNo, FacetInfo const& info, Vector<double>& B0,
                              Vector<double>& B1, LinearAllocator<double>& scratch) const;
    bool rhs_history_boundary(std::size_t fctNo, FacetInfo const& info, Vector<double>& B0,
                              LinearAllocator<double>& scratch) const;

    // Neumann / prescribed traction on BC::Natural facets.
    bool rhs_traction_boundary(std::size_t fctNo, FacetInfo const& info, Vector<double>& B0,
                               LinearAllocator<double>& scratch) const;

    //------------ matrix free------------

    void apply(std::size_t elNo, mneme::span<SideInfo> info, Vector<double const> const& x_0,
               std::array<Vector<double const>, NumFacets> const& x_n, Vector<double>& y_0) const;

    std::size_t flops_apply(std::size_t elNo, mneme::span<SideInfo> info) const;

    //------------ viscoelastic state management------------

    void local_relaxation_time(std::size_t elNo, double& dt_min, LinearAllocator<double>& scratch);
    void set_relaxation_time_global(double tau) { relaxation_time_global_ = tau; }
    double viscoelastic_theta() const { return theta_; }
    void set_viscoelastic_theta(double theta) { theta_ = theta; }

    /// dt = theta * tau_global.  Does NOT recompute coefficients (may be called
    /// before the material has been projected).
    void set_viscoelastic_time_step(double tau_global) { dt_ = theta_ * tau_global; }
    /// Set dt directly.  The caller must follow with the
    /// update_time_dependent_precomputation_* sweep.
    void set_viscoelastic_time_step_value(double dt) { dt_ = dt; }
    double get_viscoelastic_time_step() const { return dt_; }

    void update_time_dependent_precomputation_volume(std::size_t elNo);
    void update_time_dependent_precomputation_skeleton(std::size_t fctNo);
    void update_time_dependent_precomputation_boundary(std::size_t fctNo);

    void store_displacement_field(std::size_t elNo, double const* u);
    void initialize_displacement_field(std::size_t elNo);

    void initialize_strain_tensor_Q(std::size_t elNo);
    void initialize_strain_tensor_q(std::size_t fctNo);
    void initialize_partial_strain_Q(std::size_t elNo);
    void initialize_partial_strain_q(std::size_t fctNo);

    void compute_deviatoric_strain_Q(std::size_t elNo);
    void compute_deviatoric_strain_q(std::size_t fctNo, FacetInfo const& info);
    void compute_partial_strain_Q(std::size_t elNo);
    void compute_partial_strain_q(std::size_t fctNo, FacetInfo const& info);
    void update_deviatoric_strain_Q(std::size_t elNo);
    void update_deviatoric_strain_q(std::size_t fctNo);
    void update_partial_strain_Q(std::size_t elNo);
    void update_partial_strain_q(std::size_t fctNo);

    //------------ output------------

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
        return FiniteElementFunction<DomainDimension>(
            materialSpace_.clone(), {"mu0", "mu1", "tau", "mu_eff"}, numLocalElements);
    }
    void coefficients_volume(std::size_t elNo, Matrix<double>& C, LinearAllocator<double>&) const;

    //------------ functionals------------

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
    void set_traction_boundary(facet_functional_t fun) { fun_traction = std::move(fun); }

private:
    double penalty(std::size_t fctNo) const {
        if (method_ == DGMethod::BR2) {
            return NumFacets;
        }
        return penalty_[fctNo];
    }
    void compute_mass_matrix(std::size_t elNo, double* M) const;
    void compute_inverse_mass_matrix(std::size_t elNo, double* Minv) const;
    void compute_K_Dx_q(std::size_t fctNo, FacetInfo const& info,
                        std::array<double*, 2> K_Dx_q) const;
    void compute_K_q(std::size_t fctNo, FacetInfo const& info, std::array<double*, 2> K_q) const;
    bool bc_skeleton(std::size_t fctNo, BC bc, double f_q_raw[]) const;
    bool bc_boundary(std::size_t fctNo, BC bc, double f_q_raw[]) const;

    // Viscoelastic helpers
    void compute_time_coefficients(std::size_t elNo);
    void compute_history_weights_Q(std::size_t elNo);
    void compute_history_weights_q(std::size_t fctNo, int side, std::size_t elNo,
                                   std::size_t localNo);
    void compute_J_W_K_Q(std::size_t elNo);
    void compute_K_G_q(std::size_t fctNo, int side, std::size_t elNo, std::size_t localNo);
    void compute_penalty(std::size_t fctNo);
    /// sigmaHat^n at facet quadrature points for the given side.
    void compute_sigma_hist_q(std::size_t fctNo, int side, double* sigma_hist) const;

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
    volume_functional_t fun_mu0;
    volume_functional_t fun_mu1;
    volume_functional_t fun_eta;
    volume_functional_t fun_force;
    facet_functional_t fun_dirichlet;
    facet_functional_t fun_slip;
    facet_functional_t fun_traction;

    // material (nodal material space, all elements incl. ghosts)
    template <int Id> struct MatField {
        using type = double;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    using K = MatField<0>;     // mu_0
    using Mu1 = MatField<1>;   // mu_1
    using Tau = MatField<2>;   // eta / mu_1
    using MuEff = MatField<3>; // mu_0 + g_dt * mu_1
    using Gdt = MatField<4>;   // (tau/dt)(1 - exp(-dt/tau))
    using Ratio = MatField<5>; // exp(-dt/tau)

    using material_vol_t =
        mneme::MultiStorage<mneme::DataLayout::SoA, K, Mu1, Tau, MuEff, Gdt, Ratio>;
    mneme::StridedView<material_vol_t> material;

    // volume precomputation
    struct AbsDetJWK {
        using type = std::array<double, Dim * Dim>;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    using vol_pre_t = mneme::MultiStorage<mneme::DataLayout::SoA, AbsDetJWK>;
    mneme::StridedView<vol_pre_t> volPre;

    // volume history state, per quadrature point, all elements
    template <int Id> struct QpScalar {
        using type = double;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    template <int Id> struct QpVector {
        using type = std::array<double, Dim>;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    using GdtQ = QpScalar<0>;
    using RatioQ = QpScalar<1>;
    using Mu1GWJ_Q = QpScalar<2>;
    using Mu1RWJ_Q = QpScalar<3>;
    using EpsOldQ = QpVector<0>;
    using EpsNewQ = QpVector<1>;
    using QOldQ = QpVector<2>;
    using QNewQ = QpVector<3>;

    using vol_hist_t = mneme::MultiStorage<mneme::DataLayout::SoA, GdtQ, RatioQ, Mu1GWJ_Q, Mu1RWJ_Q,
                                           EpsOldQ, EpsNewQ, QOldQ, QNewQ>;
    mneme::StridedView<vol_hist_t> volHist;

    // displacement field, all elements incl. ghosts
    struct Displacement {
        using type = double;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    using displ_t = mneme::MultiStorage<mneme::DataLayout::SoA, Displacement>;
    mneme::StridedView<displ_t> displacement;

    // facet precomputation
    struct KJInv0 {
        using type = std::array<double, Dim * Dim>;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    struct KJInv1 {
        using type = std::array<double, Dim * Dim>;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    using fct_pre_t = mneme::MultiStorage<mneme::DataLayout::SoA, KJInv0, KJInv1>;
    mneme::StridedView<fct_pre_t> fctPre;

    // facet history state, per side, per quadrature point
    template <int Id> struct FqScalar {
        using type = double;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    template <int Id> struct FqVector {
        using type = std::array<double, Dim>;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    using Gdt_q0 = FqScalar<0>;
    using Gdt_q1 = FqScalar<1>;
    using Ratio_q0 = FqScalar<2>;
    using Ratio_q1 = FqScalar<3>;
    using Mu1G_q0 = FqScalar<4>;
    using Mu1G_q1 = FqScalar<5>;
    using Mu1R_q0 = FqScalar<6>;
    using Mu1R_q1 = FqScalar<7>;
    using EpsOld_q0 = FqVector<0>;
    using EpsOld_q1 = FqVector<1>;
    using EpsNew_q0 = FqVector<2>;
    using EpsNew_q1 = FqVector<3>;
    using QOld_q0 = FqVector<4>;
    using QOld_q1 = FqVector<5>;
    using QNew_q0 = FqVector<6>;
    using QNew_q1 = FqVector<7>;

    using fct_hist_t =
        mneme::MultiStorage<mneme::DataLayout::SoA, Gdt_q0, Gdt_q1, Ratio_q0, Ratio_q1, Mu1G_q0,
                            Mu1G_q1, Mu1R_q0, Mu1R_q1, EpsOld_q0, EpsOld_q1, EpsNew_q0, EpsNew_q1,
                            QOld_q0, QOld_q1, QNew_q0, QNew_q1>;
    mneme::StridedView<fct_hist_t> fctHist;

    // Side-indexed accessors for the facet history storage.
    double* gdt_q(std::size_t fctNo, int side);
    double* ratio_q(std::size_t fctNo, int side);
    double* mu1_g_q(std::size_t fctNo, int side);
    double* mu1_r_q(std::size_t fctNo, int side);
    double* eps_old_q(std::size_t fctNo, int side);
    double const* eps_old_q(std::size_t fctNo, int side) const;
    double* eps_new_q(std::size_t fctNo, int side);
    double const* eps_new_q(std::size_t fctNo, int side) const;
    double* q_old_q(std::size_t fctNo, int side);
    double* q_new_q(std::size_t fctNo, int side);
    double const* q_new_q(std::size_t fctNo, int side) const;

    /// localNo / up cached during preparation, because the
    /// update_time_dependent_precomputation_* hooks receive only fctNo.
    struct CachedFacet {
        std::array<std::size_t, 2> localNo{0, 0};
        std::array<std::size_t, 2> up{0, 0};
        bool skeleton = false;
    };
    std::vector<CachedFacet> fctCache_;

    std::vector<double> penalty_;

    std::size_t numElements_ = 0;
    std::size_t numLocalElements_ = 0;
    std::size_t numLocalFacets_ = 0;

    // Viscoelastic time stepping
    double theta_ = 0.1;
    double dt_ = 0.0;
    double relaxation_time_global_ = 0.0;

    // Options
    constexpr static double epsilon = -1.0;
};

} // namespace tndm

#endif // PoissonViscoelasticity_20200910_H
