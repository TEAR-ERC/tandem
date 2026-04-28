#ifndef VISCOELASTICITY_20260312_H
#define VISCOELASTICITY_20260312_H

#include "basis/WarpAndBlend.h"
#include "config.h"
#include "localoperator/NodalInterpolation.h"

#include "form/DGCurvilinearCommon.h"
#include "form/FacetInfo.h"
#include "form/FiniteElementFunction.h"
#include "form/RefElement.h"
#include "geometry/Curvilinear.h"
#include "tensor/Managed.h"
#include "tensor/Tensor.h"
#include "util/LinearAllocator.h"

#include "mneme/allocators.hpp"
#include "mneme/storage.hpp"
#include "mneme/view.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace tndm {

class Viscoelasticity : public DGCurvilinearCommon<DomainDimension> {
public:
    using base = DGCurvilinearCommon<DomainDimension>;
    constexpr static std::size_t Dim = DomainDimension;
    constexpr static std::size_t NumQuantities = DomainDimension;
    constexpr static std::size_t ScalarQuantity = 1;

    Viscoelasticity(std::shared_ptr<Curvilinear<DomainDimension>> cl, functional_t<1> lam,
                    functional_t<1> mu0, functional_t<1> mu1, functional_t<1> viscosity,
                    functional_t<1> relaxation_time, double theta,
                    std::optional<functional_t<1>> rho = std::nullopt,
                    DGMethod method = DGMethod::IP);

    constexpr std::size_t alignment() const { return ALIGNMENT; }
    std::size_t block_size() const { return space_.numBasisFunctions() * NumQuantities; }
    auto make_interpolation_op() const {
        return std::make_unique<NodalInterpolation<Dim>>(
            PolynomialDegree, WarpAndBlendFactory<Dim>(), NumQuantities, alignment());
    }

    void begin_preparation(std::size_t numElements, std::size_t numLocalElements,
                           std::size_t numLocalFacets);
    void prepare_volume(std::size_t elNo, LinearAllocator<double>& scratch);
    void prepare_skeleton(std::size_t fctNo, FacetInfo const& info,
                          LinearAllocator<double>& scratch);
    void prepare_boundary(std::size_t fctNo, FacetInfo const& info,
                          LinearAllocator<double>& scratch);
    void prepare_volume_post_skeleton(std::size_t elNo, LinearAllocator<double>& scratch);
    void prepare_penalty(std::size_t fctNo, FacetInfo const& info, LinearAllocator<double>&);
    void prepare_cfl(std::size_t elNo, mneme::span<SideInfo> info, LinearAllocator<double>&);

    void initialize_strain_tensor_Q(std::size_t elNo);
    void initialize_strain_tensor_q(std::size_t fctNo);
    void initialize_displacement_field(std::size_t elNo);
    void local_relaxation_time(std::size_t elNo, double& relaxation_time_global,
                               LinearAllocator<double>& scratch);

    bool assemble_volume(std::size_t elNo, Matrix<double>& A00,
                         LinearAllocator<double>& scratch) const;
    bool assemble_skeleton(std::size_t fctNo, FacetInfo const& info, Matrix<double>& A00,
                           Matrix<double>& A01, Matrix<double>& A10, Matrix<double>& A11,
                           LinearAllocator<double>& scratch) const;
    bool assemble_boundary(std::size_t fctNo, FacetInfo const& info, Matrix<double>& A00,
                           LinearAllocator<double>& scratch) const;
    bool assemble_boundary_free_slip(std::size_t fctNo, FacetInfo const& info, Matrix<double>& A00,
                                     LinearAllocator<double>& scratch) const;
    bool rhs_volume(std::size_t elNo, Vector<double>& B, LinearAllocator<double>& scratch) const;
    bool rhs_skeleton(std::size_t fctNo, FacetInfo const& info, Vector<double>& B0,
                      Vector<double>& B1, LinearAllocator<double>& scratch) const;
    bool rhs_boundary(std::size_t fctNo, FacetInfo const& info, Vector<double>& B0,
                      LinearAllocator<double>& scratch) const;
    bool rhs_boundary_free_slip(std::size_t fctNo, FacetInfo const& info, Vector<double>& B0,
                                LinearAllocator<double>& scratch) const;
    bool rhs_traction_boundary(std::size_t fctNo, FacetInfo const& info, Vector<double>& B0,
                               LinearAllocator<double>& scratch) const;
    bool rhs_free_slip_boundary(std::size_t fctNo, FacetInfo const& info, Vector<double>& B0,
                                LinearAllocator<double>& scratch) const;
    bool rhs_history_skeleton(std::size_t fctNo, FacetInfo const& info, Vector<double>& B0,
                              Vector<double>& B1, LinearAllocator<double>& scratch) const;
    bool rhs_history_boundary(std::size_t fctNo, FacetInfo const& info, Vector<double>& B0,
                              LinearAllocator<double>& scratch) const;
    bool rhs_history_boundary_for_free_slip(std::size_t fctNo, FacetInfo const& info,
                                            Vector<double>& B0,
                                            LinearAllocator<double>& scratch) const;
    void apply(std::size_t elNo, mneme::span<SideInfo> info, Vector<double const> const& x_0,
               std::array<Vector<double const>, NumFacets> const& x_n, Vector<double>& y_0) const;
    void wave_rhs(std::size_t elNo, mneme::span<SideInfo> info, Vector<double const> const& x_0,
                  std::array<Vector<double const>, NumFacets> const& x_n,
                  Vector<double>& y_0) const;
    void project(std::size_t elNo, volume_functional_t x, Vector<double>& y) const;

    std::size_t flops_apply(std::size_t elNo, mneme::span<SideInfo> info) const;

    TensorBase<Matrix<double>> tractionResultInfo() const;
    void traction_skeleton(std::size_t fctNo, FacetInfo const& info, Vector<double const>& u0,
                           Vector<double const>& u1, Matrix<double>& result) const;
    void traction_boundary(std::size_t fctNo, FacetInfo const& info, Vector<double const>& u0,
                           Matrix<double>& result) const;

    void store_displacement_field(std::size_t elNo, const double* U_data) const;
    void compute_deviatoric_strain_Q(std::size_t elNo);
    void compute_deviatoric_strain_q(std::size_t fctNo, FacetInfo const& info);
    void compute_partial_strain_Q(std::size_t elNo);
    void compute_partial_strain_q(std::size_t fctNo, FacetInfo const& info);
    void update_deviatoric_strain_Q(std::size_t elNo);
    void update_deviatoric_strain_q(std::size_t fctNo);
    void update_partial_strain_Q(std::size_t elNo);
    void update_partial_strain_q(std::size_t fctNo);
    void compute_sigma_hat_n_facets();

    // Stress field computation for VTU output
    FiniteElementFunction<DomainDimension> stress_prototype(std::size_t numLocalElements) const;
    void stress_volume(std::size_t elNo, Matrix<double>& stress_components,
                       LinearAllocator<double>& scratch) const;

    inline double cfl_time_step(std::size_t elNo) const { return cfl_dt_[elNo]; }
    inline double theta() const { return theta_; }

    FiniteElementFunction<DomainDimension> solution_prototype(std::size_t numLocalElements) const {
        auto names = std::vector<std::string>(NumQuantities);
        char buf[100];
        for (std::size_t q = 0; q < NumQuantities; ++q) {
            snprintf(buf, sizeof(buf), "u%lu", q);
            names[q] = buf;
        }
        return FiniteElementFunction<DomainDimension>(space_.clone(), names, numLocalElements);
    }

    FiniteElementFunction<DomainDimension>
    coefficients_prototype(std::size_t numLocalElements) const {
        return FiniteElementFunction<DomainDimension>(
            materialSpace_.clone(), {"lambda", "mu0", "mu1", "viscosity"}, numLocalElements);
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
    void set_traction_boundary(functional_t<NumQuantities> fun) {
        fun_traction = make_facet_functional(std::move(fun));
    }
    void set_traction_boundary(functional_t<NumQuantities> fun,
                               std::array<double, DomainDimension> const& refNormal) {
        fun_traction = make_facet_functional(std::move(fun), refNormal);
    }
    void set_traction_boundary(facet_functional_t fun) { fun_traction = std::move(fun); }

    void set_free_slip_boundary(functional_t<ScalarQuantity> fun) {
        fun_free_slip = make_facet_functional(std::move(fun));
    }

    void set_free_slip_boundary(functional_t<ScalarQuantity> fun,
                                std::array<double, DomainDimension> const& refNormal) {
        fun_free_slip = make_facet_functional(std::move(fun), refNormal);
    }

    void set_free_slip_boundary(facet_functional_t fun) { fun_free_slip = std::move(fun); }

    void set_viscoelastic_time_step(double dt) { dt_viscoelastic_ = dt; }
    inline double get_viscoelastic_time_step() const { return dt_viscoelastic_; }
    void update_time_dependent_precomputation_volume(std::size_t elNo);
    void update_time_dependent_precomputation_skeleton(std::size_t fctNo);
    void update_time_dependent_precomputation_boundary(std::size_t fctNo);

private:
    template <bool WithRHS>
    void apply_(std::size_t elNo, mneme::span<SideInfo> info, Vector<double const> const& x_0,
                std::array<Vector<double const>, NumFacets> const& x_n, Vector<double>& y_0) const;

    double penalty(std::size_t fctNo) const {
        if (method_ == DGMethod::BR2) {
            return NumFacets;
        }
        return penalty_[fctNo];
    }
    std::pair<double, double> stiffness_tensor_bounds(std::size_t elNo) const;
    double inverse_density_upper_bound(std::size_t elNo) const;
    void compute_mass_matrix(std::size_t elNo, double* M) const;
    void compute_inverse_mass_matrix(std::size_t elNo, double* Minv) const;
    bool bc_skeleton(std::size_t fctNo, BC bc, double f_q_raw[]) const;
    bool bc_boundary(std::size_t fctNo, BC bc, double f_q_raw[]) const;
    bool bc_traction(std::size_t fctNo, BC bc, double f_q_raw[]) const;
    bool bc_free_slip(std::size_t fctNo, BC bc, double f_q_raw[]) const;

    void transpose_JInv(std::size_t fctNo, int side);
    double compute_g_dt(double dt, double tau, int) const;
    void update_time_dependent_precomputation_surface(std::size_t fctNo, int numSides);
    void precompute_facet_material(std::size_t fctNo, FacetInfo const& info, int numSides);
    void compute_lift_boundary(std::size_t fctNo, FacetInfo const& info, double* L_q0) const;
    void compute_lift_skeleton(std::size_t fctNo, FacetInfo const& info, double* L_q0,
                               double* L_q1) const;
    void compute_rhs_lift(std::size_t fctNo, FacetInfo const& info, bool is_skeleton,
                          const double* f_q_raw, double* f_lifted_q) const;
    void compute_traction_history_average(std::size_t fctNo, const double* dev_q0,
                                          const double* partial_q0, const double* mu1_q0,
                                          const double* dev_q1, const double* partial_q1,
                                          const double* mu1_q1) const;

    DGMethod method_;

    // Ref elements
    NodalRefElement<DomainDimension> space_;
    NodalRefElement<DomainDimension> materialSpace_;

    // Matrices
    Managed<Matrix<double>> MhatInv;
    Managed<Matrix<double>> E_Q;
    Managed<Matrix<double>> E_Q_T;
    Managed<Matrix<double>> negative_E_Q_T;
    Managed<Tensor<double, 3u>> Dxi_Q;
    Managed<Tensor<double, 3u>> Dxi_Q_120;
    Managed<Tensor<double, 3u>> Dxi;
    std::vector<Managed<Matrix<double>>> E_q;
    std::vector<Managed<Matrix<double>>> E_q_T;
    std::vector<Managed<Matrix<double>>> negative_E_q;
    std::vector<Managed<Matrix<double>>> negative_E_q_T;
    std::vector<Managed<Tensor<double, 3u>>> Dxi_q;
    std::vector<Managed<Tensor<double, 3u>>> Dxi_q_120;

    Managed<Matrix<double>> matE_Q_T;
    std::vector<Managed<Matrix<double>>> matE_q_T;

    // Input functionals
    volume_functional_t fun_lam;
    volume_functional_t fun_mu0;
    volume_functional_t fun_mu1;
    volume_functional_t fun_viscosity;
    volume_functional_t fun_relaxation_time;
    volume_functional_t fun_rho;
    double theta_;
    double dt_viscoelastic_ = 0.0;
    std::optional<volume_functional_t> fun_force = std::nullopt;
    std::optional<facet_functional_t> fun_dirichlet = std::nullopt;
    std::optional<facet_functional_t> fun_slip = std::nullopt;
    std::optional<facet_functional_t> fun_traction = std::nullopt;
    std::optional<facet_functional_t> fun_free_slip = std::nullopt;

    // Precomputed data — material storage
    struct lam {
        using type = double;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    struct mu0 {
        using type = double;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    struct mu1 {
        using type = double;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    struct viscosity {
        using type = double;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    struct rhoInv {
        using type = double;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };

    // Volume precomputed quantities
    struct lam_W_J_Q {
        using type = double;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    struct mu0_W_J_Q {
        using type = double;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    struct mu1_W_J_Q {
        using type = double;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    struct viscosity_W_J_Q {
        using type = double;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    struct negative_rhoInv_W_Jinv_Q {
        using type = double;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    struct relaxation_time_local {
        using type = double;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    struct g_dt_Q {
        using type = double;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    struct ratio_Q {
        using type = double;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    struct A_dt {
        using type = double;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    struct B_dt {
        using type = double;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    struct JInvT {
        using type = std::array<double, Dim * Dim>;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };

    // Facet precomputed quantities — side 0
    struct lam_q_0 {
        using type = double;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    struct mu0_q_0 {
        using type = double;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    struct mu1_q_0 {
        using type = double;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    struct viscosity_q_0 {
        using type = double;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    struct A_dt_q_0 {
        using type = double;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    struct B_dt_q_0 {
        using type = double;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    struct JInvT0 {
        using type = std::array<double, Dim * Dim>;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };

    // Facet precomputed quantities — side 1
    struct lam_q_1 {
        using type = double;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    struct mu0_q_1 {
        using type = double;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    struct mu1_q_1 {
        using type = double;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    struct viscosity_q_1 {
        using type = double;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    struct A_dt_q_1 {
        using type = double;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    struct B_dt_q_1 {
        using type = double;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    struct JInvT1 {
        using type = std::array<double, Dim * Dim>;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };

    // Facet precomputed quantities — shared (not per-side)
    struct g_dt_q {
        using type = double;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    struct ratio_q {
        using type = double;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };

    // Strain history storage — volume quad points
    struct partial_strain_old_Q {
        using type = std::array<double, Dim * Dim>;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    struct partial_strain_new_Q {
        using type = std::array<double, Dim * Dim>;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    struct deviatoric_strain_old_Q {
        using type = std::array<double, Dim * Dim>;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    struct deviatoric_strain_new_Q {
        using type = std::array<double, Dim * Dim>;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    struct total_strain_Q {
        using type = std::array<double, Dim * Dim>;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    struct trace_tensor_Q {
        using type = std::array<double, Dim * Dim>;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    struct strain_trace_Q {
        using type = double;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };

    // Strain history storage — facet quad points (side 0)
    struct partial_strain_old_q_0 {
        using type = std::array<double, Dim * Dim>;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    struct partial_strain_new_q_0 {
        using type = std::array<double, Dim * Dim>;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    struct deviatoric_strain_old_q_0 {
        using type = std::array<double, Dim * Dim>;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    struct deviatoric_strain_new_q_0 {
        using type = std::array<double, Dim * Dim>;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    struct total_strain_q_0 {
        using type = std::array<double, Dim * Dim>;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    struct trace_tensor_q_0 {
        using type = std::array<double, Dim * Dim>;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    struct strain_trace_q_0 {
        using type = double;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };

    // Strain history storage — facet quad points (side 1)
    struct partial_strain_old_q_1 {
        using type = std::array<double, Dim * Dim>;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    struct partial_strain_new_q_1 {
        using type = std::array<double, Dim * Dim>;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    struct deviatoric_strain_old_q_1 {
        using type = std::array<double, Dim * Dim>;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    struct deviatoric_strain_new_q_1 {
        using type = std::array<double, Dim * Dim>;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    struct total_strain_q_1 {
        using type = std::array<double, Dim * Dim>;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    struct trace_tensor_q_1 {
        using type = std::array<double, Dim * Dim>;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    struct strain_trace_q_1 {
        using type = double;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };

    // Traction history
    struct average_traction {
        using type = std::array<double, Dim>;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };
    struct traction_history_normal {
        using type = double;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };

    // Displacement field storage
    struct displacement_field {
        using type = std::array<double, Dim>;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };

    // Stress field storage — symmetric tensor components for VTU output
    // 3D: 6 components (xx, yy, zz, xy, xz, yz)
    // 2D: 3 components (xx, yy, xy)
    struct stress_nodal {
        using type = std::array<double, Dim * Dim>;
        using allocator = mneme::AlignedAllocator<type, ALIGNMENT>;
    };

    using material_vol_t =
        mneme::MultiStorage<mneme::DataLayout::SoA, lam, mu0, mu1, viscosity, rhoInv>;
    mneme::StridedView<material_vol_t> material;

    using vol_pre_t =
        mneme::MultiStorage<mneme::DataLayout::SoA, lam_W_J_Q, mu0_W_J_Q, mu1_W_J_Q,
                            viscosity_W_J_Q, negative_rhoInv_W_Jinv_Q, relaxation_time_local,
                            g_dt_Q, ratio_Q, A_dt, B_dt, JInvT>;
    mneme::StridedView<vol_pre_t> volPre;

    using fct_pre_t =
        mneme::MultiStorage<mneme::DataLayout::SoA, lam_q_0, mu0_q_0, mu1_q_0, viscosity_q_0,
                            A_dt_q_0, B_dt_q_0, JInvT0, lam_q_1, mu0_q_1, mu1_q_1, viscosity_q_1,
                            A_dt_q_1, B_dt_q_1, JInvT1, g_dt_q, ratio_q>;
    mneme::StridedView<fct_pre_t> fctPre;

    using strain_tensor_Q_t =
        mneme::MultiStorage<mneme::DataLayout::SoA, partial_strain_old_Q, partial_strain_new_Q,
                            deviatoric_strain_old_Q, deviatoric_strain_new_Q, total_strain_Q,
                            trace_tensor_Q, strain_trace_Q>;
    mneme::StridedView<strain_tensor_Q_t> strainHistory_Q;

    using strain_tensor_q_t =
        mneme::MultiStorage<mneme::DataLayout::SoA, partial_strain_old_q_0, partial_strain_new_q_0,
                            deviatoric_strain_old_q_0, deviatoric_strain_new_q_0, total_strain_q_0,
                            trace_tensor_q_0, strain_trace_q_0, partial_strain_old_q_1,
                            partial_strain_new_q_1, deviatoric_strain_old_q_1,
                            deviatoric_strain_new_q_1, total_strain_q_1, trace_tensor_q_1,
                            strain_trace_q_1, average_traction, traction_history_normal>;
    mneme::StridedView<strain_tensor_q_t> strainHistory_q;

    using displacement_field_t = mneme::MultiStorage<mneme::DataLayout::SoA, displacement_field>;
    mneme::StridedView<displacement_field_t> displacementField;

    std::vector<double> penalty_;
    std::vector<double> cfl_dt_;

    constexpr static double epsilon = -1.0;
};

} // namespace tndm

#endif // VISCOELASTICITY_20260312_H
