#ifndef DGOPERATOR_20200909_H
#define DGOPERATOR_20200909_H

#include "form/AbstractDGOperator.h"
#include "form/AbstractInterpolationOperator.h"
#include "form/DGOperatorTopo.h"
#include "form/FiniteElementFunction.h"
#include "form/InterpolationOperator.h"
#include "interface/BlockMatrix.h"
#include "interface/BlockVector.h"
#include "parallel/LocalGhostCompositeView.h"
#include "parallel/Scatter.h"
#include "parallel/SparseBlockVector.h"
#include "tensor/Managed.h"
#include "tensor/Reshape.h"
#include "tensor/Tensor.h"
#include "util/Scratch.h"

#include <cassert>
#include <experimental/type_traits>
#include <memory>
#include <utility>

namespace tndm {

template <typename LocalOperator> class DGOperator : public AbstractDGOperator<LocalOperator::Dim> {
public:
    using base = AbstractDGOperator<LocalOperator::Dim>;
    using local_operator_t = LocalOperator;
    constexpr static std::size_t NumFacets = LocalOperator::Dim + 1;

    template <class T> using prepare_volume_t = decltype(&T::prepare_volume);
    template <class T> using prepare_skeleton_t = decltype(&T::prepare_skeleton);
    template <class T> using prepare_boundary_t = decltype(&T::prepare_boundary);
    template <class T>
    using prepare_volume_post_skeleton_t = decltype(&T::prepare_volume_post_skeleton);
    template <class T> using prepare_penalty_t = decltype(&T::prepare_penalty);
    template <class T> using prepare_cfl_t = decltype(&T::prepare_cfl);
    template <class T> using assemble_volume_t = decltype(&T::assemble_volume);
    template <class T> using assemble_skeleton_t = decltype(&T::assemble_skeleton);
    template <class T> using assemble_boundary_t = decltype(&T::assemble_boundary);
    template <class T>
    using assemble_volume_post_skeleton_t = decltype(&T::assemble_volume_post_skeleton);
    template <class T> using rhs_volume_t = decltype(&T::rhs_volume);
    template <class T> using rhs_skeleton_t = decltype(&T::rhs_skeleton);
    template <class T> using rhs_boundary_t = decltype(&T::rhs_boundary);
    template <class T> using rhs_volume_post_skeleton_t = decltype(&T::rhs_volume_post_skeleton);
    template <class T> using apply_t = decltype(&T::apply);
    template <class T> using flops_apply_t = decltype(&T::flops_apply);
    template <class T> using wave_rhs_t = decltype(&T::wave_rhs);
    template <class T> using project_t = decltype(&T::project);
    template <class T> using cfl_time_step_t = decltype(&T::cfl_time_step);

    DGOperator(std::shared_ptr<DGOperatorTopo> const& topo, std::shared_ptr<LocalOperator> lop)
        : topo_(std::move(topo)), lop_(std::move(lop)),
          scratch_(lop_->scratch_mem_size(), lop_->alignment()),
          scatter_(topo->elementScatterPlan()),
          ghost_(scatter_.recv_prototype<double>(lop_->block_size(), lop_->alignment())) {

        lop_->begin_preparation(topo_->numElements(), topo_->numLocalElements(),
                                topo_->numLocalFacets());
        double relaxation_time_element_local = 0.0;
        for (std::size_t elNo = 0; elNo < topo_->numElements(); ++elNo) {
            lop_->local_relaxation_time(elNo, relaxation_time_element_local);
        }
        MPI_Allreduce(&relaxation_time_element_local, &relaxation_time_global_, 1, MPI_DOUBLE,
                      MPI_MIN, MPI_COMM_WORLD);
        lop_->set_relaxation_time_global(relaxation_time_global_);
        lop_->set_viscoelastic_time_step(relaxation_time_global_);
        if constexpr (std::experimental::is_detected_v<prepare_volume_t, LocalOperator>) {
            for (std::size_t elNo = 0; elNo < topo_->numElements(); ++elNo) {
                scratch_.reset();
                lop_->prepare_volume(elNo, scratch_);
            }
        }
        if constexpr (std::experimental::is_detected_v<prepare_skeleton_t, LocalOperator> ||
                      std::experimental::is_detected_v<prepare_boundary_t, LocalOperator>) {
            for (std::size_t fctNo = 0; fctNo < topo_->numLocalFacets(); ++fctNo) {
                scratch_.reset();
                auto const& info = topo_->info(fctNo);
                auto ib0 = info.g_up[0];
                auto ib1 = info.g_up[1];
                if (info.up[0] != info.up[1]) {
                    lop_->prepare_skeleton(fctNo, info, scratch_);
                } else {
                    lop_->prepare_boundary(fctNo, info, scratch_);
                }
            }
        }
        if constexpr (std::experimental::is_detected_v<prepare_volume_post_skeleton_t,
                                                       LocalOperator>) {
            for (std::size_t elNo = 0; elNo < topo_->numElements(); ++elNo) {
                scratch_.reset();
                lop_->prepare_volume_post_skeleton(elNo, scratch_);
            }
        }
        lop_->end_preparation(topo_->elementScatterPlan());

        if constexpr (std::experimental::is_detected_v<prepare_penalty_t, LocalOperator>) {
            for (std::size_t fctNo = 0; fctNo < topo_->numLocalFacets(); ++fctNo) {
                scratch_.reset();
                auto const& info = topo_->info(fctNo);
                lop_->prepare_penalty(fctNo, info, scratch_);
            }
        }

        if constexpr (std::experimental::is_detected_v<prepare_cfl_t, LocalOperator>) {
            for (std::size_t elNo = 0; elNo < topo_->numLocalElements(); ++elNo) {
                scratch_.reset();
                lop_->prepare_cfl(elNo, topo_->neighbours(elNo), scratch_);
            }
        }
    }

    std::size_t block_size() const override { return lop_->block_size(); }
    DGOperatorTopo const& topo() const override { return *topo_; }

    LocalOperator& lop() { return *lop_; }
    std::size_t num_local_elements() const override { return topo_->numLocalElements(); }
    std::shared_ptr<DGOperatorTopo> topo_ptr() const { return topo_; }

    auto interpolation_operator() -> std::unique_ptr<AbstractInterpolationOperator> override {
        auto lop = lop_->make_interpolation_op();
        using i_lop_t = typename decltype(lop)::element_type;
        return std::make_unique<InterpolationOperator<i_lop_t>>(num_local_elements(),
                                                                std::move(lop));
    }

    void assemble(BlockMatrix& matrix) override {
        auto bs = lop_->block_size();

        auto A_size = LinearAllocator<double>::allocation_size(bs * bs, lop_->alignment());
        auto a_scratch = Scratch<double>(4 * A_size, lop_->alignment());
        auto scratch_matrix = [&bs](LinearAllocator<double>& scratch) {
            double* buffer = scratch.allocate(bs * bs);
            return Matrix<double>(buffer, bs, bs);
        };

        matrix.begin_assembly();
        if constexpr (std::experimental::is_detected_v<assemble_volume_t, LocalOperator>) {
            for (std::size_t elNo = 0; elNo < topo_->numLocalElements(); ++elNo) {
                scratch_.reset();
                a_scratch.reset();
                auto A00 = scratch_matrix(a_scratch);
                if (lop_->assemble_volume(elNo, A00, scratch_)) {
                    matrix.add_block(elNo, elNo, A00);
                }
            }
        }
        if constexpr (std::experimental::is_detected_v<assemble_skeleton_t, LocalOperator> ||
                      std::experimental::is_detected_v<assemble_boundary_t, LocalOperator>) {
            for (std::size_t fctNo = 0; fctNo < topo_->numLocalFacets(); ++fctNo) {
                scratch_.reset();
                a_scratch.reset();
                auto const& info = topo_->info(fctNo);
                auto ib0 = info.up[0];
                auto ib1 = info.up[1];
                if (info.up[0] != info.up[1]) {
                    auto A00 = scratch_matrix(a_scratch);
                    auto A01 = scratch_matrix(a_scratch);
                    auto A10 = scratch_matrix(a_scratch);
                    auto A11 = scratch_matrix(a_scratch);
                    if (lop_->assemble_skeleton(fctNo, info, A00, A01, A10, A11, scratch_)) {
                        if (info.inside[0]) {
                            matrix.add_block(ib0, ib0, A00);
                            matrix.add_block(ib0, ib1, A01);
                        }
                        if (info.inside[1]) {
                            matrix.add_block(ib1, ib0, A10);
                            matrix.add_block(ib1, ib1, A11);
                        }
                    }
                } else {
                    if (info.inside[0]) {
                        auto A00 = scratch_matrix(a_scratch);
                        if (lop_->assemble_boundary(fctNo, info, A00, scratch_)) {
                            matrix.add_block(ib0, ib0, A00);
                        }
                    }
                }
            }
        }
        if constexpr (std::experimental::is_detected_v<assemble_volume_post_skeleton_t,
                                                       LocalOperator>) {
            for (std::size_t elNo = 0; elNo < topo_->numLocalElements(); ++elNo) {
                scratch_.reset();
                a_scratch.reset();
                auto A00 = scratch_matrix(a_scratch);
                if (lop_->assemble_volume_post_skeleton(elNo, A00, scratch_)) {
                    matrix.add_block(elNo, elNo, A00);
                }
            }
        }
        matrix.end_assembly();
    }

    void rhs(BlockVector& vector) override {
        auto bs = lop_->block_size();

        auto b_size = LinearAllocator<double>::allocation_size(bs, lop_->alignment());
        auto a_scratch = Scratch<double>(2 * b_size, lop_->alignment());
        auto sv = [&bs](LinearAllocator<double>& scratch) {
            double* buffer = scratch.allocate(bs);
            return Vector<double>(buffer, bs);
        };

        auto access_handle = vector.begin_access();
        if constexpr (std::experimental::is_detected_v<rhs_volume_t, LocalOperator>) {
            for (std::size_t elNo = 0; elNo < topo_->numLocalElements(); ++elNo) {
                scratch_.reset();
                auto B0 = access_handle.subtensor(slice{}, elNo);
                lop_->rhs_volume(elNo, B0, scratch_);
            }
        }
        if constexpr (std::experimental::is_detected_v<rhs_skeleton_t, LocalOperator> ||
                      std::experimental::is_detected_v<rhs_boundary_t, LocalOperator>) {
            for (std::size_t fctNo = 0; fctNo < topo_->numLocalFacets(); ++fctNo) {
                scratch_.reset();
                a_scratch.reset();
                auto const& info = topo_->info(fctNo);
                auto ib0 = info.up[0];
                auto ib1 = info.up[1];
                if (info.up[0] != info.up[1]) {
                    auto B0 =
                        info.inside[0] ? access_handle.subtensor(slice{}, ib0) : sv(a_scratch);
                    auto B1 =
                        info.inside[1] ? access_handle.subtensor(slice{}, ib1) : sv(a_scratch);
                    lop_->rhs_skeleton(fctNo, info, B0, B1, scratch_);
                } else {
                    if (info.inside[0]) {
                        auto B0 = access_handle.subtensor(slice{}, ib0);
                        lop_->rhs_boundary(fctNo, info, B0, scratch_);
                    }
                }
            }
        }
        if constexpr (std::experimental::is_detected_v<rhs_volume_post_skeleton_t, LocalOperator>) {
            for (std::size_t elNo = 0; elNo < topo_->numLocalElements(); ++elNo) {
                scratch_.reset();
                auto B0 = access_handle.subtensor(slice{}, elNo);
                lop_->rhs_volume_post_skeleton(elNo, B0, scratch_);
            }
        }
        vector.end_access(access_handle);
    }

    void apply(BlockVector const& x, BlockVector& y) override {
        if constexpr (std::experimental::is_detected_v<apply_t, LocalOperator>) {
            apply_(x, y, &LocalOperator::apply);
        }
    }

    void wave_rhs(BlockVector const& x, BlockVector& y) override {
        if constexpr (std::experimental::is_detected_v<wave_rhs_t, LocalOperator>) {
            apply_(x, y, &LocalOperator::wave_rhs);
        }
    }

    void project(typename base::volume_functional_t x, BlockVector& y) override {
        auto y_handle = y.begin_access();
        if constexpr (std::experimental::is_detected_v<project_t, LocalOperator>) {
            for (std::size_t elNo = 0, num = topo_->numLocalElements(); elNo < num; ++elNo) {
                auto y_block = y_handle.subtensor(slice{}, elNo);
                lop_->project(elNo, x, y_block);
            }
        }
        y.end_access(y_handle);
    }

    double local_cfl_time_step() const override {
        double dt = std::numeric_limits<double>::max();
        if constexpr (std::experimental::is_detected_v<cfl_time_step_t, LocalOperator>) {
            for (std::size_t elNo = 0, num = num_local_elements(); elNo < num; ++elNo) {
                dt = std::min(dt, lop_->cfl_time_step(elNo));
            }
        }
        return dt;
    }

    std::size_t flops_apply() const override {
        std::size_t flops = 0;
        if constexpr (std::experimental::is_detected_v<flops_apply_t, LocalOperator>) {
            for (std::size_t elNo = 0; elNo < topo_->numLocalElements(); ++elNo) {
                flops += lop_->flops_apply(elNo, topo_->neighbours(elNo));
            }
        }
        return flops;
    }

    template <typename Iterator>
    auto solution(BlockVector const& vector, Iterator first, Iterator last) const {
        auto num_elements = std::distance(first, last);
        auto soln = lop_->solution_prototype(num_elements);
        auto& values = soln.values();
        auto value_matrix = reshape(values, lop_->block_size(), num_elements);

        auto access_handle = vector.begin_access_readonly();
        std::size_t out_no = 0;
        for (; first != last; ++first) {
            std::size_t elNo = *first;
            assert(elNo < num_local_elements());

            auto value_matrix = values.subtensor(slice{}, slice{}, out_no++);
            auto state_block = access_handle.subtensor(slice{}, elNo);
            auto state_matrix = reshape(state_block, value_matrix.shape(0), value_matrix.shape(1));
            value_matrix.copy_values(state_matrix);
        }
        vector.end_access_readonly(access_handle);

        return soln;
    }

    auto solution(BlockVector const& vector, std::vector<std::size_t> const& subset)
        -> FiniteElementFunction<LocalOperator::Dim> override {
        return solution(vector, subset.begin(), subset.end());
    }
    auto solution(BlockVector const& vector, std::optional<Range<std::size_t>> range = std::nullopt)
        -> FiniteElementFunction<LocalOperator::Dim> override {
        if (!range) {
            *range = Range<std::size_t>(0, num_local_elements());
        }
        return solution(vector, range->begin(), range->end());
    }

    template <typename Iterator> auto params(Iterator first, Iterator last) {
        auto num_elements = std::distance(first, last);
        auto coeffs = lop_->coefficients_prototype(num_elements);
        auto& values = coeffs.values();

        auto scratch = Scratch<double>(lop_->scratch_mem_size(), lop_->alignment());
        std::size_t out_no = 0;
        for (; first != last; ++first) {
            std::size_t elNo = *first;
            assert(elNo < num_local_elements());

            scratch.reset();
            auto C = values.subtensor(slice{}, slice{}, out_no++);
            lop_->coefficients_volume(elNo, C, scratch);
        }
        return coeffs;
    }

    auto params(std::vector<std::size_t> const& subset)
        -> FiniteElementFunction<LocalOperator::Dim> override {
        return params(subset.begin(), subset.end());
    }

    auto params(std::optional<Range<std::size_t>> range = std::nullopt)
        -> FiniteElementFunction<LocalOperator::Dim> override {
        if (!range) {
            *range = Range<std::size_t>(0, num_local_elements());
        }
        return params(range->begin(), range->end());
    }

    void set_force(typename base::volume_functional_t fun) override {
        lop_->set_force(std::move(fun));
    }
    void set_slip(typename base::facet_functional_t fun) override {
        lop_->set_slip(std::move(fun));
    }
    void set_dirichlet(typename base::facet_functional_t fun) override {
        lop_->set_dirichlet(std::move(fun));
    }

private:
    using apply_fun_ptr = void (LocalOperator::*)(
        std::size_t, mneme::span<SideInfo>, Vector<double const> const&,
        std::array<Vector<double const>, NumFacets> const&, Vector<double>&) const;

    void apply_(BlockVector const& x, BlockVector& y, apply_fun_ptr apply_fun) {
        auto y_handle = y.begin_access();

        auto copy_first = topo_->numInteriorElements();
        auto ghost_first = topo_->numLocalElements();
        auto block_view = LocalGhostCompositeView(x, ghost_);

        const auto lop_apply = [&](std::size_t elNo) {
            auto y_0 = y_handle.subtensor(slice{}, elNo);
            auto x_0 = block_view.get_block(elNo);
            auto info = topo_->neighbours(elNo);
            assert(info.size() == NumFacets);
            std::array<decltype(x_0), NumFacets> x_n;
            for (std::size_t d = 0; d < NumFacets; ++d) {
                x_n[d] = block_view.get_block(info[d].lid);
            }
            ((lop_.get())->*apply_fun)(elNo, info, x_0, x_n, y_0);
        };

        scatter_.begin_scatter(x, ghost_);

        for (std::size_t elNo = 0; elNo < copy_first; ++elNo) {
            lop_apply(elNo);
            scatter_.test_scatter();
        }

        scatter_.wait_scatter();

        for (std::size_t elNo = copy_first; elNo < ghost_first; ++elNo) {
            lop_apply(elNo);
        }

        y.end_access(y_handle);
    }

    std::shared_ptr<DGOperatorTopo> topo_;
    std::shared_ptr<LocalOperator> lop_;
    Scratch<double> scratch_;
    Scatter scatter_;
    SparseBlockVector<double> ghost_;
    double relaxation_time_global_ = 0.0;
};

} // namespace tndm

#endif // DGOPERATOR_20200909_H
