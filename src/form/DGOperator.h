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

    template <class T> using prepare_volume_t = decltype(&T::prepare_volume);
    template <class T> using prepare_skeleton_t = decltype(&T::prepare_skeleton);
    template <class T> using prepare_boundary_t = decltype(&T::prepare_boundary);
    template <class T>
    using prepare_volume_post_skeleton_t = decltype(&T::prepare_volume_post_skeleton);
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
    template <class T> using apply_inverse_mass_t = decltype(&T::apply_inverse_mass);
    template <class T> using project_t = decltype(&T::project);

    DGOperator(std::shared_ptr<DGOperatorTopo> const& topo, std::shared_ptr<LocalOperator> lop)
        : topo_(std::move(topo)), lop_(std::move(lop)),
          scratch_(lop_->scratch_mem_size(), lop_->alignment()),
          scatter_(topo->elementScatterPlan()),
          ghost_(scatter_.recv_prototype<double>(lop_->block_size(), lop_->alignment())) {

        lop_->begin_preparation(topo_->numElements(), topo_->numLocalElements(),
                                topo_->numLocalFacets());
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
        constexpr std::size_t NumFacets = LocalOperator::Dim + 1;
        auto y_handle = y.begin_access();
        if constexpr (std::experimental::is_detected_v<apply_t, LocalOperator>) {
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
                lop_->apply(elNo, info, x_0, x_n, y_0);
            };

            scatter_.begin_scatter(x, ghost_);

            for (std::size_t elNo = 0; elNo < copy_first; ++elNo) {
                lop_apply(elNo);
            }

            scatter_.wait_scatter();

            for (std::size_t elNo = copy_first; elNo < ghost_first; ++elNo) {
                lop_apply(elNo);
            }
        }
        y.end_access(y_handle);
    }

    void apply_inverse_mass(BlockVector const& x, BlockVector& y) override {
        auto x_handle = x.begin_access_readonly();
        auto y_handle = y.begin_access();
        if constexpr (std::experimental::is_detected_v<apply_inverse_mass_t, LocalOperator>) {

            for (std::size_t elNo = 0, num = topo_->numLocalElements(); elNo < num; ++elNo) {
                auto y_block = y_handle.subtensor(slice{}, elNo);
                auto x_block = x_handle.subtensor(slice{}, elNo);
                lop_->apply_inverse_mass(elNo, x_block, y_block);
            }
        }
        y.end_access(y_handle);
        x.end_access_readonly(x_handle);
    }

    void project(typename base::volume_functional_t x, BlockVector& y) override {
        auto y_handle = y.begin_access();
        if constexpr (std::experimental::is_detected_v<apply_inverse_mass_t, LocalOperator>) {

            for (std::size_t elNo = 0, num = topo_->numLocalElements(); elNo < num; ++elNo) {
                auto y_block = y_handle.subtensor(slice{}, elNo);
                lop_->project(elNo, x, y_block);
            }
        }
        y.end_access(y_handle);
    }

    std::size_t flops_apply() {
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

    auto coefficients() const {
        auto coeffs = lop_->coefficients_prototype(topo_->numLocalElements());
        auto& values = coeffs.values();

        auto scratch = Scratch<double>(lop_->scratch_mem_size(), lop_->alignment());
        for (std::size_t elNo = 0; elNo < topo_->numLocalElements(); ++elNo) {
            scratch.reset();
            auto C = values.subtensor(slice{}, slice{}, elNo);
            lop_->coefficients_volume(elNo, C, scratch);
        }
        return coeffs;
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
    std::shared_ptr<DGOperatorTopo> topo_;
    std::shared_ptr<LocalOperator> lop_;
    Scratch<double> scratch_;
    Scatter scatter_;
    SparseBlockVector<double> ghost_;
};

} // namespace tndm

#endif // DGOPERATOR_20200909_H
