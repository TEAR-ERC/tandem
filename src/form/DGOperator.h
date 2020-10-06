#ifndef DGOPERATOR_20200909_H
#define DGOPERATOR_20200909_H

#include "form/DGOperatorTopo.h"
#include "form/FiniteElementFunction.h"
#include "tensor/Managed.h"
#include "tensor/Reshape.h"
#include "tensor/Tensor.h"
#include "util/LinearAllocator.h"

#include <cassert>
#include <experimental/type_traits>
#include <memory>
#include <utility>

namespace tndm {

template <typename LocalOperator> class DGOperator {
public:
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

    DGOperator(std::shared_ptr<DGOperatorTopo> const& topo, std::unique_ptr<LocalOperator> lop)
        : topo_(std::move(topo)), lop_(std::move(lop)) {
        scratch_mem_ = std::make_unique<char[]>(lop_->scratch_mem_size());

        auto scratch = make_scratch();
        lop_->begin_preparation(topo_->numElements(), topo_->numLocalElements(),
                                topo_->numLocalFacets());
        if constexpr (std::experimental::is_detected_v<prepare_volume_t, LocalOperator>) {
            for (std::size_t elNo = 0; elNo < topo_->numElements(); ++elNo) {
                scratch.reset();
                lop_->prepare_volume(elNo, scratch);
            }
        }
        if constexpr (std::experimental::is_detected_v<prepare_skeleton_t, LocalOperator> ||
                      std::experimental::is_detected_v<prepare_boundary_t, LocalOperator>) {
            for (std::size_t fctNo = 0; fctNo < topo_->numLocalFacets(); ++fctNo) {
                scratch.reset();
                auto const& info = topo_->info(fctNo);
                auto ib0 = info.g_up[0];
                auto ib1 = info.g_up[1];
                if (info.up[0] != info.up[1]) {
                    lop_->prepare_skeleton(fctNo, info, scratch);
                } else {
                    lop_->prepare_boundary(fctNo, info, scratch);
                }
            }
        }
        if constexpr (std::experimental::is_detected_v<prepare_volume_post_skeleton_t,
                                                       LocalOperator>) {
            for (std::size_t elNo = 0; elNo < topo_->numElements(); ++elNo) {
                scratch.reset();
                lop_->prepare_volume_post_skeleton(elNo, scratch);
            }
        }
        lop_->end_preparation(topo_->elementScatter());
    }

    LocalOperator& lop() { return *lop_; }
    std::size_t block_size() const { return lop_->block_size(); }
    std::size_t numLocalElements() const { return topo_->numLocalElements(); }
    DGOperatorTopo const& topo() const { return *topo_; }

    template <typename BlockMatrix> void assemble(BlockMatrix& matrix) {
        auto bs = lop_->block_size();

        auto a_scratch_mem_size = 4 * bs * bs * sizeof(double);
        auto a_scratch_mem = std::make_unique<char[]>(a_scratch_mem_size);
        auto a_scratch =
            LinearAllocator(a_scratch_mem.get(), a_scratch_mem.get() + a_scratch_mem_size);

        auto scratch_matrix = [&bs](LinearAllocator& scratch) {
            double* buffer = scratch.allocate<double>(bs * bs);
            return Matrix<double>(buffer, bs, bs);
        };

        auto l_scratch = make_scratch();
        matrix.begin_assembly();
        if constexpr (std::experimental::is_detected_v<assemble_volume_t, LocalOperator>) {
            for (std::size_t elNo = 0; elNo < topo_->numLocalElements(); ++elNo) {
                l_scratch.reset();
                a_scratch.reset();
                auto ib = topo_->gid(elNo);
                auto A00 = scratch_matrix(a_scratch);
                if (lop_->assemble_volume(elNo, A00, l_scratch)) {
                    matrix.add_block(ib, ib, A00);
                }
            }
        }
        if constexpr (std::experimental::is_detected_v<assemble_skeleton_t, LocalOperator> ||
                      std::experimental::is_detected_v<assemble_boundary_t, LocalOperator>) {
            for (std::size_t fctNo = 0; fctNo < topo_->numLocalFacets(); ++fctNo) {
                l_scratch.reset();
                a_scratch.reset();
                auto const& info = topo_->info(fctNo);
                auto ib0 = info.g_up[0];
                auto ib1 = info.g_up[1];
                if (info.up[0] != info.up[1]) {
                    auto A00 = scratch_matrix(a_scratch);
                    auto A01 = scratch_matrix(a_scratch);
                    auto A10 = scratch_matrix(a_scratch);
                    auto A11 = scratch_matrix(a_scratch);
                    if (lop_->assemble_skeleton(fctNo, info, A00, A01, A10, A11, l_scratch)) {
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
                        if (lop_->assemble_boundary(fctNo, info, A00, l_scratch)) {
                            matrix.add_block(ib0, ib0, A00);
                        }
                    }
                }
            }
        }
        if constexpr (std::experimental::is_detected_v<assemble_volume_post_skeleton_t,
                                                       LocalOperator>) {
            for (std::size_t elNo = 0; elNo < topo_->numLocalElements(); ++elNo) {
                l_scratch.reset();
                a_scratch.reset();
                auto ib = topo_->gid(elNo);
                auto A00 = scratch_matrix(a_scratch);
                if (lop_->assemble_volume_post_skeleton(elNo, A00, l_scratch)) {
                    matrix.add_block(ib, ib, A00);
                }
            }
        }
        matrix.end_assembly();
    }

    template <typename BlockVector> void rhs(BlockVector& vector) {
        auto bs = lop_->block_size();

        auto a_scratch_mem_size = 2 * bs * sizeof(double);
        auto a_scratch_mem = std::make_unique<char[]>(a_scratch_mem_size);
        auto a_scratch =
            LinearAllocator(a_scratch_mem.get(), a_scratch_mem.get() + a_scratch_mem_size);

        auto sv = [&bs](LinearAllocator& scratch) {
            double* buffer = scratch.allocate<double>(bs);
            return Vector<double>(buffer, bs);
        };

        auto l_scratch = make_scratch();
        auto access_handle = vector.begin_access();
        if constexpr (std::experimental::is_detected_v<rhs_volume_t, LocalOperator>) {
            for (std::size_t elNo = 0; elNo < topo_->numLocalElements(); ++elNo) {
                l_scratch.reset();
                auto B0 = vector.get_block(access_handle, elNo);
                lop_->rhs_volume(elNo, B0, l_scratch);
            }
        }
        if constexpr (std::experimental::is_detected_v<rhs_skeleton_t, LocalOperator> ||
                      std::experimental::is_detected_v<rhs_boundary_t, LocalOperator>) {
            for (std::size_t fctNo = 0; fctNo < topo_->numLocalFacets(); ++fctNo) {
                l_scratch.reset();
                a_scratch.reset();
                auto const& info = topo_->info(fctNo);
                auto ib0 = info.up[0];
                auto ib1 = info.up[1];
                if (info.up[0] != info.up[1]) {
                    auto B0 = info.inside[0] ? vector.get_block(access_handle, ib0) : sv(a_scratch);
                    auto B1 = info.inside[1] ? vector.get_block(access_handle, ib1) : sv(a_scratch);
                    lop_->rhs_skeleton(fctNo, info, B0, B1, l_scratch);
                } else {
                    if (info.inside[0]) {
                        auto B0 = vector.get_block(access_handle, ib0);
                        lop_->rhs_boundary(fctNo, info, B0, l_scratch);
                    }
                }
            }
        }
        if constexpr (std::experimental::is_detected_v<rhs_volume_post_skeleton_t, LocalOperator>) {
            for (std::size_t elNo = 0; elNo < topo_->numLocalElements(); ++elNo) {
                l_scratch.reset();
                auto B0 = vector.get_block(access_handle, elNo);
                lop_->rhs_volume_post_skeleton(elNo, B0, l_scratch);
            }
        }
        vector.end_access(access_handle);
    }

    template <typename BlockVector> auto solution(BlockVector& vector) const {
        auto soln = lop_->solution_prototype(topo_->numLocalElements());
        auto& values = soln.values();
        auto value_matrix = reshape(values, lop_->block_size(), topo_->numLocalElements());

        auto access_handle = vector.begin_access_readonly();
        for (std::size_t elNo = 0; elNo < topo_->numLocalElements(); ++elNo) {
            auto block = value_matrix.subtensor(slice{}, elNo);
            vector.copy(access_handle, elNo, block);
        }
        vector.end_access_readonly(access_handle);
        return soln;
    }

    auto coefficients() const {
        auto coeffs = lop_->coefficients_prototype(topo_->numLocalElements());
        auto& values = coeffs.values();

        auto scratch = make_scratch();
        for (std::size_t elNo = 0; elNo < topo_->numLocalElements(); ++elNo) {
            scratch.reset();
            auto C = values.subtensor(slice{}, slice{}, elNo);
            lop_->coefficients_volume(elNo, C, scratch);
        }
        return coeffs;
    }

private:
    auto make_scratch() const {
        return LinearAllocator(scratch_mem_.get(), scratch_mem_.get() + lop_->scratch_mem_size());
    }
    std::shared_ptr<DGOperatorTopo> topo_;
    std::unique_ptr<LocalOperator> lop_;
    std::unique_ptr<char[]> scratch_mem_;
};

} // namespace tndm

#endif // DGOPERATOR_20200909_H
