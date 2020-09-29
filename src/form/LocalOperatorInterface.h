#ifndef LOCALOPERATORINTERFACE_20200910_H
#define LOCALOPERATORINTERFACE_20200910_H

#include "form/FacetInfo.h"
#include "parallel/Scatter.h"
#include "tensor/Tensor.h"
#include "util/LinearAllocator.h"

#include <cstddef>

namespace tndm {

class LocalOperatorInterface {
public:
    std::size_t block_size() const { return 1; }
    std::size_t scratch_mem_size() const { return 1; }

    void begin_preparation(std::size_t numElements, std::size_t numLocalElements,
                           std::size_t numLocalFacets) {}
    void prepare_volume(std::size_t elNo, LinearAllocator& scratch) {}
    void prepare_skeleton(std::size_t fctNo, FacetInfo const& info, LinearAllocator& scratch) {}
    void prepare_boundary(std::size_t fctNo, FacetInfo const& info, LinearAllocator& scratch) {}
    void prepare_volume_post_skeleton(std::size_t elNo, LinearAllocator& scratch) {}
    void end_preparation(Scatter& elementScatter) {}

    bool assemble_volume(std::size_t elNo, Matrix<double>& A00, LinearAllocator& scratch) const {
        return false;
    }
    bool assemble_skeleton(std::size_t fctNo, FacetInfo const& info, Matrix<double>& A00,
                           Matrix<double>& A01, Matrix<double>& A10, Matrix<double>& A11,
                           LinearAllocator& scratch) const {
        return false;
    }
    bool assemble_boundary(std::size_t fctNo, FacetInfo const& info, Matrix<double>& A00,
                           Matrix<double>& A01, LinearAllocator& scratch) const {
        return false;
    }
    bool assemble_volume_post_skeleton(std::size_t elNo, Matrix<double>& A00,
                                       LinearAllocator& scratch) const {
        return false;
    }

    bool rhs_volume(std::size_t elNo, Vector<double>& B, LinearAllocator& scratch) const {
        return false;
    }
    bool rhs_skeleton(std::size_t fctNo, FacetInfo const& info, Vector<double>& B0,
                      Vector<double>& B1, LinearAllocator& scratch) const {
        return false;
    }
    bool rhs_boundary(std::size_t fctNo, FacetInfo const& info, Vector<double>& B0,
                      LinearAllocator& scratch) const {
        return false;
    }
    bool rhs_volume_post_skeleton(std::size_t elNo, Vector<double>& B,
                                  LinearAllocator& scratch) const {
        return false;
    }

    auto solution_prototype(std::size_t numLocalElements) const;
    auto coefficients_prototype(std::size_t numLocalElements) const;
    void coefficients_volume(std::size_t elNo, Matrix<double>& C, LinearAllocator& scratch) const;
};

} // namespace tndm

#endif // LOCALOPERATORINTERFACE_20200910_H
