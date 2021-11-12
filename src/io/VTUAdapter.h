#ifndef VTUADAPTER_20200827_H
#define VTUADAPTER_20200827_H

#include "geometry/Curvilinear.h"
#include "mesh/LocalSimplexMesh.h"
#include "tensor/Tensor.h"
#include "tensor/TensorBase.h"
#include "tensor/Utility.h"
#include "util/LinearAllocator.h"
#include "util/Range.h"

#include <cstddef>
#include <memory>
#include <vector>

namespace tndm {

template <std::size_t D> class VTUAdapter {
public:
    virtual ~VTUAdapter() {}

    virtual std::size_t numElements() const = 0;
    virtual std::size_t pointDim() const = 0;
    virtual void setRefNodes(std::vector<std::array<double, D>> const& refNodes) = 0;
    virtual void map(std::size_t elNo, Tensor<double, 2u>& result) const = 0;

    virtual std::size_t scratch_mem_size() const = 0;
    virtual TensorBase<Tensor<double, 3u>> jacobianResultInfo() const = 0;
    virtual void jacobianInv(std::size_t elNo, Tensor<double, 3u>& result,
                             LinearAllocator<double>& scratch) const = 0;
};

template <std::size_t D> class CurvilinearVTUAdapter : public VTUAdapter<D> {
public:
    CurvilinearVTUAdapter(std::shared_ptr<Curvilinear<D>> cl, Range<std::size_t> elementRange)
        : cl_(std::move(cl)), range_(std::move(elementRange)) {
        assert(range_.from <= cl_->numElements() && range_.to <= cl_->numElements());
    }

    std::size_t numElements() const override { return range_.size(); }
    std::size_t pointDim() const override { return D; };
    void setRefNodes(std::vector<std::array<double, D>> const& points) override {
        numPoints_ = points.size();
        E_ = cl_->evaluateBasisAt(points);
        Dxi_ = cl_->evaluateGradientAt(points);
    }
    void map(std::size_t elNo, Tensor<double, 2u>& result) const override {
        cl_->map(elNo + range_.from, E_, result);
    }

    std::size_t scratch_mem_size() const override {
        return Tensor<double, 3u>(nullptr, jacobianResultInfo()).size();
    }
    TensorBase<Tensor<double, 3u>> jacobianResultInfo() const override {
        return cl_->jacobianResultInfo(numPoints_);
    }
    void jacobianInv(std::size_t elNo, Tensor<double, 3u>& result,
                     LinearAllocator<double>& scratch) const override {
        auto J = make_scratch_tensor(scratch, result);
        cl_->jacobian(elNo, Dxi_, J);
        cl_->jacobianInv(J, result);
    }

private:
    std::shared_ptr<Curvilinear<D>> cl_;
    Range<std::size_t> range_;
    std::size_t numPoints_ = 0;
    Managed<Matrix<double>> E_;
    Managed<Tensor<double, 3u>> Dxi_;
};

template <std::size_t D> class CurvilinearBoundaryVTUAdapter : public VTUAdapter<D - 1u> {
public:
    CurvilinearBoundaryVTUAdapter(LocalSimplexMesh<D> const& mesh,
                                  std::shared_ptr<Curvilinear<D>> cl,
                                  std::vector<std::size_t> const& fctNos)
        : cl_(std::move(cl)) {
        bnds_.reserve(fctNos.size());
        for (auto const& fctNo : fctNos) {
            auto elNos = mesh.template upward<D - 1u>(fctNo);
            assert(elNos.size() >= 1u);
            auto elNo = elNos[0];
            assert(elNo < cl_->numElements());
            auto dws = mesh.template downward<D - 1u, D>(elNo);
            int localFaceNo = std::distance(dws.begin(), std::find(dws.begin(), dws.end(), fctNo));
            assert(localFaceNo < D + 1u);
            bnds_.emplace_back(std::make_pair(elNo, localFaceNo));
        }
    }

    std::size_t numElements() const override { return bnds_.size(); }
    std::size_t pointDim() const override { return D; };
    void setRefNodes(std::vector<std::array<double, D - 1u>> const& points) override {
        numPoints_ = points.size();
        E_.clear();
        Dxi_.clear();
        for (std::size_t f = 0; f < D + 1u; ++f) {
            auto facetParam = cl_->facetParam(f, points);
            E_.emplace_back(cl_->evaluateBasisAt(facetParam));
            Dxi_.emplace_back(cl_->evaluateGradientAt(facetParam));
        }
    }
    void map(std::size_t no, Tensor<double, 2u>& result) const override {
        assert(no < bnds_.size());
        auto const& bnd = bnds_[no];
        cl_->map(bnd.first, E_[bnd.second], result);
    }

    std::size_t scratch_mem_size() const override {
        return Tensor<double, 3u>(nullptr, jacobianResultInfo()).size();
    }
    TensorBase<Tensor<double, 3u>> jacobianResultInfo() const override {
        return cl_->jacobianResultInfo(numPoints_);
    }
    void jacobianInv(std::size_t no, Tensor<double, 3u>& result,
                     LinearAllocator<double>& scratch) const override {
        assert(no < bnds_.size());
        auto const& bnd = bnds_[no];
        auto J = make_scratch_tensor(scratch, result);
        cl_->jacobian(bnd.first, Dxi_[bnd.second], J);
        cl_->jacobianInv(J, result);
    }

private:
    std::shared_ptr<Curvilinear<D>> cl_;
    std::vector<std::pair<std::size_t, int>> bnds_;
    std::size_t numPoints_;
    std::vector<Managed<Matrix<double>>> E_;
    std::vector<Managed<Tensor<double, 3u>>> Dxi_;
};

} // namespace tndm

#endif // VTUADAPTER_20200827_H
