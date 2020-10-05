#ifndef VTUADAPTER_20200827_H
#define VTUADAPTER_20200827_H

#include "geometry/Curvilinear.h"
#include "mesh/LocalSimplexMesh.h"
#include "util/Range.h"

#include <cstddef>
#include <vector>

namespace tndm {

template <std::size_t D> class VTUAdapter {
public:
    virtual ~VTUAdapter() {}

    virtual std::size_t numElements() const = 0;
    virtual std::size_t pointDim() const = 0;
    virtual void setRefNodes(std::vector<std::array<double, D>> const& refNodes) = 0;
    virtual void map(std::size_t elNo, Tensor<double, 2u>& result) const = 0;
};

template <std::size_t D> class CurvilinearVTUAdapter : public VTUAdapter<D> {
public:
    CurvilinearVTUAdapter(Curvilinear<D> const& cl, Range<std::size_t> elementRange)
        : cl_(&cl), range_(std::move(elementRange)) {
        assert(range_.from <= cl_->numElements() && range_.to <= cl_->numElements());
    }

    std::size_t numElements() const override { return range_.size(); }
    std::size_t pointDim() const override { return D; };
    void setRefNodes(std::vector<std::array<double, D>> const& points) override {
        E_ = cl_->evaluateBasisAt(points);
    }
    void map(std::size_t elNo, Tensor<double, 2u>& result) const override {
        cl_->map(elNo + range_.from, E_, result);
    }

private:
    Curvilinear<D> const* cl_;
    Range<std::size_t> range_;
    Managed<Matrix<double>> E_;
};

template <std::size_t D> class CurvilinearBoundaryVTUAdapter : public VTUAdapter<D - 1u> {
public:
    CurvilinearBoundaryVTUAdapter(LocalSimplexMesh<D> const& mesh, Curvilinear<D> const& cl,
                                  std::vector<std::size_t> const& fctNos)
        : cl_(&cl) {
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
        E_.clear();
        for (std::size_t f = 0; f < D + 1u; ++f) {
            E_.emplace_back(cl_->evaluateBasisAt(cl_->facetParam(f, points)));
        }
    }
    void map(std::size_t no, Tensor<double, 2u>& result) const override {
        assert(no < bnds_.size());
        auto const& bnd = bnds_[no];
        cl_->map(bnd.first, E_[bnd.second], result);
    }

private:
    Curvilinear<D> const* cl_;
    std::vector<std::pair<std::size_t, int>> bnds_;
    std::vector<Managed<Matrix<double>>> E_;
};

} // namespace tndm

#endif // VTUADAPTER_20200827_H
