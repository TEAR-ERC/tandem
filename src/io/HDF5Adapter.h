#ifndef HDF5ADAPTER_20231010_H
#define HDF5ADAPTER_20231010_H

#include "Endianness.h"
#include "VTUWriter.h"
#include "form/FiniteElementFunction.h"
#include "geometry/Curvilinear.h"
#include "io/VTUAdapter.h"
#include "mesh/LocalSimplexMesh.h"
#include "tensor/Managed.h"
#include "tensor/Tensor.h"
#include "tensor/TensorBase.h"
#include "util/LinearAllocator.h"
#include "util/Range.h"
#include "util/Scratch.h"

#include <array>
#include <cstddef>
#include <hdf5.h>
#include <memory>
#include <vector>

namespace tndm {

template <std::size_t D> class HDF5Adapter {
public:
    virtual ~HDF5Adapter() = default;

    /**
     * @brief Get the number of elements in the mesh.
     * @return The number of elements.
     */
    virtual std::size_t numElements() const = 0;

    /**
     * @brief Get the dimension of the points (e.g., 2 for 2D, 3 for 3D).
     * @return The dimension of the points.
     */
    virtual std::size_t pointDim() const = 0;

    /**
     * @brief Set the reference nodes for the mesh.
     * @param refNodes The reference nodes to set.
     */
    virtual void setRefNodes(std::vector<std::array<double, D - 1u>> const& refNodes) = 0;

    /**
     * @brief Get the vertices of the mesh.
     * @param vertices Output vector to store the vertices.
     */
    virtual std::vector<double> getVertices() = 0;

    /**
     * @brief Get the face connectivity numbers (fctNos) of the mesh.
     * @param fctNos Output vector to store the face connectivity numbers.
     */
};

template <std::size_t D> class CurvilinearBoundaryHDF5Adapter : public HDF5Adapter<D> {
public:
    CurvilinearBoundaryHDF5Adapter(LocalSimplexMesh<D> const& mesh,
                                   std::shared_ptr<Curvilinear<D>> cl,
                                   std::vector<std::size_t> const& fctNos, unsigned degree)
        : cl_(std::move(cl)),
          refNodes_(EquidistantNodesFactory<D - 1>(NumberingConvention::VTK)(degree)) {
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
        setRefNodes(refNodes_);
        numPoints_ = refNodes_.size();
    }

    std::size_t numElements() const override { return bnds_.size(); }
    std::size_t pointDim() const override { return D; }

    // void setRefNodes(std::vector<std::array<double, D - 1u>> const& points) override {}
    void map(std::size_t no, Tensor<double, 2u>& result) const {
        assert(no < bnds_.size());
        auto const& bnd = bnds_[no];
        cl_->map(bnd.first, E_[bnd.second], result);
    }
    void setRefNodes(std::vector<std::array<double, D - 1u>> const& points) override {
        numPoints_ = points.size();
        E_.clear();
        for (std::size_t f = 0; f < D + 1u; ++f) {
            auto facetParam = cl_->facetParam(f, points);
            E_.emplace_back(cl_->evaluateBasisAt(facetParam));
        }
    }
    std::vector<double> getVertices() override {
        const std::size_t totalValues = D * numElements() * numPoints_;
        const std::size_t triangleVertices = 3;
        faultVertices.resize(totalValues);

        for (std::size_t elNo = 0; elNo < numElements(); ++elNo) {
            // Get pointer to the position for this element's vertices
            double* elementStart = &faultVertices[elNo * D * triangleVertices];

            // Create matrix view of the storage location
            auto result = Matrix<double>(elementStart, D, numPoints_);

            // Map the vertices directly into the array
            map(elNo, result);
        }
        return faultVertices;
    }

private:
    std::shared_ptr<Curvilinear<D>> cl_;
    std::size_t numPoints_ = 0;
    std::vector<Managed<Matrix<double>>> E_;
    std::vector<std::array<double, D - 1u>> refNodes_;
    std::vector<std::array<double, D>> refNodes3D_;
    std::vector<std::pair<std::size_t, int>> bnds_;
    std::vector<double> faultVertices;
    
};

} // namespace tndm

#endif // HDF5ADAPTER_20231010_H