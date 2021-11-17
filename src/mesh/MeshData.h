#ifndef MESHDATA_H
#define MESHDATA_H

#include "form/BC.h"
#include "parallel/CommPattern.h"
#include "parallel/MPITraits.h"
#include "tensor/Managed.h"
#include "tensor/Tensor.h"
#include "util/Algorithm.h"

#include <mpi.h>

#include <array>
#include <cassert>
#include <cstddef>
#include <memory>
#include <vector>

namespace tndm {

class MeshData {
public:
    virtual ~MeshData() {}
    virtual std::size_t size() const = 0;
    virtual std::unique_ptr<MeshData> redistributed(std::vector<std::size_t> const& lids,
                                                    AllToAllV const& a2a) const = 0;
    virtual void permute(std::vector<std::size_t> const& permutation) = 0;
};

template <std::size_t SpaceD> class VertexData : public MeshData {
public:
    using vertex_t = std::array<double, SpaceD>;

    VertexData(std::vector<vertex_t>&& vertices) : vertices(std::move(vertices)) {}
    virtual ~VertexData() {}

    std::size_t size() const override { return vertices.size(); }

    std::unique_ptr<MeshData> redistributed(std::vector<std::size_t> const& lids,
                                            AllToAllV const& a2a) const override {
        std::vector<vertex_t> requestedVertices;
        requestedVertices.reserve(lids.size());
        for (auto& lid : lids) {
            assert(lid < vertices.size());
            requestedVertices.emplace_back(vertices[lid]);
        }

        mpi_array_type<double> mpi_t(SpaceD);
        auto newVertices = a2a.exchange(requestedVertices, mpi_t.get());
        return std::make_unique<VertexData>(std::move(newVertices));
    }

    void permute(std::vector<std::size_t> const& permutation) override {
        apply_permutation(vertices, permutation);
    }

    std::vector<vertex_t> const& getVertices() const { return vertices; }

private:
    std::vector<vertex_t> vertices;
};

class BoundaryData : public MeshData {
public:
    BoundaryData(std::vector<BC>&& BCs) : boundaryConditions(std::move(BCs)) {}
    virtual ~BoundaryData() {}

    std::size_t size() const override { return boundaryConditions.size(); }

    std::unique_ptr<MeshData> redistributed(std::vector<std::size_t> const& lids,
                                            AllToAllV const& a2a) const override {
        std::vector<BC> requestedBCs;
        requestedBCs.reserve(lids.size());
        for (auto& lid : lids) {
            if (lid == std::numeric_limits<std::size_t>::max()) {
                requestedBCs.emplace_back(BC::None);
            } else {
                requestedBCs.emplace_back(boundaryConditions[lid]);
            }
        }

        static_assert(sizeof(BC::None) == sizeof(int));
        auto newBCs = a2a.exchange(requestedBCs, mpi_type_t<int>());
        return std::make_unique<BoundaryData>(std::move(newBCs));
    }

    void permute(std::vector<std::size_t> const& permutation) override {
        apply_permutation(boundaryConditions, permutation);
    }

    std::vector<BC> const& getBoundaryConditions() const { return boundaryConditions; }

private:
    std::vector<BC> boundaryConditions;
};

class ElementData : public MeshData {
public:
    using nodes_t = Managed<Tensor<double, 3u>>;
    constexpr static std::size_t ElementMode = 2;

    ElementData(nodes_t&& nodes) : nodes_(std::move(nodes)) {}
    virtual ~ElementData() {}

    std::size_t size() const override { return nodes_.size(); }

    std::unique_ptr<MeshData> redistributed(std::vector<std::size_t> const& lids,
                                            AllToAllV const& a2a) const override {
        auto shape = nodes_.shape();
        shape[ElementMode] = lids.size();
        auto requestedNodes = nodes_t(shape);
        std::size_t I = nodes_.shape(0), J = nodes_.shape(1), K = nodes_.shape(2);
        std::size_t k = 0;
        for (auto& lid : lids) {
            assert(lid < K);
            for (std::size_t j = 0; j < J; ++j) {
                for (std::size_t i = 0; i < I; ++i) {
                    requestedNodes(i, j, k) = nodes_(i, j, lid);
                }
            }
            ++k;
        }

        auto newNodes = a2a.exchange(requestedNodes);
        return std::make_unique<ElementData>(std::move(newNodes));
    }

    void permute(std::vector<std::size_t> const& permutation) override {
        apply_permutation(nodes_, permutation, ElementMode);
    }

    nodes_t const& getNodes() const { return nodes_; }

private:
    nodes_t nodes_;
};

} // namespace tndm

#endif // MESHDATA_H
