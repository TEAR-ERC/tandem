#ifndef MESHDATA_H
#define MESHDATA_H

#include "GlobalSimplexMesh.h"
#include "LocalFaces.h"
#include "parallel/CommPattern.h"

#include <mpi.h>

#include <array>
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
};

template <std::size_t SpaceD> class VertexData : public MeshData {
public:
    using vertex_t = std::array<double, SpaceD>;

    VertexData(std::vector<vertex_t>&& vertices) : vertices(std::move(vertices)) {}
    virtual ~VertexData() {}

    std::size_t size() const override { return vertices.size(); }

    std::unique_ptr<MeshData> redistributed(std::vector<std::size_t> const& lids,
                                            AllToAllV const& a2a) const override {
        std::vector<std::array<double, SpaceD>> requestedVertices;
        requestedVertices.reserve(lids.size());
        for (auto& lid : lids) {
            requestedVertices.emplace_back(vertices[lid]);
        }

        mpi_array_type<double> mpi_t(SpaceD);
        auto newVertices = a2a.exchange(requestedVertices, mpi_t.get());
        return std::make_unique<VertexData>(std::move(newVertices));
    }

    std::vector<vertex_t> const& getVertices() const { return vertices; }

private:
    std::vector<vertex_t> vertices;
};

} // namespace tndm

#endif // MESHDATA_H
