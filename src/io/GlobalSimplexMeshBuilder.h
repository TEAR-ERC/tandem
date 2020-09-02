#ifndef GLOBALSIMPLEXMESHBUILDER_20200901_H
#define GLOBALSIMPLEXMESHBUILDER_20200901_H

#include "io/GMSHParser.h"
#include "mesh/GlobalSimplexMesh.h"
#include "mesh/MeshData.h"
#include "mesh/Simplex.h"

#include <mpi.h>

#include <array>
#include <cassert>

namespace tndm {

template <std::size_t D> struct GMSHSimplexType {};
template <> struct GMSHSimplexType<1u> { static constexpr long type = 1; };
template <> struct GMSHSimplexType<2u> { static constexpr long type = 2; };
template <> struct GMSHSimplexType<3u> { static constexpr long type = 4; };

template <std::size_t D> class GlobalSimplexMeshBuilder : public GMSHMeshBuilder {
private:
    std::vector<std::array<double, D>> vertices;
    std::vector<Simplex<D>> elements;
    std::vector<Simplex<D - 1u>> facets;
    std::vector<BC> bcs;

    std::size_t ignoredElems = 0;
    std::size_t unknownBC = 0;

public:
    void setNumVertices(std::size_t numVertices) { vertices.resize(numVertices); }
    void setVertex(long id, std::array<double, 3> const& x) {
        for (std::size_t i = 0; i < D; ++i) {
            vertices[id][i] = x[i];
        }
    }
    void setNumElements(std::size_t numElements) {
        elements.reserve(numElements);
        facets.reserve(numElements);
    }
    void addElement(long type, long tag, long* node, std::size_t numNodes) {
        if (type == GMSHSimplexType<D>::type) {
            assert(numNodes == D + 1u);
            std::array<uint64_t, D + 1u> elem;
            std::copy(node, node + D + 1u, elem.begin());
            elements.emplace_back(Simplex<D>(elem));
        } else if (type == GMSHSimplexType<D - 1u>::type) {
            assert(numNodes == D);
            std::array<uint64_t, D> elem;
            std::copy(node, node + D, elem.begin());
            facets.emplace_back(Simplex<D - 1u>(elem));
            BC bc = BC::None;
            switch (tag) {
            case static_cast<long>(BC::None):
                bc = BC::None;
                break;
            case static_cast<long>(BC::Dirichlet):
                bc = BC::Dirichlet;
                break;
            case static_cast<long>(BC::Fault):
                bc = BC::Fault;
                break;
            case static_cast<long>(BC::Natural):
                bc = BC::Natural;
                break;
            default:
                ++unknownBC;
                break;
            }
            bcs.push_back(bc);
        } else {
            ++ignoredElems;
        }
    }

    auto getUnknownBC() const { return unknownBC; }
    std::unique_ptr<GlobalSimplexMesh<D>> create(MPI_Comm comm) {
        auto boundaryData = std::make_unique<BoundaryData>(std::move(bcs));
        auto boundaryMesh = std::make_unique<GlobalSimplexMesh<D - 1u>>(
            std::move(facets), nullptr, std::move(boundaryData), comm);
        auto vertexData = std::make_unique<VertexData<D>>(std::move(vertices));
        auto mesh = std::make_unique<GlobalSimplexMesh<D>>(std::move(elements),
                                                           std::move(vertexData), nullptr, comm);
        mesh->setBoundaryMesh(std::move(boundaryMesh));
        return mesh;
    }
};

} // namespace tndm

#endif // GLOBALSIMPLEXMESHBUILDER_20200901_H
