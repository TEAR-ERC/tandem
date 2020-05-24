#ifndef GENMESH_H
#define GENMESH_H

#include "GlobalSimplexMesh.h"
#include "MeshData.h"
#include "parallel/Distribute.h"
#include "util/MultiIndex.h"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>

namespace tndm {

template<std::size_t D> struct TessInfo;
template<> struct TessInfo<1> {
    static constexpr int NumSimplices = 1;
    static constexpr int NumInVertGIDs = 2;
};
template<> struct TessInfo<2> {
    static constexpr int NumSimplices = 2;
    static constexpr int NumInVertGIDs = 4;
};
template<> struct TessInfo<3> {
    static constexpr int NumSimplices = 5;
    static constexpr int NumInVertGIDs = 8;
};


/**
 * @brief Mesh generation class for cuboid meshes
 *
 * @tparam D simplex/space dimension
 */
template<std::size_t D>
class GenMesh {
public:
    using mesh_t = GlobalSimplexMesh<D>;
    using simplex_t = typename mesh_t::simplex_t;
    using boundary_mesh_t = GlobalSimplexMesh<D - 1>;
    using boundary_simplex_t = typename boundary_mesh_t::simplex_t;
    using boundary_data_t = BoundaryData;
    using vertex_data_t = VertexData<D>;
    using vertex_t = typename vertex_data_t::vertex_t;

    /**
    * @brief Tessellate a line (D=1) / rectangle (D=2) / cuboid (D=3) into D-simplices.
    *
    * @param corners global vertex ids
    * @param isOdd
    *
    * @return tessellation
    */
    static std::array<simplex_t, TessInfo<D>::NumSimplices>
    tessellate(std::array<uint64_t, TessInfo<D>::NumInVertGIDs> const& vertGIDs, bool isOdd);

    /**
     * @brief
     *
     * @param N array of (cuboid) elements in dimension d
     */
    GenMesh(std::array<uint64_t, D> const& N) : N(N) {
        // vertices live on grid with size (N_1+1) x ... x (N_d+1)
        for (std::size_t d = 0; d < D; ++d) {
            Np1[d] = N[d] + 1;
        }
    }

    /**
     * @brief Uniform SimplexMesh generation in D dimensions.
     *
     * @return Mesh of D-simplices and boundary mesh of (D-1)-simplices
     */
    std::unique_ptr<mesh_t> uniformMesh() {
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        std::size_t numVertsGlobal = 1u;
        for (auto& np1 : Np1) {
            numVertsGlobal *= np1;
        }

        // A vertex has multi-index (v_1,...,v_d)
        // For parallel mesh generation we flatten the index and equally distribute vertices
        auto vertsLocal = distribute(Range(numVertsGlobal), rank, size);

        const std::size_t numVertices = vertsLocal.length();
        std::vector<vertex_t> vertices(numVertices);

        // Map global vertex id to position
        auto vertex_pos = [&](std::array<uint64_t, D> const& v) {
            vertex_t vert;
            for (std::size_t d = 0; d < D; ++d) {
                vert[d] = static_cast<double>(v[d]) / N[d];
            }
            return vert;
        };

        for (auto vflat : vertsLocal) {
            auto v = unflatten(vflat, Np1);
            vertices[vflat-vertsLocal.from] = vertex_pos(v);
        }

        // elements live on grid with size N_1 x ... x N_d
        unsigned numElemsGlobal = 1u;
        for (auto& n : N) {
            numElemsGlobal *= n;
        }

        // An element (cuboid for D=3) has multi-index (n_1,...,n_d)
        // For parallel mesh generation we flatten the index and equally distribute elements
        auto elemsLocal = distribute(Range(numElemsGlobal), rank, size);

        // We have 2 vertices per dim i.e. 2^D vertices per element
        constexpr uint64_t numVertGIDs = (1 << D);
        std::array<uint64_t, D> D2;
        std::fill(D2.begin(), D2.end(), 2);

        const std::size_t numElements = elemsLocal.length() * TessInfo<D>::NumSimplices;
        std::vector<simplex_t> elements(numElements);

        for (auto eflat : elemsLocal) {
            auto e = unflatten(eflat, N);
            std::array<uint64_t, numVertGIDs> vertGIDs;
            // Get 2^D vertices
            for (uint64_t vflat = 0; vflat < numVertGIDs; ++vflat) {
                auto v = e + unflatten(vflat, D2);
                vertGIDs[vflat] = flatten(v,Np1);
            }
            bool isOdd = false;
            for (auto&& ee : e) {
                isOdd ^= (ee%2 == 1);
            }

            // Tessellate cuboid into D-simplices
            auto simplices = tessellate(vertGIDs, isOdd);
            std::copy(simplices.begin(), simplices.end(),
                      elements.begin()+TessInfo<D>::NumSimplices*(eflat-elemsLocal.from));
        }

        auto vertexData = std::make_unique<vertex_data_t>(std::move(vertices));
        auto mesh = std::make_unique<mesh_t>(std::move(elements), std::move(vertexData));
        auto boundaryMesh = extractBoundaryMesh(*mesh);
        mesh->template setBoundaryMesh<D - 1>(std::move(boundaryMesh));
        return mesh;
    }

    std::unique_ptr<boundary_mesh_t> extractBoundaryMesh(mesh_t const& mesh) {
        std::vector<boundary_simplex_t> boundaryElements;
        std::vector<int> boundaryConditions;
        for (auto& elem : mesh.getElements()) {
            for (auto& face : elem.downward()) {
                for (std::size_t d = 0; d < D; ++d) {
                    bool faceAtBnd = true;
                    for (auto& vflat : face) {
                        auto v = unflatten(vflat, Np1);
                        faceAtBnd = faceAtBnd && ((v[d] == 0) || (v[d] == Np1[d] - 1));
                    }
                    if (faceAtBnd) {
                        boundaryElements.emplace_back(face);
                        boundaryConditions.emplace_back(d + 1);
                    }
                }
            }
        }
        auto boundaryData = std::make_unique<boundary_data_t>(std::move(boundaryConditions));
        return std::make_unique<boundary_mesh_t>(std::move(boundaryElements), nullptr,
                                                 std::move(boundaryData));
    }

private:
    std::array<uint64_t, D> N;
    std::array<uint64_t, D> Np1;
};

}

#endif // GENMESH_H
