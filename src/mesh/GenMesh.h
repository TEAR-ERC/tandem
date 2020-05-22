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
    static std::array<simplex_t,TessInfo<D>::NumSimplices> tessellate(std::array<int,TessInfo<D>::NumInVertGIDs> const& vertGIDs, bool isOdd);


    /**
     * @brief Uniform SimplexMesh generation in D dimensions.
     *
     * @param N array of (cuboid) elements in dimension d
     *
     * @return Mesh of D-simplices
     */
    mesh_t uniformMesh(std::array<int, D> N) {
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        // vertices live on grid with size (N_1+1) x ... x (N_d+1)
        std::array<int,D> Np1;
        for (std::size_t d = 0; d < D; ++d) {
            Np1[d] = N[d] + 1;
        }
        unsigned numVertsGlobal = 1u;
        for (auto& np1 : Np1) {
            numVertsGlobal *= np1;
        }

        // A vertex has multi-index (v_1,...,v_d)
        // For parallel mesh generation we flatten the index and equally distribute vertices
        auto vertsLocal = distribute(Range(numVertsGlobal), rank, size);

        const int numVertices = vertsLocal.length();
        std::vector<vertex_t> vertices(numVertices);

        // Map global vertex id to position
        auto vertex_pos = [&N](std::array<int,D> const& v) {
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
        constexpr int numVertGIDs = (1 << D);
        std::array<int,D> D2;
        std::fill(D2.begin(), D2.end(), 2);

        const int numElements = elemsLocal.length() * TessInfo<D>::NumSimplices;
        std::vector<simplex_t> elements(numElements);

        for (auto eflat : elemsLocal) {
            auto e = unflatten(eflat, N);
            std::array<int,numVertGIDs> vertGIDs;
            // Get 2^D vertices
            for (int vflat = 0; vflat < numVertGIDs; ++vflat) {
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
        return mesh_t(std::move(elements), std::move(vertexData));
    }
};

}

#endif // GENMESH_H
