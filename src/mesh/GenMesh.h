#ifndef GENMESH_H
#define GENMESH_H

#include <cstddef>
#include <array>

#include <mpi.h>

#include "parallel/Distribute.h"
#include "util/MultiIndex.h"
#include "SimplexMesh.h"

namespace tndm {

template<std::size_t D> struct TessInfo;
template<> struct TessInfo<1> {
    static constexpr int NumSimplices = 1;
    static constexpr int NumCorners   = 2;
};
template<> struct TessInfo<2> {
    static constexpr int NumSimplices = 2;
    static constexpr int NumCorners   = 4;
};
template<> struct TessInfo<3> {
    static constexpr int NumSimplices = 5;
    static constexpr int NumCorners   = 8;
};

/**
* @brief Tessellate a line (D=1) / rectangle (D=2) / cuboid (D=3) into D-simplices.
*
* @param corners global vertex ids
* @param isOdd
*
* @return tessellation
*/
template<std::size_t D>
std::array<Simplex<D>,TessInfo<D>::NumSimplices> tessellate(std::array<int,TessInfo<D>::NumCorners> const& corners, bool isOdd);


/**
 * @brief Uniform SimplexMesh generation in D dimensions.
 *
 * @tparam D dimension
 * @param N array of (cuboid) elements in dimension d
 *
 * @return Mesh of D-simplices
 */
template<std::size_t D>
SimplexMesh<D> generateUniformMesh(std::array<int,D> N) {
    using vertex_t = typename SimplexMesh<D>::vertex_t;

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // We start with a mesh of cuboids with numElemsGlobal elements
    unsigned numElemsGlobal = 1u;
    for (auto& n : N) {
        numElemsGlobal *= n;
    }

    // vertices live on grid with size (N_1+1) x ... x (N_d+1)
    std::array<int,D> Np1;
    for (std::size_t d = 0; d < D; ++d) {
        Np1[d] = N[d] + 1;
    }

    // A cuboid has multi-index (n_1,...,n_d)
    // For parallel mesh generation we flatten the index and equally distribute cuboids
    auto range = distribute(Range(numElemsGlobal), rank, size);

    // We have 2 vertices per dim, therefore 2^D vertices per (D-)cuboid
    constexpr int numCorners = (1 << D);
    std::array<int,D> C;
    std::fill(C.begin(), C.end(), 2);
    const auto first = unflatten(range.from, N);

    // Function returns local vertex id \in [0,numVertices)
    // (see numVertices below)
    auto vertex_lid = [&first,&Np1](std::array<int,D> const& v) {
        return flatten(v-first, Np1);
    };

    // Map global vertex id to position
    auto vertex_pos = [&N](std::array<int,D> const& v) {
        vertex_t vert;
        for (std::size_t d = 0; d < D; ++d) {
            vert[d] = static_cast<double>(v[d]) / N[d];
        }
        return vert;
    };

    // Preallocate local vertices and elements
    const auto maxE = unflatten(range.to-1, N);
    const auto maxCorner = maxE + unflatten(numCorners-1, C);
    const int numVertices = vertex_lid(maxCorner)+1;
    const int numElements = range.length() * TessInfo<D>::NumSimplices;
    std::vector<vertex_t> vertices(numVertices);
    std::vector<Simplex<D>> elements(numElements);
    auto elementIn = elements.begin();

    for (auto eflat : range) {
        auto e = unflatten(eflat, N);
        std::array<int,numCorners> corners;
        // Insert 2^D vertices, if not present
        for (int corner = 0; corner < numCorners; ++corner) {
            auto v = e + unflatten(corner, C);
            auto lid = vertex_lid(v);
            corners[corner] = lid;
            vertices[lid] = vertex_pos(v);
        }
        bool isOdd = false;
        for (auto&& ee : e) {
            isOdd ^= (ee%2 == 1);
        }

        // Tessellate cuboid into D-simplices
        auto simplices = tessellate<D>(corners, isOdd);
        for (auto&& plex : simplices) {
            *(elementIn++) = plex;
        }
    }

    return SimplexMesh<D>(std::move(vertices), std::move(elements));
}

}

#endif // GENMESH_H
