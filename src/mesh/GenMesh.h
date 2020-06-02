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

template <std::size_t D> struct TessInfo;
template <> struct TessInfo<1> {
    static constexpr int NumSimplices = 1;
    static constexpr int NumInVertGIDs = 2;
};
template <> struct TessInfo<2> {
    static constexpr int NumSimplices = 2;
    static constexpr int NumInVertGIDs = 4;
};
template <> struct TessInfo<3> {
    static constexpr int NumSimplices = 5;
    static constexpr int NumInVertGIDs = 8;
};

/**
 * @brief Mesh generation class for cuboid meshes
 *
 * @tparam D simplex/space dimension
 */
template <std::size_t D> class GenMesh {
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
     * @param transform Function that maps from the unit square or cube to another domain.
     *
     * @return Mesh of D-simplices and boundary mesh of (D-1)-simplices
     */
    std::unique_ptr<mesh_t> uniformMesh();

    std::unique_ptr<boundary_mesh_t> extractBoundaryMesh(mesh_t const& mesh);

private:
    std::array<uint64_t, D> N;
    std::array<uint64_t, D> Np1;
};

} // namespace tndm

#endif // GENMESH_H
