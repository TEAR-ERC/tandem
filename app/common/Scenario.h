#ifndef SCENARIO_20200817_H
#define SCENARIO_20200817_H

#include "mesh/GenMesh.h"
#include "mesh/GlobalSimplexMesh.h"
#include "mesh/MeshData.h"

#include <array>
#include <cstddef>
#include <optional>
#include <vector>

namespace tndm {

template <std::size_t D> struct BCConfig {
    BC bc;
    std::size_t plane;
    std::optional<std::array<std::size_t, D - 1u>> region;
};

template <std::size_t D> struct GenMeshConfig {
    std::array<std::vector<double>, D> intercepts;
    std::array<std::vector<BCConfig<D>>, D> bcs;

    GenMesh<D> create(double resolution, MPI_Comm comm);
};

} // namespace tndm

#endif // SCENARIO_20200817_H
