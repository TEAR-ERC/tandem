#ifndef MESHCONFIG_20200930_H
#define MESHCONFIG_20200930_H

#include "form/BC.h"
#include "mesh/GenMesh.h"

#include <mpi.h>

#include <array>
#include <cstddef>
#include <optional>
#include <vector>

namespace tndm {

template <typename T> class TableSchema;

template <std::size_t D> struct BCConfig {
    BC bc;
    std::size_t plane;
    std::optional<std::array<std::size_t, D - 1u>> region;
};

template <std::size_t D> struct GenMeshConfig {
    std::array<std::vector<double>, D> intercepts;
    std::array<std::vector<BCConfig<D>>, D> bcs;

    GenMesh<D> create(double resolution, MPI_Comm comm) const;
    static void setSchema(TableSchema<GenMeshConfig<D>>& schema);
};

} // namespace tndm

#endif // MESHCONFIG_20200930_H
