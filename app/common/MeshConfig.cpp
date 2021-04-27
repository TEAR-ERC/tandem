#include "MeshConfig.h"
#include "util/Schema.h"

#include <stdexcept>
#include <string>
#include <string_view>

namespace tndm {

template <std::size_t D>
GenMesh<D> GenMeshConfig<D>::create(double resolution, MPI_Comm comm) const {
    std::array<double, D> h;
    h.fill(resolution);
    std::array<typename GenMesh<D>::bc_fun_t, D> BCs;
    for (std::size_t d = 0; d < D; ++d) {
        auto& bcs_d = bcs[d];
        BCs[d] = [bcs_d](std::size_t plane, std::array<std::size_t, D - 1u> const& region) -> BC {
            for (auto&& bc : bcs_d) {
                if (bc.plane == plane && (!bc.region || *bc.region == region)) {
                    return bc.bc;
                }
            }
            return BC::None;
        };
    }

    return GenMesh(intercepts, h, BCs, comm);
}

template <std::size_t D> void GenMeshConfig<D>::setSchema(TableSchema<GenMeshConfig<D>>& schema) {
    schema.add_array("intercepts", &GenMeshConfig<D>::intercepts).of_arrays().min(2).of_values();
    auto& bcConfigSchema = schema.add_array("bcs", &GenMeshConfig<D>::bcs).of_arrays().of_tables();
    bcConfigSchema.add_value("bc", &BCConfig<D>::bc).converter([](std::string_view bc) {
        if (!bc.empty()) {
            switch (bc[0]) {
            case 'd':
            case 'D':
                return BC::Dirichlet;
            case 'n':
            case 'N':
                return BC::Natural;
            case 'f':
            case 'F':
                return BC::Fault;
            default:
                break;
            }
        }
        throw std::invalid_argument("Unknown boundary condition type " + std::string(bc));
        return BC::None;
    });
    bcConfigSchema.add_value("plane", &BCConfig<D>::plane);
    bcConfigSchema.add_array("region", &BCConfig<D>::region).of_values();
}

template class GenMeshConfig<2u>;
template class GenMeshConfig<3u>;

} // namespace tndm
