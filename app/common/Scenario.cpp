#include "Scenario.h"

namespace tndm {

template <std::size_t D> GenMesh<D> GenMeshConfig<D>::create(double resolution, MPI_Comm comm) {
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

template class GenMeshConfig<2u>;
template class GenMeshConfig<3u>;

} // namespace tndm
