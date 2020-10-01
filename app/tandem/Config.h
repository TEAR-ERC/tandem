#ifndef CONFIG_20200825_H
#define CONFIG_20200825_H

#include "common/MeshConfig.h"
#include "common/PoissonScenario.h"

#include <optional>
#include <string>

namespace tndm {

struct Config {
    std::optional<double> resolution;
    double final_time;
    std::optional<std::string> output;
    std::optional<double> output_interval;
    std::optional<std::string> mesh_file;
    PoissonScenarioConfig poisson;
    std::optional<GenMeshConfig<DomainDimension>> generate_mesh;
};

} // namespace tndm

#endif // CONFIG_20200825_H
