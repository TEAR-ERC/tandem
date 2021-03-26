#ifndef CONFIG_20200825_H
#define CONFIG_20200825_H

#include "common/MGConfig.h"
#include "common/MeshConfig.h"
#include "tandem/AdaptiveOutputStrategy.h"
#include "tandem/FrictionConfig.h"
#include "tandem/SeasScenario.h"

#include <optional>
#include <string>

namespace tndm {

struct OutputConfig {
    std::string prefix;
    double V_ref;
    double t_min;
    double t_max;
    AdaptiveOutputStrategy strategy;
};

struct Config {
    std::optional<double> resolution;
    double final_time;
    std::optional<std::string> mesh_file;
    bool matrix_free;
    MGStrategy mg_strategy;
    unsigned mg_coarse_level;
    SeasScenarioConfig seas;
    DieterichRuinaAgeingConfig friction;
    std::optional<GenMeshConfig<DomainDimension>> generate_mesh;
    std::optional<OutputConfig> output;
};

} // namespace tndm

#endif // CONFIG_20200825_H
