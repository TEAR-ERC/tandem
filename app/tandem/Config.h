#ifndef CONFIG_20200825_H
#define CONFIG_20200825_H

#include "common/Scenario.h"
#include "config.h"
#include "tandem/Scenario.h"

#include <optional>
#include <string>

namespace tndm {

struct Config {
    double resolution;
    double final_time;
    std::optional<std::string> output;
    std::optional<double> output_interval;
    ProblemConfig problem;
    GenMeshConfig<DomainDimension> generate_mesh;
};

} // namespace tndm

#endif // CONFIG_20200825_H
