#ifndef CONFIG_20200825_H
#define CONFIG_20200825_H

#include "common/MGConfig.h"
#include "common/MeshConfig.h"
#include "config.h"
#include "io/Probe.h"
#include "tandem/AdaptiveOutputStrategy.h"
#include "tandem/FrictionConfig.h"
#include "tandem/SeasScenario.h"
#include "util/Schema.h"

#include <optional>
#include <string>
#include <vector>

namespace tndm {

struct OutputConfig {
    std::string prefix;
    double V_ref;
    double t_min;
    double t_max;
    AdaptiveOutputStrategy strategy;

    AdaptiveOutputInterval make_adaptive_output_interval() const {
        return AdaptiveOutputInterval(V_ref, t_min, t_max, strategy);
    }
};

struct ProbeOutputConfig : OutputConfig {
    std::vector<Probe<DomainDimension>> probes;
};

template <typename Derived> void setOutputConfigSchema(TableSchema<Derived>& outputSchema) {
    auto cast = [](auto ptr) {
        using type = std::remove_reference_t<decltype(std::declval<OutputConfig>().*ptr)>;
        return static_cast<type Derived::*>(ptr);
    };
    outputSchema.add_value("prefix", cast(&Derived::prefix)).help("Output file name prefix");
    outputSchema.add_value("V_ref", cast(&Derived::V_ref))
        .validator([](auto&& x) { return x > 0; })
        .default_value(0.1)
        .help("Output is written every t_min if this slip-rate is reached");
    outputSchema.add_value("t_min", cast(&Derived::t_min))
        .validator([](auto&& x) { return x > 0; })
        .default_value(0.1)
        .help("Minimum output interval");
    outputSchema.add_value("t_max", cast(&Derived::t_max))
        .validator([](auto&& x) { return x > 0; })
        .default_value(365 * 24 * 3600)
        .help("Maximum output interval");
    outputSchema.add_value("strategy", cast(&Derived::strategy))
        .default_value(AdaptiveOutputStrategy::Threshold)
        .help("Adaptive output strategy")
        .converter([](std::string_view value) {
            if (iEquals(value, "threshold")) {
                return AdaptiveOutputStrategy::Threshold;
            } else if (iEquals(value, "exponential")) {
                return AdaptiveOutputStrategy::Exponential;
            } else {
                return AdaptiveOutputStrategy::Unknown;
            }
        })
        .validator([](AdaptiveOutputStrategy const& type) {
            return type != AdaptiveOutputStrategy::Unknown;
        });
};

template <typename Derived> void setProbeOutputConfigSchema(TableSchema<Derived>& outputSchema) {
    auto cast = [](auto ptr) {
        using type = std::remove_reference_t<decltype(std::declval<ProbeOutputConfig>().*ptr)>;
        return static_cast<type Derived::*>(ptr);
    };
    auto& probeSchema = outputSchema.add_array("probes", cast(&Derived::probes)).of_tables();
    probeSchema.add_value("name", &Probe<DomainDimension>::name);
    probeSchema.add_array("x", &Probe<DomainDimension>::x).of_values();
};

struct Config {
    std::optional<double> resolution;
    double final_time;
    std::optional<std::string> mesh_file;
    bool discrete_green;
    bool matrix_free;
    MGStrategy mg_strategy;
    unsigned mg_coarse_level;
    SeasScenarioConfig seas;
    DieterichRuinaAgeingConfig friction;
    std::optional<GenMeshConfig<DomainDimension>> generate_mesh;
    std::optional<OutputConfig> domain_output;
    std::optional<OutputConfig> fault_output;
    std::optional<ProbeOutputConfig> fault_probe_output;
};

} // namespace tndm

#endif // CONFIG_20200825_H
