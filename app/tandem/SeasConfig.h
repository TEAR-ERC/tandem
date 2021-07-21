#ifndef CONFIG_20200825_H
#define CONFIG_20200825_H

#include "common/MGConfig.h"
#include "common/MeshConfig.h"
#include "config.h"
#include "io/Probe.h"
#include "tandem/AdaptiveOutputStrategy.h"
#include "tandem/SeasType.h"
#include "util/Schema.h"
#include "util/SchemaHelper.h"

#include <optional>
#include <string>
#include <vector>

namespace tndm {

struct OutputConfig {
    std::string prefix;
    double atol;
    double rtol;
    double t_min;
    double t_max;

    AdaptiveOutputInterval make_adaptive_output_interval() const {
        return AdaptiveOutputInterval(atol, rtol, t_min, t_max);
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
    outputSchema.add_value("prefix", cast(&Derived::prefix))
        .validator(ParentPathExists())
        .help("Output file name prefix");
    outputSchema.add_value("atol", cast(&Derived::atol))
        .validator([](auto&& x) { return x >= 0; })
        .default_value(1e-50)
        .help("Absolute tolerance for VMax");
    outputSchema.add_value("rtol", cast(&Derived::rtol))
        .validator([](auto&& x) { return x >= 0; })
        .default_value(0.01)
        .help("Relative tolerance for VMax");
    outputSchema.add_value("t_min", cast(&Derived::t_min))
        .validator([](auto&& x) { return x >= 0; })
        .default_value(0.0)
        .help("Minimum time difference between samples");
    outputSchema.add_value("t_max", cast(&Derived::t_max))
        .validator([](auto&& x) { return x > 0; })
        .default_value(365 * 24 * 3600)
        .help("Maximum time difference between samples");
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

    SeasType type;
    std::string lib;
    std::string scenario;
    std::array<double, DomainDimension> up;
    std::array<double, DomainDimension> ref_normal;
    bool boundary_linear;

    bool discrete_green;
    bool matrix_free;
    MGStrategy mg_strategy;
    unsigned mg_coarse_level;

    std::optional<GenMeshConfig<DomainDimension>> generate_mesh;
    std::optional<OutputConfig> fault_output;
    std::optional<OutputConfig> fault_scalar_output;
    std::optional<OutputConfig> domain_output;
    std::optional<ProbeOutputConfig> fault_probe_output;
    std::optional<ProbeOutputConfig> domain_probe_output;
};

void setConfigSchema(TableSchema<Config>& schema,
                     MakePathRelativeToOtherPath const& path_converter);

} // namespace tndm

#endif // CONFIG_20200825_H
