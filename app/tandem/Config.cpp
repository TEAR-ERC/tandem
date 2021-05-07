#include "Config.h"

namespace tndm {

void setConfigSchema(TableSchema<Config>& schema,
                     MakePathRelativeToOtherPath const& path_converter) {
    schema.add_value("resolution", &Config::resolution)
        .validator([](auto&& x) { return x > 0; })
        .help("Non-negative resolution parameter");
    schema.add_value("final_time", &Config::final_time)
        .validator([](auto&& x) { return x >= 0; })
        .help("Non-negative final time of simulation");
    schema.add_value("mesh_file", &Config::mesh_file)
        .converter(path_converter)
        .validator(PathExists());

    schema.add_value("type", &Config::type)
        .converter([](std::string_view value) {
            if (iEquals(value, "poisson")) {
                return SeasType::Poisson;
            } else if (iEquals(value, "elastic") || iEquals(value, "elasticity")) {
                return SeasType::Elasticity;
            } else {
                return SeasType::Unknown;
            }
        })
        .validator([](SeasType const& type) { return type != SeasType::Unknown; });
    schema.add_value("lib", &Config::lib).converter(path_converter).validator(PathExists());
    schema.add_value("scenario", &Config::scenario);
    auto default_up = std::array<double, DomainDimension>{};
    default_up.back() = 1.0;
    schema.add_array("up", &Config::up).default_value(std::move(default_up)).of_values();
    schema.add_array("ref_normal", &Config::ref_normal).of_values();
    schema.add_value("boundary_linear", &Config::boundary_linear)
        .default_value(false)
        .help("Assert that boundary is a linear function of time (i.e. boundary(x, t) = f(x) t).");

    schema.add_value("discrete_green", &Config::discrete_green)
        .default_value(false)
        .help("Pre-compute discrete Green's function");
    schema.add_value("matrix_free", &Config::matrix_free)
        .default_value(false)
        .help("Use matrix-free operators");
    schema.add_value("mg_coarse_level", &Config::mg_coarse_level)
        .default_value(1)
        .help("Polynomial degree of coarsest MG level");
    schema.add_value("mg_strategy", &Config::mg_strategy)
        .converter([](std::string_view value) {
            if (iEquals(value, "TwoLevel")) {
                return MGStrategy::TwoLevel;
            } else if (iEquals(value, "Logarithmic")) {
                return MGStrategy::Logarithmic;
            } else if (iEquals(value, "Full")) {
                return MGStrategy::Full;
            } else {
                return MGStrategy::Unknown;
            }
        })
        .default_value(MGStrategy::TwoLevel)
        .validator([](MGStrategy const& type) { return type != MGStrategy::Unknown; })
        .help("MG level selection strategy (TwoLevel|Logarithmic|Full)");

    auto& genMeshSchema = schema.add_table("generate_mesh", &Config::generate_mesh);
    GenMeshConfig<DomainDimension>::setSchema(genMeshSchema);

    auto& faultOutputSchema = schema.add_table("fault_output", &Config::fault_output);
    setOutputConfigSchema(faultOutputSchema);
    auto& domainOutputSchema = schema.add_table("domain_output", &Config::domain_output);
    setOutputConfigSchema(domainOutputSchema);
    auto& faultProbeOutputSchema =
        schema.add_table("fault_probe_output", &Config::fault_probe_output);
    setOutputConfigSchema(faultProbeOutputSchema);
    setProbeOutputConfigSchema(faultProbeOutputSchema);
}

} // namespace tndm
