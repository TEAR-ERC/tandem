#include "SeasConfig.h"

namespace tndm {

namespace detail {

template <typename T, typename P> auto up_cast(P ptr) {
    using type = std::remove_reference_t<decltype(std::declval<T>().*ptr)>;
    return static_cast<type T::*>(ptr);
};

template <typename Derived> void setOutputConfigSchema(TableSchema<Derived>& outputSchema) {
    outputSchema.add_value("prefix", up_cast<Derived>(&Derived::prefix))
        .validator(ParentPathExists())
        .help("Output file name prefix.");
    outputSchema.add_value("atol", up_cast<Derived>(&Derived::atol))
        .validator([](auto&& x) { return x >= 0; })
        .default_value(1e-50)
        .help("Absolute tolerance for VMax.");
    outputSchema.add_value("rtol", up_cast<Derived>(&Derived::rtol))
        .validator([](auto&& x) { return x >= 0; })
        .default_value(0.01)
        .help("Relative tolerance for VMax.");
    outputSchema.add_value("t_min", up_cast<Derived>(&Derived::t_min))
        .validator([](auto&& x) { return x >= 0; })
        .default_value(0.0)
        .help("Minimum time difference between samples.");
    outputSchema.add_value("t_max", up_cast<Derived>(&Derived::t_max))
        .validator([](auto&& x) { return x > 0; })
        .default_value(365 * 24 * 3600)
        .help("Maximum time difference between samples.");
}

template <typename Derived> void setDomainOutputConfigSchema(TableSchema<Derived>& outputSchema) {
    setOutputConfigSchema(outputSchema);

    outputSchema.add_value("jacobian", up_cast<Derived>(&Derived::jacobian))
        .default_value(false)
        .help("Output Jacobian.");
};

template <typename Derived> void setTabularOutputConfigSchema(TableSchema<Derived>& outputSchema) {
    setOutputConfigSchema(outputSchema);

    outputSchema.add_value("type", up_cast<Derived>(&Derived::type))
        .converter([](std::string_view value) {
            if (iEquals(value, "Tecplot")) {
                return TableWriterType::Tecplot;
            } else if (iEquals(value, "CSV")) {
                return TableWriterType::CSV;
            } else {
                return TableWriterType::Unknown;
            }
        })
        .validator([](TableWriterType const& mode) { return mode != TableWriterType::Unknown; })
        .default_value(TableWriterType::CSV);
}

template <typename Derived> void setProbeOutputConfigSchema(TableSchema<Derived>& outputSchema) {
    setTabularOutputConfigSchema(outputSchema);

    auto& probeSchema =
        outputSchema.add_array("probes", up_cast<Derived>(&Derived::probes)).of_tables();
    probeSchema.add_value("name", &Probe<DomainDimension>::name);
    probeSchema.add_array("x", &Probe<DomainDimension>::x).of_values();
};

} // namespace detail

void setConfigSchema(TableSchema<Config>& schema,
                     MakePathRelativeToOtherPath const& path_converter) {
    schema.add_value("resolution", &Config::resolution)
        .validator([](auto&& x) { return x > 0; })
        .help("Non-negative resolution parameter.");
    schema.add_value("final_time", &Config::final_time)
        .validator([](auto&& x) { return x >= 0; })
        .help("Non-negative final time of simulation.");
    schema.add_value("mesh_file", &Config::mesh_file)
        .converter(path_converter)
        .validator(PathExists())
        .help("Mesh file.");
    schema.add_value("cfl", &Config::cfl)
        .validator([](auto&& x) { return x > 0.0; })
        .default_value(1.0)
        .help("CFL tuning parameter (typically <= 1.0).");

    schema.add_value("mode", &Config::mode)
        .converter([](std::string_view value) {
            if (iEquals(value, "QuasiDynamic") || iEquals(value, "QD")) {
                return SeasMode::QuasiDynamic;
            } else if (iEquals(value, "QuasiDynamicDiscreteGreen") || iEquals(value, "QDGreen")) {
                return SeasMode::QuasiDynamicDiscreteGreen;
            } else if (iEquals(value, "FullyDynamic") || iEquals(value, "FD")) {
                return SeasMode::FullyDynamic;
            } else {
                return SeasMode::Unknown;
            }
        })
        .validator([](SeasMode const& mode) { return mode != SeasMode::Unknown; })
        .help("Mode of SEAS simulation "
              "(QuasiDynamic/QD|QuasiDynamicDiscreteGreen/QDGreen|FullyDynamic/FD).");
    schema.add_value("type", &Config::type)
        .converter([](std::string_view value) {
            if (iEquals(value, "poisson")) {
                return LocalOpType::Poisson;
            } else if (iEquals(value, "elastic") || iEquals(value, "elasticity")) {
                return LocalOpType::Elasticity;
            } else {
                return LocalOpType::Unknown;
            }
        })
        .validator([](LocalOpType const& type) { return type != LocalOpType::Unknown; })
        .help("Type of problem (poisson|elastic/elasticity).");
    schema.add_value("lib", &Config::lib)
        .converter(path_converter)
        .validator(PathExists())
        .help("Lua file containing material & frictional paramters.");
    schema.add_value("scenario", &Config::scenario)
        .help("Name of the specific scenario defined in the Lua library.");
    auto default_up = std::array<double, DomainDimension>{};
    default_up.back() = 1.0;
    schema.add_array("up", &Config::up)
        .default_value(std::move(default_up))
        .of_values()
        .help("Define up direction vector.");
    schema.add_array("ref_normal", &Config::ref_normal)
        .of_values()
        .help("Define reference normal vector.");
    schema.add_value("boundary_linear", &Config::boundary_linear)
        .default_value(false)
        .help("Assert that boundary is a linear function of time (i.e. boundary(x, t) = f(x) t).");

    schema.add_value("gf_checkpoint_prefix", &Config::gf_checkpoint_prefix)
        .help("Path where Green's function operator and RHS will be checkpointed.");
    schema.add_value("gf_checkpoint_every_nmins", &Config::gf_checkpoint_every_nmins)
        .default_value(30.0)
        .help("time interval, in minutes, at which the Green's function operator data is saved to "
              "disk.");

    schema.add_value("matrix_free", &Config::matrix_free)
        .default_value(false)
        .help("Use matrix-free operators.");
    schema.add_value("mg_coarse_level", &Config::mg_coarse_level)
        .default_value(1)
        .help("Polynomial degree of coarsest MG level.");
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
        .help("MG level selection strategy (TwoLevel|Logarithmic|Full).");

    auto& genMeshSchema = schema.add_table("generate_mesh", &Config::generate_mesh);
    GenMeshConfig<DomainDimension>::setSchema(genMeshSchema);

    auto& faultOutputSchema = schema.add_table("fault_output", &Config::fault_output);
    detail::setOutputConfigSchema(faultOutputSchema);
    auto& faultScalarOutputSchema =
        schema.add_table("fault_scalar_output", &Config::fault_scalar_output);
    detail::setTabularOutputConfigSchema(faultScalarOutputSchema);
    auto& domainOutputSchema = schema.add_table("domain_output", &Config::domain_output);
    detail::setDomainOutputConfigSchema(domainOutputSchema);
    auto& faultProbeOutputSchema =
        schema.add_table("fault_probe_output", &Config::fault_probe_output);
    detail::setProbeOutputConfigSchema(faultProbeOutputSchema);
    auto& domainProbeOutputSchema =
        schema.add_table("domain_probe_output", &Config::domain_probe_output);
    detail::setProbeOutputConfigSchema(domainProbeOutputSchema);
}

} // namespace tndm
