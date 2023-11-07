#ifndef CONFIG_20200825_H
#define CONFIG_20200825_H

#include "common/MGConfig.h"
#include "common/MeshConfig.h"
#include "config.h"
#include "io/CSVWriter.h"
#include "io/Probe.h"
#include "io/TecplotWriter.h"
#include "tandem/AdaptiveOutputStrategy.h"
#include "tandem/SeasType.h"
#include "util/Schema.h"
#include "util/SchemaHelper.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace tndm {

enum class TableWriterType { Tecplot, CSV, Unknown };

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

struct DomainOutputConfig : OutputConfig {
    bool jacobian;
};

struct TabularOutputConfig : OutputConfig {
    TableWriterType type;

    std::unique_ptr<TableWriter> make_writer() const {
        switch (type) {
        case TableWriterType::Tecplot:
            return std::make_unique<TecplotWriter>();
        case TableWriterType::CSV:
            return std::make_unique<CSVWriter>();
        case TableWriterType::Unknown:
            return nullptr;
        }
        return nullptr;
    }
};

struct ProbeOutputConfig : TabularOutputConfig {
    std::vector<Probe<DomainDimension>> probes;
};

enum tsCheckpointStorageType { NONE, UNLIMITED, LIMITED, UNKNOWN };

struct Config {
    std::optional<double> resolution;
    double final_time;
    std::optional<std::string> mesh_file;
    double cfl;

    SeasMode mode;
    SeasType type;
    std::string lib;
    std::string scenario;
    std::array<double, DomainDimension> up;
    std::array<double, DomainDimension> ref_normal;
    bool boundary_linear;

    bool matrix_free;
    MGStrategy mg_strategy;
    unsigned mg_coarse_level;

    std::optional<GenMeshConfig<DomainDimension>> generate_mesh;
    std::optional<OutputConfig> fault_output;
    std::optional<TabularOutputConfig> fault_scalar_output;
    std::optional<DomainOutputConfig> domain_output;
    std::optional<ProbeOutputConfig> fault_probe_output;
    std::optional<ProbeOutputConfig> domain_probe_output;

    std::optional<std::string> ts_checkpoint_load_directory;
    std::string ts_checkpoint_save_directory;
    int ts_checkpoint_frequency_step;
    double ts_checkpoint_frequency_cputime_minutes;
    double ts_checkpoint_frequency_time_physical;
    tsCheckpointStorageType ts_checkpoint_storage_type;
    int ts_checkpoint_storage_limited_size;
};

void setConfigSchema(TableSchema<Config>& schema,
                     MakePathRelativeToOtherPath const& path_converter);

} // namespace tndm

#endif // CONFIG_20200825_H
