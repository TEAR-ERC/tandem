#ifndef CMDLINE_20200817_H
#define CMDLINE_20200817_H

#include "util/Schema.h"

#include <argparse.hpp>
#include <fstream>
#include <toml.hpp>

#include <optional>
#include <string_view>

namespace tndm {

template <typename T>
std::optional<T>
readFromConfigurationFileAndCmdLine(TableSchema<T> const& schema, argparse::ArgumentParser& program,
                                    int argc, char* argv[], std::string_view configPar = "config") {
    schema.cmd_line_args([&program](std::string_view key, std::string_view help) {
        program.add_argument("--" + std::string(key)).help(std::string(help));
    });

    try {
        program.parse_args(argc, argv);
    } catch (std::runtime_error& err) {
        std::cout << err.what() << std::endl;
        std::cout << program;
        return std::nullopt;
    }

    toml::table rawCfg;
    try {
        rawCfg = toml::parse_file(program.get(configPar));
    } catch (toml::parse_error const& err) {
        std::cerr << "Parsing failed of " << program.get(configPar) << " failed:" << std::endl
                  << err << std::endl;
        return std::nullopt;
    }

    T cfg;
    try {
        cfg = schema.translate(rawCfg);
        schema.cmd_line_args(
            [&cfg, &program, &schema](std::string_view key, std::string_view help) {
                if (auto val = program.present("--" + std::string(key))) {
                    schema.set(cfg, key, *val);
                }
            });
    } catch (std::runtime_error const& e) {
        std::cerr << "Error in configuration file" << std::endl
                  << "---------------------------" << std::endl
                  << e.what() << std::endl
                  << std::endl
                  << "You provided" << std::endl
                  << "------------" << std::endl
                  << rawCfg << std::endl
                  << std::endl
                  << "Schema" << std::endl
                  << "------" << std::endl
                  << schema << std::endl;
        return std::nullopt;
    }
    return std::make_optional(cfg);
}

} // namespace tndm

#endif // CMDLINE_20200817_H
