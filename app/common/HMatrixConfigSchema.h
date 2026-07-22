#ifndef HMATRIXCONFIGSCHEMA_20260722_H
#define HMATRIXCONFIGSCHEMA_20260722_H

#include "common/HMatrixConfig.h"
#include "util/Schema.h"

#include <string>

namespace tndm {

// Populates the schema for the [gf_compression] table. Factored out of setConfigSchema so
// the round-trip / removed-key behaviour can be unit-tested without the whole Config schema.
// The table is marked strict(): any key not listed here (e.g. a removed inert dial) fails
// loudly instead of being silently ignored.
inline void setGfCompressionConfigSchema(TableSchema<HMatrixConfig>& s) {
    s.strict();
    s.add_value("gf_compression", &HMatrixConfig::gf_compression)
        .default_value(std::string("none"))
        .validator([](auto&& x) { return x == "none" || x == "strumpack"; })
        .help("Green's-function compression backend: \"none\" (dense MatMult, default) or "
              "\"strumpack\" (matrix-free STRUMPACK HODLR operator).");
    s.add_value("leaf_size", &HMatrixConfig::leaf_size)
        .default_value(32)
        .validator([](auto&& x) { return x > 0; })
        .help("Cluster tree leaf size (tree depth is a function of this alone).");
    s.add_value("rtol", &HMatrixConfig::rtol)
        .default_value(1e-4)
        .validator([](auto&& x) { return x > 0.0; })
        .help("Relative compression tolerance. Explicit value is an override of the "
              "size-scaled policy default (see E1).");
    s.add_value("less_adapt", &HMatrixConfig::less_adapt)
        .default_value(true)
        .help("ButterflyPACK less_adapt. True (default) adapts the block rank only at the "
              "coarsest levels and extrapolates deeper, leaving deep-level errors above rtol. "
              "Set false to make every level adapt to rtol, at the cost of more matvecs.");
    s.add_value("rank_guess", &HMatrixConfig::rank_guess)
        .default_value(128)
        .validator([](auto&& x) { return x > 0; })
        .help("ButterflyPACK rank0: initial rank guess per off-diagonal block.");
    s.add_value("rank_rate", &HMatrixConfig::rank_rate)
        .default_value(2.0)
        .validator([](auto&& x) { return x > 1.0; })
        .help("ButterflyPACK rankrate: rank growth factor between adaptive trials.");
    s.add_value("bf_sampling", &HMatrixConfig::bf_sampling)
        .default_value(1.2)
        .validator([](auto&& x) { return x > 0.0; })
        .help("ButterflyPACK sample_para: oversampling factor for randomized construction.");
    s.add_value("format", &HMatrixConfig::format)
        .default_value(std::string("hodlr"))
        .validator([](auto&& x) { return x == "hodlr" || x == "hodbf"; })
        .help("Compressed format: \"hodlr\" (low-rank off-diagonal blocks) or \"hodbf\" "
              "(butterfly off-diagonal blocks).");
    s.add_value("cluster_tree", &HMatrixConfig::cluster_tree)
        .default_value(std::string("kdtree"))
        .validator([](auto&& x) { return x == "petsc1d" || x == "kdtree"; })
        .help("Cluster tree for the STRUMPACK matrix-free operator: \"kdtree\" (2D/3D "
              "median-split spatial tree, rank-invariant, correct in 1D and 2D; default) or "
              "\"petsc1d\" (binary bisection matching PETSc's distribution; only for "
              "reproducing old 1D-fault runs).");
    s.add_value("planar_fault", &HMatrixConfig::planar_fault)
        .default_value(false)
        .help("Skip compressing normal-traction components that are exactly zero for planar "
              "faults in homogeneous media (tangential slip -> zero normal traction). "
              "Validates that the assembled GF confirms near-zero normal coupling before "
              "skipping. Leave false (default) unless your fault geometry satisfies this.");
}

} // namespace tndm

#endif // HMATRIXCONFIGSCHEMA_20260722_H
