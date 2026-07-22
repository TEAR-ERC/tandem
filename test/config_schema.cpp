#include "doctest.h"

#include "common/HMatrixConfig.h"
#include "common/HMatrixConfigSchema.h"
#include "util/Schema.h"

#include <toml.hpp>

#include <stdexcept>
#include <string>

using namespace tndm;

namespace {
TableSchema<HMatrixConfig> makeSchema() {
    TableSchema<HMatrixConfig> schema;
    setGfCompressionConfigSchema(schema);
    return schema;
}
} // namespace

TEST_CASE("gf_compression schema round-trips every key") {
    auto schema = makeSchema();

    // A TOML table setting every key to a non-default value.
    toml::table t;
    t.insert_or_assign("gf_compression", "strumpack");
    t.insert_or_assign("leaf_size", 64);
    t.insert_or_assign("rtol", 1e-6);
    t.insert_or_assign("less_adapt", false);
    t.insert_or_assign("rank_guess", 200);
    t.insert_or_assign("rank_rate", 3.0);
    t.insert_or_assign("bf_sampling", 1.5);
    t.insert_or_assign("format", "hodbf");
    t.insert_or_assign("cluster_tree", "petsc1d");
    t.insert_or_assign("planar_fault", true);

    HMatrixConfig c = schema.translate(t);

    CHECK(c.gf_compression == "strumpack");
    CHECK(c.leaf_size == 64);
    CHECK(c.rtol == doctest::Approx(1e-6));
    CHECK(c.less_adapt == false);
    CHECK(c.rank_guess == 200);
    CHECK(c.rank_rate == doctest::Approx(3.0));
    CHECK(c.bf_sampling == doctest::Approx(1.5));
    CHECK(c.format == "hodbf");
    CHECK(c.cluster_tree == "petsc1d");
    CHECK(c.planar_fault == true);
}

TEST_CASE("gf_compression schema supplies the documented defaults") {
    auto schema = makeSchema();

    toml::table empty;
    HMatrixConfig c = schema.translate(empty);

    CHECK(c.gf_compression == "none");   // compression off by default
    CHECK(c.cluster_tree == "kdtree");   // rank-invariant tree is the production default
    CHECK(c.leaf_size == 32);
    CHECK(c.rtol == doctest::Approx(1e-4));
    CHECK(c.format == "hodlr");
}

TEST_CASE("a removed/inert key fails loudly instead of being ignored") {
    auto schema = makeSchema();

    // eta, max_rank, basis_order, batch_size and use_hmatrix were removed in A3.1; any of
    // them appearing in a config must be rejected, not silently dropped.
    for (auto const& removed : {"eta", "max_rank", "basis_order", "batch_size", "use_hmatrix"}) {
        toml::table t;
        t.insert_or_assign("gf_compression", "strumpack");
        t.insert_or_assign(removed, 1);
        CHECK_THROWS_AS(schema.translate(t), std::runtime_error);
    }
}

TEST_CASE("an unrecognised key fails loudly") {
    auto schema = makeSchema();
    toml::table t;
    t.insert_or_assign("not_a_key", 3);
    CHECK_THROWS_AS(schema.translate(t), std::runtime_error);
}

TEST_CASE("invalid enum values are rejected") {
    auto schema = makeSchema();

    SUBCASE("gf_compression") {
        toml::table t;
        t.insert_or_assign("gf_compression", "htool");
        CHECK_THROWS_AS(schema.translate(t), std::runtime_error);
    }
    SUBCASE("cluster_tree") {
        toml::table t;
        t.insert_or_assign("cluster_tree", "hilbert");
        CHECK_THROWS_AS(schema.translate(t), std::runtime_error);
    }
    SUBCASE("format") {
        toml::table t;
        t.insert_or_assign("format", "dense");
        CHECK_THROWS_AS(schema.translate(t), std::runtime_error);
    }
}
