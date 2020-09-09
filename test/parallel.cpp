#include "doctest.h"
#include "parallel/SortedDistribution.h"

#include <cstddef>
#include <vector>

using tndm::SortedDistributionToRank;

TEST_CASE("parallel") {
    SUBCASE("SortedDistributionToRank") {
        std::vector<std::size_t> distribution({0, 5, 10, 15, 15, 21});
        SortedDistributionToRank p2r(distribution);
        CHECK(p2r(6) == 1);
        CHECK(p2r(20) == 4);
        CHECK(p2r(15) == 4);
        CHECK(p2r(14) == 2);
        CHECK(p2r(0) == 0);
    }
}
