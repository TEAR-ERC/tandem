#include "doctest.h"
#include "util/Math.h"

using tndm::binom;
using tndm::Choose;

TEST_CASE("Math") {
    SUBCASE("binom") {
        CHECK(binom(4,3) == 4);
        CHECK(binom(4,1) == 4);
        CHECK(binom(4,0) == 1);
        CHECK(binom(4,4) == 1);
        CHECK(binom(4,2) == 6);
    }

    SUBCASE("Choose 0") {
        Choose<0> choose(3);
        CHECK(choose.next() == false);
    }

    SUBCASE("Choose 1") {
        Choose<1> choose(4);
        unsigned i = 0;
        do {
            CHECK(choose.current()[0] == i);
            ++i;
        } while (choose.next());
    }

    SUBCASE("Choose 2") {
        Choose<2> choose(3);
        CHECK(choose.current()[0] == 0);
        CHECK(choose.current()[1] == 1);
        CHECK(choose.next());
        CHECK(choose.current()[0] == 0);
        CHECK(choose.current()[1] == 2);
        CHECK(choose.next());
        CHECK(choose.current()[0] == 1);
        CHECK(choose.current()[1] == 2);
        CHECK(!choose.next());
    }

    SUBCASE("Choose all") {
        Choose<4> choose(4);
        do {
            CHECK(choose.current()[0] == 0);
            CHECK(choose.current()[1] == 1);
            CHECK(choose.current()[2] == 2);
            CHECK(choose.current()[3] == 3);
        } while (choose.next());
    }
}
