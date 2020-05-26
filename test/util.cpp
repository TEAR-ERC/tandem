#include "doctest.h"
#include "util/Combinatorics.h"

using tndm::AllIntegerSums;
using tndm::binom;
using tndm::Choose;

TEST_CASE("Combinatorics") {
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

    SUBCASE("all integer sums: 1 operand") {
        int N = 6;
        AllIntegerSums<1> sums(N);
        auto it = sums.begin();
        for (int i = 0; i <= N; ++i) {
            CHECK((*it)[0] == i);
            ++it;
        }
        CHECK(it == sums.end());
    }

    SUBCASE("all integer sums: 2 operands") {
        int N = 7;
        AllIntegerSums<2> sums(N);
        auto it = sums.begin();
        for (int i = 0; i <= N; ++i) {
            for (unsigned j = 0; j <= i; ++j) {
                CHECK((*it)[0] == i - j);
                CHECK((*it)[1] == j);
                ++it;
            }
        }
        CHECK(it == sums.end());
    }

    SUBCASE("all integer sums: 3 operands") {
        int N = 5;
        AllIntegerSums<3> sums(N);
        auto it = sums.begin();
        for (unsigned i = 0; i <= N; ++i) {
            for (unsigned j = 0; j <= i; ++j) {
                for (unsigned k = 0; k <= j; ++k) {
                    CHECK((*it)[0] == i - j);
                    CHECK((*it)[1] == j - k);
                    CHECK((*it)[2] == k);
                    ++it;
                }
            }
        }
        CHECK(it == sums.end());
    }
}
