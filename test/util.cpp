#include "doctest.h"
#include "tensor/Managed.h"
#include "tensor/Tensor.h"
#include "util/Algorithm.h"
#include "util/Combinatorics.h"
#include "util/LinearAllocator.h"
#include "util/Zero.h"

#include <array>
#include <cmath>
#include <new>

using namespace tndm;

TEST_CASE("Combinatorics") {
    SUBCASE("binom") {
        CHECK(binom(4, 3) == 4);
        CHECK(binom(4, 1) == 4);
        CHECK(binom(4, 0) == 1);
        CHECK(binom(4, 4) == 1);
        CHECK(binom(4, 2) == 6);
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

    SUBCASE("all integer sums (equal): 1 operand") {
        int N = 6;
        AllIntegerSums<1, false> sums(N);
        auto it = sums.begin();
        CHECK((*it)[0] == N);
        ++it;
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

    SUBCASE("all integer sums (equal): 2 operands") {
        int N = 7;
        AllIntegerSums<2, false> sums(N);
        auto it = sums.begin();
        for (unsigned j = 0; j <= N; ++j) {
            CHECK((*it)[0] == N - j);
            CHECK((*it)[1] == j);
            ++it;
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

    SUBCASE("all integer sums (equal): 3 operands") {
        int N = 5;
        AllIntegerSums<3, false> sums(N);
        auto it = sums.begin();
        for (unsigned j = 0; j <= N; ++j) {
            for (unsigned k = 0; k <= j; ++k) {
                CHECK((*it)[0] == N - j);
                CHECK((*it)[1] == j - k);
                CHECK((*it)[2] == k);
                ++it;
            }
        }
        CHECK(it == sums.end());
    }
}

TEST_CASE("Root finding") {
    auto F1 = [](double x) { return x * (x * x - 2.0) - 5.0; };
    CHECK(zeroIn(2.0, 3.0, F1) == doctest::Approx(2.0945514815));

    auto F2 = [](double x) { return 1.0 / (x - 3.0) - 6.0; };
    CHECK(zeroIn(3.0 + 1.0 / 100.0, 4.0, F2) == doctest::Approx(3.0 + 1.0 / 6.0));

    auto F3 = [](double x) { return x; };
    CHECK(zeroIn(-1.0, 0.0, F3) == 0.0);

    auto F4 = [](double x) { return 1.0 - std::asinh(x) - x; };
    CHECK(zeroIn(-1.0, 1.0, F4) == doctest::Approx(0.50992693151945222578));
}

TEST_CASE("Allocator") {
    alignas(32) double test[16];
    auto allocator = LinearAllocator<double>(test, test + 16, 32);

    double* mem = allocator.allocate(4);
    CHECK(mem == &test[0]);

    mem = allocator.allocate(2);
    CHECK(mem == &test[4]);

    mem = allocator.allocate(1);
    CHECK(mem == reinterpret_cast<void*>(&test[8]));

    bool except = false;
    try {
        allocator.allocate(1000);
    } catch (std::bad_alloc const&) {
        except = true;
    }
    REQUIRE(except);

    allocator.reset();
    mem = allocator.allocate(16);
    CHECK(mem == &test[0]);

    try {
        allocator.allocate(1);
    } catch (std::bad_alloc const&) {
        except = true;
    }
    REQUIRE(except);
}

TEST_CASE("Algorithm") {
    SUBCASE("Find blocks") {
        auto indices = std::array<std::size_t, 10>{5, 6, 7, 3, 1, 45, 46, 47, 49, 50};
        auto [block_lengths, displacements] = find_blocks(indices);

        CHECK(block_lengths.size() == 5);
        CHECK(displacements.size() == block_lengths.size());

        auto ref_displs = std::vector<std::size_t>{5, 3, 1, 45, 49};
        CHECK(std::equal(ref_displs.begin(), ref_displs.end(), displacements.begin()));
    }

    SUBCASE("Apply permutation") {
        auto tensor = Managed<Tensor<double, 3u>>(2, 3, 2);
        std::iota(tensor.data(), tensor.data() + tensor.size(), 0);

        CHECK(tensor(0, 0, 0) == 0);
        CHECK(tensor(1, 0, 0) == 1);
        CHECK(tensor(0, 1, 0) == 2);
        CHECK(tensor(1, 1, 0) == 3);
        CHECK(tensor(0, 2, 0) == 4);
        CHECK(tensor(1, 2, 0) == 5);
        CHECK(tensor(0, 0, 1) == 6);
        CHECK(tensor(1, 0, 1) == 7);
        CHECK(tensor(0, 1, 1) == 8);
        CHECK(tensor(1, 1, 1) == 9);
        CHECK(tensor(0, 2, 1) == 10);
        CHECK(tensor(1, 2, 1) == 11);

        apply_permutation(tensor, std::vector<std::ptrdiff_t>{2, 0, 1}, 1);

        CHECK(tensor(0, 0, 0) == 4);
        CHECK(tensor(1, 0, 0) == 5);
        CHECK(tensor(0, 1, 0) == 0);
        CHECK(tensor(1, 1, 0) == 1);
        CHECK(tensor(0, 2, 0) == 2);
        CHECK(tensor(1, 2, 0) == 3);
        CHECK(tensor(0, 0, 1) == 10);
        CHECK(tensor(1, 0, 1) == 11);
        CHECK(tensor(0, 1, 1) == 6);
        CHECK(tensor(1, 1, 1) == 7);
        CHECK(tensor(0, 2, 1) == 8);
        CHECK(tensor(1, 2, 1) == 9);

        apply_permutation(tensor, std::vector<std::ptrdiff_t>{1, 0}, 2);

        CHECK(tensor(0, 0, 0) == 10);
        CHECK(tensor(1, 0, 0) == 11);
        CHECK(tensor(0, 1, 0) == 6);
        CHECK(tensor(1, 1, 0) == 7);
        CHECK(tensor(0, 2, 0) == 8);
        CHECK(tensor(1, 2, 0) == 9);
        CHECK(tensor(0, 0, 1) == 4);
        CHECK(tensor(1, 0, 1) == 5);
        CHECK(tensor(0, 1, 1) == 0);
        CHECK(tensor(1, 1, 1) == 1);
        CHECK(tensor(0, 2, 1) == 2);
        CHECK(tensor(1, 2, 1) == 3);

        apply_permutation(tensor, std::vector<std::ptrdiff_t>{1, 0}, 0);

        CHECK(tensor(0, 0, 0) == 11);
        CHECK(tensor(1, 0, 0) == 10);
        CHECK(tensor(0, 1, 0) == 7);
        CHECK(tensor(1, 1, 0) == 6);
        CHECK(tensor(0, 2, 0) == 9);
        CHECK(tensor(1, 2, 0) == 8);
        CHECK(tensor(0, 0, 1) == 5);
        CHECK(tensor(1, 0, 1) == 4);
        CHECK(tensor(0, 1, 1) == 1);
        CHECK(tensor(1, 1, 1) == 0);
        CHECK(tensor(0, 2, 1) == 3);
        CHECK(tensor(1, 2, 1) == 2);
    }
}
