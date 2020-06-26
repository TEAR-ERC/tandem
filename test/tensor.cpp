#include "doctest.h"
#include "tensor/Tensor.h"
#include "tensor/Managed.h"

#include <numeric>

using namespace tndm;

TEST_CASE("Subtensor") {
    auto tensor = Managed<Tensor<double, 4u>>(4, 7, 3, 9);
    auto subtensor = tensor.subtensor(slice{}, 5, 2, slice{});
    auto subtensor2 = tensor.subtensor(slice{}, slice{}, 2, 8);

    REQUIRE(tensor.size() == 756);
    std::iota(tensor.data(), tensor.data() + tensor.size(), 0);

    SUBCASE("Packed") {
        CHECK(detail::traits<decltype(tensor)>::Packed);
        CHECK(!detail::traits<decltype(subtensor)>::Packed);
        CHECK(detail::traits<decltype(subtensor2)>::Packed);
    }

    SUBCASE("Shape ok") {
        REQUIRE(subtensor.shape(0) == 4);
        REQUIRE(subtensor.shape(1) == 9);
    }

    SUBCASE("Offset and stride ok") {
        REQUIRE(subtensor.data() - tensor.data() == 76);
        REQUIRE(subtensor.stride(0) == 1);
        REQUIRE(subtensor.stride(1) == 84);
    }

    SUBCASE("Values match") {
        for (std::ptrdiff_t i = 0; i < subtensor.shape(0); ++i) {
            for (std::ptrdiff_t j = 0; j < subtensor.shape(1); ++j) {
                CHECK(subtensor(i, j) == tensor(i, 5, 2, j));
            }
        }
    }
}
