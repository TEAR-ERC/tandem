#include "tensor/Tensor.h"
#include "doctest.h"
#include "tensor/Managed.h"
#include "tensor/TensorBase.h"

#include <cstddef>
#include <numeric>

using namespace tndm;

TEST_CASE("Subtensor") {
    auto tensor = Managed<Tensor<double, 4u>>(4, 7, 3, 9);
    auto subtensor = tensor.subtensor(slice{}, 5, 2, slice{});
    auto subtensor2 = tensor.subtensor(slice{}, slice{}, 2, 8);
    auto subtensor3 = subtensor2.subtensor(slice{}, 1);
    auto subtensor4 = subtensor.subtensor(1, slice{});
    auto subtensor5 = subtensor.subtensor(slice{}, 1);

    REQUIRE(tensor.size() == 756);
    std::iota(tensor.data(), tensor.data() + tensor.size(), 0);

    SUBCASE("Packed") {
        CHECK(detail::traits<decltype(tensor)>::Packed);
        CHECK(!detail::traits<decltype(subtensor)>::Packed);
        CHECK(detail::traits<decltype(subtensor2)>::Packed);
        CHECK(detail::traits<decltype(subtensor3)>::Packed);
        CHECK(!detail::traits<decltype(subtensor4)>::Packed);
        CHECK(!detail::traits<decltype(subtensor5)>::Packed);
    }

    SUBCASE("Shape ok") {
        CHECK(subtensor.shape(0) == 4);
        CHECK(subtensor.shape(1) == 9);
        CHECK(subtensor2.shape(0) == 4);
        CHECK(subtensor2.shape(1) == 7);
        CHECK(subtensor3.shape(0) == 4);
        CHECK(subtensor4.shape(0) == 9);
        CHECK(subtensor5.shape(0) == 4);
    }

    SUBCASE("Offset and stride ok") {
        CHECK(subtensor.data() - tensor.data() == 76);
        CHECK(subtensor.stride(0) == 1);
        CHECK(subtensor.stride(1) == 84);
        CHECK(subtensor2.stride(0) == 1);
        CHECK(subtensor2.stride(1) == 4);
        CHECK(subtensor3.stride(0) == 1);
        CHECK(subtensor4.stride(0) == 84);
        CHECK(subtensor5.stride(0) == 1);
    }

    SUBCASE("Values match") {
        for (std::ptrdiff_t i = 0; i < subtensor.shape(0); ++i) {
            for (std::ptrdiff_t j = 0; j < subtensor.shape(1); ++j) {
                CHECK(subtensor(i, j) == tensor(i, 5, 2, j));
            }
        }
        for (std::ptrdiff_t i = 0; i < subtensor2.shape(0); ++i) {
            for (std::ptrdiff_t j = 0; j < subtensor2.shape(1); ++j) {
                CHECK(subtensor2(i, j) == tensor(i, j, 2, 8));
            }
        }
        for (std::ptrdiff_t i = 0; i < subtensor3.shape(0); ++i) {
            CHECK(subtensor3(i) == subtensor2(i, 1));
        }
        for (std::ptrdiff_t i = 0; i < subtensor3.shape(0); ++i) {
            CHECK(subtensor4(i) == subtensor(1, i));
            CHECK(subtensor4(i) == tensor(1, 5, 2, i));
        }
    }

    SUBCASE("Full copy ok") {
        auto new_tensor = Managed<Tensor<double, 4u>>(tensor.shape());
        new_tensor.copy_values(tensor);
        for (std::ptrdiff_t i = 0; i < tensor.size(); ++i) {
            CHECK(new_tensor.data()[i] == tensor.data()[i]);
        }
    }

    SUBCASE("Slice copy ok") {
        auto new_tensor = Managed<Matrix<double>>(tensor.shape(1), tensor.shape(3));
        new_tensor.copy_values(tensor.subtensor(1, slice{}, 2, slice{}));
        for (std::ptrdiff_t i = 0; i < new_tensor.shape(0); ++i) {
            for (std::ptrdiff_t j = 0; j < new_tensor.shape(1); ++j) {
                CHECK(new_tensor(i, j) == tensor(1, i, 2, j));
            }
        }
    }

    SUBCASE("Vector copy ok") {
        auto new_tensor = Managed<Vector<double>>(tensor.shape(3));
        new_tensor.copy_values(tensor.subtensor(1, 0, 2, slice{}));
        for (std::ptrdiff_t i = 0; i < new_tensor.shape(0); ++i) {
            CHECK(new_tensor(i) == tensor(1, 0, 2, i));
        }
    }
}
