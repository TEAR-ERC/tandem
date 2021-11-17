#include "Math.h"

namespace tndm {

uint64_t rangeProduct(uint64_t from, uint64_t to) {
    uint64_t product = 1;
    for (; from <= to; ++from) {
        product *= from;
    }
    return product;
}

} // namespace tndm
