#ifndef ENDIANNESS_20200630_H
#define ENDIANNESS_20200630_H

#include <cstdint>

namespace tndm {

static bool isBigEndian() {
    union {
        uint32_t i;
        char c[4];
    } test = {0x11223344};
    return test.c[0] == 0x11;
}

} // namespace tndm

#endif // ENDIANNESS_20200630_H
