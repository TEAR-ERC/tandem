#ifndef DISTRIBUTEDCSR_H
#define DISTRIBUTEDCSR_H

#include <vector>

namespace tndm {

template <typename T> struct DistributedCSR {
    std::vector<T> dist;
    std::vector<T> colInd;
    std::vector<T> rowPtr;
};

} // namespace tndm

#endif // DISTRIBUTEDCSR_H
