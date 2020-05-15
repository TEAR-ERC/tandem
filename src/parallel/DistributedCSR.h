#ifndef DISTRIBUTEDCSR_H
#define DISTRIBUTEDCSR_H

#include <vector>

template<typename T>
struct DistributedCSR {
    std::vector<T> dist;
    std::vector<T> colInd;
    std::vector<T> rowPtr;
};


#endif // DISTRIBUTEDCSR_H
