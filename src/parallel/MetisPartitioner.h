#ifndef METISPARTITIONER_H
#define METISPARTITIONER_H

#include <parmetis.h>

#include "DistributedCSR.h"

namespace tndm {

class MetisPartitioner {
public:
    static constexpr int METIS_RANDOM_SEED = 42;

    static std::vector<idx_t> partition(DistributedCSR<idx_t>& csr,
                                        idx_t ncommonnodes,
                                        real_t imbalanceTol = 1.05);
};

}

#endif // METISPARTITIONER_H
