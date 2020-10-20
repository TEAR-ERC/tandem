#ifndef GAUSSLEGENDRE_20201012_H
#define GAUSSLEGENDRE_20201012_H

#include "Nodal.h"

#include <array>
#include <vector>

namespace tndm {

class GaussLegendreFactory : public NodesFactory<1u> {
public:
    virtual std::vector<std::array<double, 1u>> operator()(unsigned degree) const;
};

} // namespace tndm

#endif // GAUSSLEGENDRE_20201012_H
