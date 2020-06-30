#ifndef EQUIDISTANT_20200630_H
#define EQUIDISTANT_20200630_H

#include "Nodal.h"

namespace tndm {

/**
 * @brief Returns equidistant nodes for visualisation purposes.
 */
template <std::size_t D> class EquidistantNodesFactory : public NodesFactory<D> {
public:
    std::vector<std::array<double, D>> operator()(unsigned degree) const override;
};

} // namespace tndm

#endif // EQUIDISTANT_20200630_H
