#ifndef MGCONFIG_20210318_H
#define MGCONFIG_20210318_H

#include <vector>

namespace tndm {

enum class MGStrategy { TwoLevel, Logarithmic, Full, Unknown };

class MGConfig {
public:
    MGConfig(unsigned coarse_level = 1, MGStrategy strategy = MGStrategy::Logarithmic);

    std::vector<unsigned> levels(unsigned max_degree) const;

private:
    unsigned coarse_level_;
    MGStrategy strategy_;
};
} // namespace tndm

#endif // MGCONFIG_20210318_H
