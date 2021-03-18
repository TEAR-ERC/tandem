#include "MGConfig.h"
#include "tandem/RateAndStateBase.h"

namespace tndm {

MGConfig::MGConfig(unsigned coarse_level, MGStrategy strategy)
    : coarse_level_(coarse_level), strategy_(strategy) {}

std::vector<unsigned> MGConfig::levels(unsigned max_degree) const {
    unsigned coarse_level = coarse_level_;
    if (max_degree <= coarse_level) {
        coarse_level = max_degree > 0 ? max_degree - 1 : 0;
    }

    std::vector<unsigned> level_degree;
    switch (strategy_) {
    case MGStrategy::TwoLevel:
        level_degree = std::vector<unsigned>{coarse_level, max_degree};
        break;
    case MGStrategy::Logarithmic:
        level_degree.reserve(max_degree);
        level_degree.push_back(max_degree);
        while (level_degree.back() > coarse_level) {
            level_degree.push_back(level_degree.back() / 2);
        }
        if (level_degree.back() != coarse_level) {
            level_degree.back() = coarse_level;
        }
        std::reverse(level_degree.begin(), level_degree.end());
        break;
    case MGStrategy::Full:
        level_degree.resize(max_degree - coarse_level + 1);
        std::iota(level_degree.begin(), level_degree.end(), coarse_level);
        break;
    default:
        break;
    }

    return level_degree;
}

} // namespace tndm
