#include "Profile.h"
#include "util/TablePrinter.h"

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <optional>

namespace tndm {

std::size_t Profile::add(std::string name) {
    regions_.emplace_back(std::move(name));
    watches_.emplace_back(Stopwatch());
    times_.emplace_back(0.0);
    return regions_.size() - 1;
}

std::size_t Profile::get(std::string_view name) const {
    auto first = regions_.cbegin();
    auto it = std::find(first, regions_.cend(), name);
    return std::distance(first, it);
}

void Profile::print(std::ostream& out, MPI_Comm comm) const {
    if (regions_.size() == 0) {
        return;
    }

    int rank;
    MPI_Comm_rank(comm, &rank);

    std::unique_ptr<TablePrinter> tp;
    if (rank == 0) {
        int w_1st_col =
            1 +
            std::max_element(regions_.begin(), regions_.end(), [](auto const& x, auto const& y) {
                return x.size() < y.size();
            })->size();
        tp = std::unique_ptr<TablePrinter>{new TablePrinter(
            out, {w_1st_col, 10}, {"Region", "t_min", "t_median", "t_mean", "t_max"})};
    }
    for (std::size_t region = 0, num = size(); region < num; ++region) {
        auto s = summary(region, comm);
        if (rank == 0) {
            *tp << regions_[region] << s.min << s.median << s.mean << s.max;
        }
    }
}

} // namespace tndm
