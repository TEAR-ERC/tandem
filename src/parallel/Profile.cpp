#include "Profile.h"
#include "util/NullStream.h"
#include "util/TablePrinter.h"

#include <algorithm>

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

    nullostream null;
    std::ostream* my_out = &null;
    if (rank == 0) {
        my_out = &out;
    }

    int w_1st_col =
        1 + std::max_element(regions_.begin(), regions_.end(), [](auto const& x, auto const& y) {
                return x.size() < y.size();
            })->size();
    auto tp =
        TablePrinter(my_out, {w_1st_col, 10}, {"Region", "t_min", "t_median", "t_mean", "t_max"});

    auto print_summary = [&tp, &comm](std::string const& name, double time) {
        auto s = Summary(time, comm);
        tp << name << s.min << s.median << s.mean << s.max;
    };

    double total_time = 0.0;
    for (std::size_t region = 0, num = size(); region < num; ++region) {
        print_summary(regions_[region], times_[region]);
        total_time += times_[region];
    }
    tp.separator();
    print_summary("Total", total_time);
}

} // namespace tndm
