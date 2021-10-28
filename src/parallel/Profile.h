#ifndef PROFILE_20210916_H
#define PROFILE_20210916_H

#include "parallel/Summary.h"
#include "util/Stopwatch.h"

#include <cstdint>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace tndm {

class Profile {
public:
    std::size_t add(std::string name);
    std::size_t get(std::string_view name) const;
    inline std::string_view get(std::size_t region) const { return regions_[region]; }
    inline std::size_t size() const { return regions_.size(); }

    inline void begin(std::size_t region) { watches_[region].start(); }
    inline void end(std::size_t region, uint64_t flops = 0) {
        double time = watches_[region].stop();
        times_[region] += time;
        flops_[region] += flops;
    }

    inline Summary summary(std::size_t region, MPI_Comm comm) const {
        return Summary(times_[region], comm);
    }

    void print(std::ostream& out, MPI_Comm comm) const;

private:
    std::vector<Stopwatch> watches_;
    std::vector<std::string> regions_;
    std::vector<double> times_;
    std::vector<uint64_t> flops_;
};

} // namespace tndm

#endif // PROFILE_20210916_H
