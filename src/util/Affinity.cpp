#include "Affinity.h"

#include <cassert>
#include <sys/sysinfo.h>

namespace tndm {

bool Affinity::is_empty(cpu_set_t const& set) { return CPU_COUNT(&set) == 0; }

std::string Affinity::to_string(cpu_set_t const& set) {
    const int np = get_nprocs();
    assert(np > 0);
    auto mask = std::string(np + (np - 1) / 10, '-');
    std::size_t pos = 0;
    for (int cpu = 0; cpu < np; ++cpu, ++pos) {
        if (cpu % 10 == 0 && cpu != 0 && cpu != np - 1) {
            mask[pos++] = '|';
        }
        if (CPU_ISSET(cpu, &set)) {
            mask[pos] = '0' + cpu % 10;
        }
    }
    return mask;
}

Affinity::Affinity() { sched_getaffinity(0, sizeof(cpu_set_t), &process_mask_); }

cpu_set_t Affinity::worker_mask() const {
    cpu_set_t u;
    CPU_ZERO(&u);
#pragma omp parallel default(none) shared(u)
    {
        cpu_set_t m;
        CPU_ZERO(&m);
        sched_getaffinity(0, sizeof(cpu_set_t), &m);
#pragma omp critical
        CPU_OR(&u, &u, &m);
    }
    return u;
}

} // namespace tndm
