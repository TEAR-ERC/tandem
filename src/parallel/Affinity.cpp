#ifdef __linux__

#include "Affinity.h"
#include "MPITraits.h"

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

auto Affinity::cpu_set_to_my_set(cpu_set_t const& set) const -> my_set {
    my_set result{};
    for (int cpu = 0; cpu < bits_per_unit() * result.size(); ++cpu) {
        if (CPU_ISSET(cpu, &set)) {
            result[cpu / bits_per_unit()] |= (1 << (cpu % bits_per_unit()));
        }
    }
    return result;
}
auto Affinity::my_set_to_cpu_set(my_set const& set) const -> cpu_set_t {
    cpu_set_t result;
    CPU_ZERO(&result);
    for (int cpu = 0; cpu < bits_per_unit() * set.size(); ++cpu) {
        if (set[cpu / bits_per_unit()] & (1 << (cpu % bits_per_unit()))) {
            CPU_SET(cpu, &result);
        }
    }
    return result;
}

cpu_set_t Affinity::worker_mask_on_node(MPI_Comm comm) const {
    auto rank_mask = cpu_set_to_my_set(worker_mask());

    MPI_Comm comm_shared;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &comm_shared);
    MPI_Allreduce(MPI_IN_PLACE, rank_mask.data(), rank_mask.size(),
                  mpi_type<my_set::value_type>::type(), MPI_BOR, comm_shared);
    MPI_Comm_free(&comm_shared);

    return my_set_to_cpu_set(rank_mask);
}

} // namespace tndm

#endif // __linux__
