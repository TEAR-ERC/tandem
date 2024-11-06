#ifndef AFFINITY_20210311_H
#define AFFINITY_20210311_H

#include <cstdint>
#include <mpi.h>
#include <string>

#ifdef __linux__

#include <array>
#include <sched.h>

namespace tndm {

class Affinity {
public:
    static bool is_empty(cpu_set_t const& set);
    static std::string to_string(cpu_set_t const& set);

    cpu_set_t worker_mask() const;
    cpu_set_t worker_mask_on_node(MPI_Comm comm) const;

private:
    using my_set = std::array<std::uint64_t, CPU_SETSIZE / 64>;

    constexpr std::size_t bits_per_unit() const { return sizeof(my_set::value_type) * 8; }
    auto cpu_set_to_my_set(cpu_set_t const& set) const -> my_set;
    auto my_set_to_cpu_set(my_set const& set) const -> cpu_set_t;
};

} // namespace tndm

#else // not __linux__

namespace tndm {

class Affinity {
public:
    using cpu_set_t = int;

    static bool is_empty(cpu_set_t const& set) { return true; }
    static std::string to_string(cpu_set_t const& set) { return "(affinity not implemented)"; }

    cpu_set_t worker_mask() const { return 0; }
    cpu_set_t worker_mask_on_node(MPI_Comm) const { return 0; }
};

} // namespace tndm

#endif // __Linux __

#endif // AFFINITY_20210311_H
