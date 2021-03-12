#ifndef AFFINITY_20210311_H
#define AFFINITY_20210311_H

#include <array>
#include <cstdint>
#include <mpi.h>
#include <sched.h>
#include <string>

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

#endif // AFFINITY_20210311_H
