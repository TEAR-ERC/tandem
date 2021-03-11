#ifndef AFFINITY_20210311_H
#define AFFINITY_20210311_H

#include <sched.h>
#include <string>

namespace tndm {

class Affinity {
public:
    static bool is_empty(cpu_set_t const& set);
    static std::string to_string(cpu_set_t const& set);

    Affinity();

    cpu_set_t process_mask() const { return process_mask_; };
    cpu_set_t worker_mask() const;

private:
    cpu_set_t process_mask_{};
};

} // namespace tndm

#endif // AFFINITY_20210311_H
