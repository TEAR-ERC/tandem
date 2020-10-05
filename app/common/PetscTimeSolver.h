#ifndef PETSCTS_20201001_H
#define PETSCTS_20201001_H

#include "common/PetscBlockVector.h"
#include "common/PetscUtil.h"
#include "tensor/Tensor.h"

#include <petscsys.h>
#include <petscsystypes.h>
#include <petscts.h>

#include <cstddef>
#include <memory>

namespace tndm {

class PetscTimeSolver {
public:
    template <typename TimeOp> PetscTimeSolver(TimeOp& timeop) {
        state_ = std::make_unique<PetscBlockVector>(timeop.block_size(), timeop.numLocalElements(),
                                                    timeop.comm());
        timeop.initial_condition(*state_);
    }
    ~PetscTimeSolver() { TSDestroy(&ts_); }

    auto& state() { return *state_; }
    auto const& state() const { return *state_; }

private:
    std::unique_ptr<PetscBlockVector> state_;
    TS ts_ = nullptr;
};

} // namespace tndm

#endif // PETSCTS_20201001_H
