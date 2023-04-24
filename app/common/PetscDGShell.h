#ifndef PETSCDGSHELL_20210302_H
#define PETSCDGSHELL_20210302_H

#include "config.h"
#include "form/AbstractDGOperator.h"
#include "common/PetscDGMatrix.h"

#include <petscmat.h>
#include <petscsystypes.h>

namespace tndm {

class PetscDGShell {
public:
    PetscDGShell(AbstractDGOperator<DomainDimension>& dgop);
    ~PetscDGShell();

    inline Mat mat() const { return A_; };

    Mat create_explicit(DGOperatorTopo const& topo, DGOpSparsityType stype);

private:
    static PetscErrorCode apply(Mat A, Vec x, Vec y);
    static PetscErrorCode apply_f(Mat A, Vec x, Vec y, void *ctx);

    std::size_t blocksize_;
    Mat A_;
};

} // namespace tndm

#endif // PETSCDGSHELL_20210302_H
