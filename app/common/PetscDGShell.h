#ifndef PETSCDGSHELL_20210302_H
#define PETSCDGSHELL_20210302_H

#include "config.h"
#include "form/AbstractDGOperator.h"

#include <petscmat.h>
#include <petscsystypes.h>

namespace tndm {

class PetscDGShell {
public:
    PetscDGShell(AbstractDGOperator<DomainDimension>& dgop);
    ~PetscDGShell();

    inline Mat mat() const { return A_; };

private:
    static PetscErrorCode apply(Mat A, Vec x, Vec y);

    Mat A_;
};

} // namespace tndm

#endif // PETSCDGSHELL_20210302_H
