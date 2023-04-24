#ifndef PETSCDGSHELLPt_20210302_H
#define PETSCDGSHELLPt_20210302_H

#include "config.h"
#include "form/AbstractDGOperator.h"
#include "common/PetscDGMatrix.h"

#include <petscmat.h>
#include <petscsystypes.h>

namespace tndm {

class PetscDGShellPtAP {
public:
    PetscDGShellPtAP(AbstractDGOperator<DomainDimension>& dgop, Mat P, std::size_t to_blocksize);
    ~PetscDGShellPtAP();

    inline Mat mat() const { return A_; };

    Mat create_explicit(DGOperatorTopo const& topo, DGOpSparsityType stype);

private:
    static PetscErrorCode destroy(Mat A);
    static PetscErrorCode apply(Mat A, Vec x, Vec y);
    static PetscErrorCode apply_f(Mat A, Vec x, Vec y, void *ctx);

    AbstractDGOperator<DomainDimension>* dgop_;
    std::size_t blocksize_, to_blocksize_;
    Mat A_, P_;
    Vec xf_, yf_;
};

} // namespace tndm

#endif // PETSCDGSHELLPt_20210302_H
