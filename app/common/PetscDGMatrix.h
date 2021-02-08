#ifndef PETSCDGMATRIX_20210208_H
#define PETSCDGMATRIX_20210208_H

#include "common/PetscUtil.h"

#include "form/DGOperatorTopo.h"
#include "tensor/Tensor.h"

#include <petscmat.h>
#include <petscsys.h>
#include <petscsystypes.h>

#include <cstddef>
#include <vector>

namespace tndm {

class PetscDGMatrix {
public:
    PetscDGMatrix(std::size_t blockSize, DGOperatorTopo const& topo);
    ~PetscDGMatrix() { MatDestroy(&A_); }

    void add_block(std::size_t ib_local, std::size_t jb_local, Matrix<double> const& values) {
        PetscInt pib = ib_local, pjb = jb_local;
        MatSetValuesBlockedLocal(A_, 1, &pib, 1, &pjb, values.data(), ADD_VALUES);
    }
    void begin_assembly() {}
    void end_assembly() {
        CHKERRTHROW(MatAssemblyBegin(A_, MAT_FINAL_ASSEMBLY));
        CHKERRTHROW(MatAssemblyEnd(A_, MAT_FINAL_ASSEMBLY));
    }

    void set_zero() { MatZeroEntries(A_); }
    Mat mat() const { return A_; };

private:
    std::vector<PetscInt> nnz_aij(std::size_t numLocalElems, unsigned const* nnz);
    std::vector<PetscInt> nnz_baij(std::size_t numLocalElems, unsigned const* nnz);
    void preallocate_SeqAIJ(DGOperatorTopo const& topo);
    void preallocate_MPIAIJ(DGOperatorTopo const& topo);
    void preallocate_SeqBAIJ(DGOperatorTopo const& topo);
    void preallocate_MPIBAIJ(DGOperatorTopo const& topo);
    void preallocate_IS(DGOperatorTopo const& topo);

    Mat A_;
    std::size_t block_size_;
};

} // namespace tndm

#endif // PETSCDGMATRIX_20210208_H
