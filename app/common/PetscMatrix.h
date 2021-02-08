#ifndef PETSCMATRIX_20210125_H
#define PETSCMATRIX_20210125_H

#include "common/PetscUtil.h"

#include "tensor/Tensor.h"

#include <petscmat.h>
#include <petscsys.h>
#include <petscsystypes.h>

#include <cstddef>
#include <vector>

namespace tndm {

class PetscMatrix {
public:
    PetscMatrix(std::size_t blockSize, std::size_t numLocalElems, std::size_t numElems,
                std::size_t const* gids, unsigned const* numLocal, unsigned const* numGhost,
                MPI_Comm comm);
    ~PetscMatrix() { MatDestroy(&A_); }

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
    void preallocate_SeqAIJ(std::size_t numLocalElems, unsigned const* numLocal, unsigned const*);
    void preallocate_MPIAIJ(std::size_t numLocalElems, unsigned const* numLocal,
                            unsigned const* numGhost);
    void preallocate_SeqBAIJ(std::size_t numLocalElems, unsigned const* numLocal, unsigned const*);
    void preallocate_MPIBAIJ(std::size_t numLocalElems, unsigned const* numLocal,
                             unsigned const* numGhost);
    void preallocate_IS(std::size_t numLocalElems, std::size_t numElems, unsigned const* numLocal,
                        unsigned const* numGhost);

    Mat A_;
    std::size_t block_size_;
};

} // namespace tndm

#endif // PETSCMATRIX_20210125_H
