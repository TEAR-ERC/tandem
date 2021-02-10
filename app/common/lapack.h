#ifndef LAPACK_20210208_H
#define LAPACK_20210208_H

#include "FC.h"

#define FC_dsyev FC_GLOBAL(dsyev, DSYEV)
extern void FC_dsyev(char const* jobz, char const* uplo, int const* n, double* A, int const* lda,
                     double* W, double* work, int const* lwork, int* info);

#endif // LAPACK_20210208_H
