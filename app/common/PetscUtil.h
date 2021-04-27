#ifndef PETSCUTIL_20200910_H
#define PETSCUTIL_20200910_H

#include "util/Hash.h"

#include <exception>
#include <petscsys.h>

#define CHKERRTHROW(ierr)                                                                          \
    do {                                                                                           \
        PetscErrorCode ierr_ = (ierr);                                                             \
        if (PetscUnlikely(ierr_)) {                                                                \
            PetscError(PETSC_COMM_SELF, __LINE__, PETSC_FUNCTION_NAME, __FILE__, ierr_,            \
                       PETSC_ERROR_REPEAT, " ");                                                   \
            char const* what;                                                                      \
            PetscErrorMessage(ierr_, &what, NULL);                                                 \
            throw ::tndm::petsc_error(what, ierr_);                                                \
        }                                                                                          \
    } while (false)

#define HASH_DEF(def) ::tndm::fnv1a((def), sizeof(def)-1)

namespace tndm {

class petsc_error : public std::exception {
public:
    petsc_error(char const* what, PetscErrorCode ierr) noexcept : what_(what), ierr_(ierr) {}
    char const* what() const noexcept override { return what_; }
    PetscErrorCode err_code() const noexcept { return ierr_; }

private:
    char const* what_;
    PetscErrorCode ierr_;
};

} // namespace tndm

#endif // PETSCUTIL_20200910_H
