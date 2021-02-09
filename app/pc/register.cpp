#include "register.h"
#include "reig.h"

namespace tndm {

PetscErrorCode register_PCs() {
#ifdef HAVE_LAPACK
    CHKERRQ(PCRegister("reig", PCCreate_reig));
#endif
    PetscFunctionReturn(0);
}

} // namespace tndm
