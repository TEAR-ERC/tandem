#include "register.h"
#include "reig.h"

namespace tndm {

PetscErrorCode register_PCs() {
    CHKERRQ(PCRegister("reig", PCCreate_reig));
    PetscFunctionReturn(0);
}

} // namespace tndm
