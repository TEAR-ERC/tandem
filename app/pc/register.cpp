#include "register.h"

#ifdef HAVE_LAPACK
extern "C" {
#include "eigdeflate.h"
}
#endif
extern "C" {
#include "lspoly.h"
}

namespace tndm {

PetscErrorCode register_PCs() {
    PetscFunctionBegin;
#ifdef HAVE_LAPACK
    CHKERRQ(PCRegister("eigdeflate", PCCreate_eigdeflate));
#endif
    PetscFunctionReturn(0);
}

PetscErrorCode register_KSPs() {
    PetscFunctionBegin;
    // CHKERRQ(KSPRegister("lspoly", KSPCreate_LSPoly));
    PetscFunctionReturn(0);
}

} // namespace tndm
