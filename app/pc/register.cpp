#include "register.h"

#ifdef HAVE_LAPACK
extern "C" {
#include "amgeig.h"
#include "eigdeflate.h"
}
#endif

namespace tndm {

PetscErrorCode register_PCs() {
#ifdef HAVE_LAPACK
    // CHKERRQ(PCRegister("amgeig", PCCreate_amgeig));
    CHKERRQ(PCRegister("eigdeflate", PCCreate_eigdeflate));
#endif
    PetscFunctionReturn(0);
}

} // namespace tndm
