
#include <petsc.h>
#include <petscvec.h>
#include <petsc/private/vecimpl.h>

static PetscErrorCode _VecView_Nest(Vec x,PetscViewer viewer)
{
  PetscBool      isascii,isbinary;
  PetscInt       i,nb;
  Vec            *v;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
  ierr = VecNestGetSubVecs(x,&nb,&v);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"VecNest, rows=%" PetscInt_FMT ",  structure: \n",nb);CHKERRQ(ierr);
    for (i=0; i<nb; i++) {
      VecType  type;
      char     name[256] = "",prefix[256] = "";
      PetscInt NR;
      const char *obj_name,*obj_prefix;

      ierr = PetscObjectGetName((PetscObject)v[i],&obj_name);CHKERRQ(ierr);
      ierr = PetscObjectGetOptionsPrefix((PetscObject)v[i],&obj_prefix);CHKERRQ(ierr);
      ierr = VecGetSize(v[i],&NR);CHKERRQ(ierr);
      ierr = VecGetType(v[i],&type);CHKERRQ(ierr);
      if (obj_name) {ierr = PetscSNPrintf(name,sizeof(name),"name=\"%s\", ",obj_name);CHKERRQ(ierr);}
      if (obj_prefix) {ierr = PetscSNPrintf(prefix,sizeof(prefix),"prefix=\"%s\", ",obj_prefix);CHKERRQ(ierr);}

      ierr = PetscViewerASCIIPrintf(viewer,"(%" PetscInt_FMT ") : %s%stype=%s, rows=%" PetscInt_FMT " \n",i,name,prefix,type,NR);CHKERRQ(ierr);

      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);             /* push1 */
      ierr = VecView(v[i],viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);              /* pop1 */
    }
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);                /* pop0 */
  } else if (isbinary) {
    for (i=0; i<nb; i++) {PetscInt NR;
      ierr = VecView(v[i],viewer);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode _VecLoad_Nest(Vec x,PetscViewer viewer)
{
  PetscBool      isbinary;
  PetscInt       i,nb;
  Vec            *v;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
  if (!isbinary) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid viewer; open viewer with PetscViewerBinaryOpen()");
  ierr = VecNestGetSubVecs(x,&nb,&v);CHKERRQ(ierr);
  for (i=0; i<nb; i++) {
    ierr = VecLoad(v[i],viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecNestUpgradeOperations(Vec x)
{
  PetscBool isnest;
  PetscObjectTypeCompare((PetscObject)x,VECNEST,&isnest);
  if (isnest) {
    x->ops->view = _VecView_Nest;
    x->ops->load = _VecLoad_Nest;
  }
  PetscFunctionReturn(0);
}
