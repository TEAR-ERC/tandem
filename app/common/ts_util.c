
#include <petsc.h>
#include <petscts.h>
#include <petsc/private/tsimpl.h>
#include "vecnest_util.h"

#include <errno.h>
#include <sys/stat.h>
#include <limits.h>

/*
https://stackoverflow.com/questions/2336242/recursive-mkdir-system-call-on-unix
*/
static int recursive_mkdir(const char *dir)
{
  int    num,error_number;
  char   tmp[PATH_MAX];
  char   *p = NULL;
  size_t len;
  
  snprintf(tmp,sizeof(tmp),"%s",dir);
  len = strlen(tmp);
  if (tmp[len - 1] == '/') {
    tmp[len - 1] = 0;
  }
  for (p = tmp + 1; *p; p++) {
    if (*p == '/') {
      *p = 0;
      num = mkdir(tmp,S_IRWXU);
      error_number = errno;
      if (num != 0) {
        switch(error_number) {
          case EEXIST:
            printf("[%s] directory exists\n",tmp);
            break;
          case EACCES:
            printf("[error %s] See EACCES man mkdir (code %d)\n",tmp,error_number);
            break;
          case EBADF:
            printf("[error %s] See EBADF man mkdir (code %d)\n",tmp,error_number);
            break;
          case EDQUOT:
            printf("[error %s] See EDQUOT man mkdir (code %d)\n",tmp,error_number);
            break;
          case EFAULT:
            printf("[error %s] See EFAULT man mkdir (code %d)\n",tmp,error_number);
            break;
          case EINVAL:
            printf("[error %s] See EINVAL man mkdir (code %d)\n",tmp,error_number);
            break;
          case ELOOP:
            printf("[error %s] See ELOOP man mkdir (code %d)\n",tmp,error_number);
            break;
          case EMLINK:
            printf("[error %s] See EMLINK man mkdir (code %d)\n",tmp,error_number);
            break;
        /*case ENOENT:
            printf("[error %s] See ENOENT man mkdir (code %d)\n",tmp,error_number);
            break;*/
          case ENOMEM:
            printf("[error %s] See ENOMEM man mkdir (code %d)\n",tmp,error_number);
            break;
          case ENOSPC:
            printf("[error %s] See ENOSPC man mkdir (code %d)\n",tmp,error_number);
            break;
          case ENOTDIR:
            printf("[error %s] See ENOTDIR man mkdir (code %d)\n",tmp,error_number);
            break;
          case EPERM:
            printf("[error %s] See EPERM man mkdir (code %d)\n",tmp,error_number);
            break;
          case EROFS:
            printf("[error %s] See EROFS man mkdir (code %d)\n",tmp,error_number);
            break;
          default:
            break;
        }
      } else {
        printf("[%s] directory created\n",tmp);
      }
      *p = '/';
    }
  }
  num = mkdir(tmp,S_IRWXU);
  error_number = errno;
  if (num != 0) {
    switch(error_number) {
      case EEXIST:
        printf("[%s] directory exists\n",tmp);
        break;
      case EACCES:
        printf("[error %s] See EACCES man mkdir (code %d)\n",tmp,error_number);
        break;
      case EBADF:
        printf("[error %s] See EBADF man mkdir (code %d)\n",tmp,error_number);
        break;
      case EDQUOT:
        printf("[error %s] See EDQUOT man mkdir (code %d)\n",tmp,error_number);
        break;
      case EFAULT:
        printf("[error %s] See EFAULT man mkdir (code %d)\n",tmp,error_number);
        break;
      case EINVAL:
        printf("[error %s] See EINVAL man mkdir (code %d)\n",tmp,error_number);
        break;
      case ELOOP:
        printf("[error %s] See ELOOP man mkdir (code %d)\n",tmp,error_number);
        break;
      case EMLINK:
        printf("[error %s] See EMLINK man mkdir (code %d)\n",tmp,error_number);
        break;
    /*case ENOENT:
        printf("[error %s] See ENOENT man mkdir (code %d)\n",tmp,error_number);
        break;*/
      case ENOMEM:
        printf("[error %s] See ENOMEM man mkdir (code %d)\n",tmp,error_number);
        break;
      case ENOSPC:
        printf("[error %s] See ENOSPC man mkdir (code %d)\n",tmp,error_number);
        break;
      case ENOTDIR:
        printf("[error %s] See ENOTDIR man mkdir (code %d)\n",tmp,error_number);
        break;
      case EPERM:
        printf("[error %s] See EPERM man mkdir (code %d)\n",tmp,error_number);
        break;
      case EROFS:
        printf("[error %s] See EROFS man mkdir (code %d)\n",tmp,error_number);
        break;
      default:
        break;
    }
  } else {
    printf("[%s] directory created\n",tmp);
  }
  return num;
}

static void MPICreateDirectory(MPI_Comm comm,const char dirname[])
{
  int rank,num,error_number;
  int ierr = 0;
  
  /* Generate a new directory on rank 0 */
  MPI_Comm_rank(comm,&rank);
  error_number = 0;
  if (rank == 0) {
    /*num = mkdir(dirname,S_IRWXU);*/
    num = recursive_mkdir(dirname);
    error_number = errno;
  }
  MPI_Bcast(&error_number,1,MPI_INT,0,comm);
  ierr = 0;
  switch(error_number) {
    case EACCES:  ierr = 1; break;
    case EBADF:   ierr = 1; break;
    case EDQUOT:  ierr = 1; break;
    case EFAULT:  ierr = 1; break;
    case EINVAL:  ierr = 1; break;
    case ELOOP:   ierr = 1; break;
    case EMLINK:  ierr = 1; break;
    /*case ENOENT:  ierr = 1; break;*/
    case ENOMEM:  ierr = 1; break;
    case ENOSPC:  ierr = 1; break;
    case ENOTDIR: ierr = 1; break;
    case EPERM:   ierr = 1; break;
    case EROFS:   ierr = 1; break;
  }
  if (ierr == 1) MPI_Abort(comm,ierr);
}

static PetscErrorCode _TSAdaptMembersView(TSAdapt adapt,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      iascii,isbinary,isnone,isglee;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt,TSADAPT_CLASSID,1);
  if (!viewer) {ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)adapt),&viewer);CHKERRQ(ierr);}
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(adapt,1,viewer,2);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
  if (!isbinary) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid viewer; open viewer with PetscViewerBinaryOpen()");

  /* Missing content which should be in TSAdaptView()*/
  ierr = PetscViewerBinaryWrite(viewer,&adapt->always_accept,1,PETSC_BOOL);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,&adapt->safety,       1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,&adapt->reject_safety,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,adapt->clip,          2,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,&adapt->dt_min,       1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,&adapt->dt_max,       1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,&adapt->ignore_max,   1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,&adapt->glee_use_local,    1,PETSC_BOOL);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,&adapt->scale_solve_failed,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,adapt->matchstepfac,       2,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,&adapt->timestepjustdecreased_delay,1,PETSC_INT);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,&adapt->timestepjustdecreased,      1,PETSC_INT);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode _TSView(TS ts,PetscViewer viewer)
{
  PetscErrorCode ierr;
  TSType         type;
  PetscBool      iascii,isstring,isundials,isbinary,isdraw;
  DMTS           sdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (!viewer) {
    ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)ts),&viewer);CHKERRQ(ierr);
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(ts,1,viewer,2);

  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERSTRING,&isstring);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERDRAW,&isdraw);CHKERRQ(ierr);

  if (iascii) {
  } else if (isstring) {
  } else if (isbinary) {
    PetscInt    classid = TS_FILE_CLASSID;
    MPI_Comm    comm;
    PetscMPIInt rank;
    char        type[256];

    ierr = PetscObjectGetComm((PetscObject)ts,&comm);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
    if (!rank) {
      ierr = PetscViewerBinaryWrite(viewer,&classid,1,PETSC_INT);CHKERRQ(ierr);
      ierr = PetscStrncpy(type,((PetscObject)ts)->type_name,256);CHKERRQ(ierr);
      ierr = PetscViewerBinaryWrite(viewer,type,256,PETSC_CHAR);CHKERRQ(ierr);

      /* PETSC FIX START */
      /* missing content */
      ierr = PetscViewerBinaryWrite(viewer,&ts->steps,1,PETSC_INT);CHKERRQ(ierr);
      ierr = PetscViewerBinaryWrite(viewer,&ts->ptime,1,PETSC_DOUBLE);CHKERRQ(ierr);
      ierr = PetscViewerBinaryWrite(viewer,&ts->time_step,1,PETSC_DOUBLE);CHKERRQ(ierr);
      ierr = PetscViewerBinaryWrite(viewer,&ts->ptime_prev,1,PETSC_DOUBLE);CHKERRQ(ierr);
      ierr = PetscViewerBinaryWrite(viewer,&ts->ptime_prev_rollback,1,PETSC_DOUBLE);CHKERRQ(ierr);
      ierr = PetscViewerBinaryWrite(viewer,&ts->solvetime,1,PETSC_DOUBLE);CHKERRQ(ierr);
      /* PETSC FIX END */
    }
    if (ts->ops->view) {
      ierr = (*ts->ops->view)(ts,viewer);CHKERRQ(ierr);
    }
    /* PETSC BUG */
    /*if (ts->adapt) {ierr = TSAdaptView(ts->adapt,viewer);CHKERRQ(ierr);}*/
    /* PETSC FIX START */
    if (!ts->adapt) {
      ierr = TSAdaptCreate(PetscObjectComm((PetscObject)ts),&ts->adapt);CHKERRQ(ierr);
      ierr = TSAdaptSetType(ts->adapt,TSADAPTNONE);CHKERRQ(ierr);
    }
    ierr = TSAdaptView(ts->adapt,viewer);CHKERRQ(ierr);
    
    /* PETSC FIX END */
    ierr = DMView(ts->dm,viewer);CHKERRQ(ierr);
    ierr = VecView(ts->vec_sol,viewer);CHKERRQ(ierr);
    ierr = DMGetDMTS(ts->dm,&sdm);CHKERRQ(ierr);
    ierr = DMTSView(sdm,viewer);CHKERRQ(ierr);

    ierr = _TSAdaptMembersView(ts->adapt,viewer);CHKERRQ(ierr);

  } else if (isdraw) {
  }
  if (ts->snes && ts->usessnes)  {
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = SNESView(ts->snes,viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  /*ierr = DMGetDMTS(ts->dm,&sdm);CHKERRQ(ierr);*/
  /*ierr = DMTSView(sdm,viewer);CHKERRQ(ierr);*/

  PetscFunctionReturn(0);
}

/*#include <../src/ts/impls/explicit/rk/rk.h>*/
/* valid for versions >= 3.13 */
typedef struct _RKTableau *RKTableau;
struct _RKTableau {
  char      *name;
  PetscInt   order;     /* Classical approximation order of the method i              */
  PetscInt   s;         /* Number of stages                                           */
  PetscInt   p;         /* Interpolation order                                        */
  PetscBool  FSAL;      /* flag to indicate if tableau is FSAL                        */
  PetscReal *A, *b, *c; /* Tableau                                                    */
  PetscReal *bembed;    /* Embedded formula of order one less (order-1)               */
  PetscReal *binterp;   /* Dense output formula                                       */
  PetscReal  ccfl;      /* Placeholder for CFL coefficient relative to forward Euler  */
};

typedef struct {
  RKTableau    tableau;
  Vec          X0;
  Vec         *Y;            /* States computed during the step                                              */
  Vec         *YdotRHS;      /* Function evaluations for the non-stiff part and contains all components      */
  Vec         *YdotRHS_fast; /* Function evaluations for the non-stiff part and contains fast components     */
  Vec         *YdotRHS_slow; /* Function evaluations for the non-stiff part and contains slow components     */
  Vec         *VecsDeltaLam; /* Increment of the adjoint sensitivity w.r.t IC at stage                       */
  Vec         *VecsSensiTemp;
  Vec          VecDeltaMu;    /* Increment of the adjoint sensitivity w.r.t P at stage                        */
  Vec         *VecsDeltaLam2; /* Increment of the 2nd-order adjoint sensitivity w.r.t IC at stage */
  Vec          VecDeltaMu2;   /* Increment of the 2nd-order adjoint sensitivity w.r.t P at stage */
  Vec         *VecsSensi2Temp;
  PetscScalar *work; /* Scalar work                                                                  */
  PetscInt     slow; /* flag indicates call slow components solver (0) or fast components solver (1) */
  PetscReal    stage_time;
  TSStepStatus status;
  PetscReal    ptime;
  PetscReal    time_step;
  PetscInt     dtratio; /* ratio between slow time step size and fast step size                         */
  IS           is_fast, is_slow;
  TS           subts_fast, subts_slow, subts_current, ts_root;
  PetscBool    use_multirate;
  Mat          MatFwdSensip0;
  Mat         *MatsFwdStageSensip;
  Mat         *MatsFwdSensipTemp;
  Vec          VecDeltaFwdSensipCol; /* Working vector for holding one column of the sensitivity matrix */
} TS_RK;

static PetscErrorCode _TSImplView_RK(TS ts,PetscViewer viewer)
{
  PetscErrorCode ierr;
  TSType         type;
  PetscBool      iascii,isstring,isundials,isbinary,isdraw;
  PetscBool      same;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  if (!viewer) {
    ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)ts),&viewer);CHKERRQ(ierr);
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(ts,1,viewer,2);

  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERSTRING,&isstring);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERDRAW,&isdraw);CHKERRQ(ierr);

  if (iascii) {
  } else if (isstring) {
  } else if (isbinary) {
    PetscInt    classid = TS_FILE_CLASSID;
    MPI_Comm    comm;
    PetscMPIInt rank;
    char        type[256];

    ierr = PetscObjectGetComm((PetscObject)ts,&comm);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
    if (!rank) {
      ierr = PetscViewerBinaryWrite(viewer,&classid,1,PETSC_INT);CHKERRQ(ierr);
      ierr = PetscStrncpy(type,((PetscObject)ts)->type_name,256);CHKERRQ(ierr);
      ierr = PetscViewerBinaryWrite(viewer,type,256,PETSC_CHAR);CHKERRQ(ierr);
    }
    
    PetscObjectTypeCompare((PetscObject)ts,TSRK,&same);
    if (same) {
      PetscInt i,s;
      TS_RK    *rk  = (TS_RK *)ts->data;
      /*RKTableau tab = rk->tableau;*/
      PetscBool FSAL;
 
      ierr = TSRKGetTableau(ts,&s,NULL,NULL,NULL,NULL,NULL,NULL,&FSAL);CHKERRQ(ierr);
      for (i=0; i<s; i++) {
/*
        PetscBool same;
        PetscInt NR;
        printf("rk->YdotRHS[%d] \n",i);
        PetscObjectTypeCompare((PetscObject)rk->YdotRHS[i],VECSEQ,&same); if (same) printf("Vec seq\n");
        PetscObjectTypeCompare((PetscObject)rk->YdotRHS[i],VECMPI,&same); if (same) printf("Vec mpi\n");
        PetscObjectTypeCompare((PetscObject)rk->YdotRHS[i],VECNEST,&same); if (same) printf("Vec nest\n");
        VecGetSize(rk->YdotRHS[i],&NR); printf("length %d\n",NR);
*/
        ierr = VecNestUpgradeOperations(rk->YdotRHS[i]);CHKERRQ(ierr);
        ierr = VecView(rk->YdotRHS[i],viewer);CHKERRQ(ierr);
      }
    }
  } else if (isdraw) {
  }
  PetscFunctionReturn(0);
}


static PetscErrorCode _TSAdaptMembersLoad(TSAdapt adapt,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      isbinary;
  char           type[256];

  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt,TSADAPT_CLASSID,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
  if (!isbinary) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid viewer; open viewer with PetscViewerBinaryOpen()");

  /* Missing content which should be included in TSAdaptLoad() */
  ierr = PetscViewerBinaryRead(viewer,&adapt->always_accept,1,NULL,PETSC_BOOL);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,&adapt->safety,       1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,&adapt->reject_safety,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,adapt->clip,          2,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,&adapt->dt_min,       1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,&adapt->dt_max,       1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,&adapt->ignore_max,   1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,&adapt->glee_use_local,    1,NULL,PETSC_BOOL);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,&adapt->scale_solve_failed,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,adapt->matchstepfac,       2,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,&adapt->timestepjustdecreased_delay,1,NULL,PETSC_INT);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,&adapt->timestepjustdecreased,      1,NULL,PETSC_INT);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode _TSLoad(TS ts, PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      isbinary;
  PetscInt       classid;
  char           type[256];
  DMTS           sdm;
  DM             dm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
  if (!isbinary) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid viewer; open viewer with PetscViewerBinaryOpen()");

  ierr = PetscViewerBinaryRead(viewer,&classid,1,NULL,PETSC_INT);CHKERRQ(ierr);
  if (classid != TS_FILE_CLASSID) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_ARG_WRONG,"Not TS next in file");
  ierr = PetscViewerBinaryRead(viewer,type,256,NULL,PETSC_CHAR);CHKERRQ(ierr);

  /* PETSC FIX START */
  /* missing content */
  ierr = PetscViewerBinaryRead(viewer,&ts->steps,1,NULL,PETSC_INT);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,&ts->ptime,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,&ts->time_step,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,&ts->ptime_prev,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,&ts->ptime_prev_rollback,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,&ts->solvetime,1,NULL,PETSC_DOUBLE);CHKERRQ(ierr);
  /* PETSC BUG FIX END */

  ierr = TSSetType(ts, type);CHKERRQ(ierr);
  if (ts->ops->load) {
    ierr = (*ts->ops->load)(ts,viewer);CHKERRQ(ierr);
  }

  /* PETSC BUG FIX START */
  if (!ts->adapt) {
    /*
       The reason why we create/load here is because _TSView() is guaranteed to write out a TSAdapt object.
       However, depending on the TSType TSAdapt may not be created by default.
       Also, some implementations of TS will actually create / load TSAdapt _within_ ts->ops->load().
       Hence if we create the TSAdapt (because it didn't exist), we must also load it.
       Putting TSAdaptLoad() outside of this if statement may sometimes result in TSAdapt being attempted to be loaded twice.
    */
    ierr = TSAdaptCreate(PetscObjectComm((PetscObject)ts),&ts->adapt);CHKERRQ(ierr);
    ierr = TSAdaptLoad(ts->adapt,viewer);CHKERRQ(ierr);
  }
  if (!ts->ops->load) { /* If load() is null, there was no possibility to load TSAdapt */
    ierr = TSAdaptLoad(ts->adapt,viewer);CHKERRQ(ierr);
  }

  /* PETSC BUG FIX END */
  ierr = DMCreate(PetscObjectComm((PetscObject)ts),&dm);CHKERRQ(ierr);
  ierr = DMLoad(dm,viewer);CHKERRQ(ierr);
  ierr = TSSetDM(ts,dm);CHKERRQ(ierr);
  if (!ts->vec_sol) {
      ierr = DMCreateGlobalVector(ts->dm,&ts->vec_sol);CHKERRQ(ierr);
  }
  ierr = VecLoad(ts->vec_sol,viewer);CHKERRQ(ierr);
  ierr = DMGetDMTS(ts->dm,&sdm);CHKERRQ(ierr);
  ierr = DMTSLoad(sdm,viewer);CHKERRQ(ierr);

  ierr = _TSAdaptMembersLoad(ts->adapt,viewer);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode _TSImplLoad_RK(TS ts, PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      isbinary;
  PetscInt       classid;
  char           type[256];
  PetscBool      same;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
  if (!isbinary) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid viewer; open viewer with PetscViewerBinaryOpen()");

  ierr = PetscViewerBinaryRead(viewer,&classid,1,NULL,PETSC_INT);CHKERRQ(ierr);
  if (classid != TS_FILE_CLASSID) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_ARG_WRONG,"Not TS next in file");
  ierr = PetscViewerBinaryRead(viewer,type,256,NULL,PETSC_CHAR);CHKERRQ(ierr);

  PetscObjectTypeCompare((PetscObject)ts,TSRK,&same);
  if (same) {
    PetscInt i,s;
    TS_RK    *rk  = (TS_RK *)ts->data;
    /*RKTableau tab = rk->tableau;*/
    PetscBool FSAL;
            
    ierr = TSRKGetTableau(ts,&s,NULL,NULL,NULL,NULL,NULL,NULL,&FSAL);CHKERRQ(ierr);
    for (i=0; i<s; i++) {
      PetscBool isnest;

      if (rk->YdotRHS[i]) {
        ierr = VecNestUpgradeOperations(rk->YdotRHS[i]);CHKERRQ(ierr);
      }
      if (rk->YdotRHS[i]) {
        ierr = VecLoad(rk->YdotRHS[i],viewer);CHKERRQ(ierr);
      }
    }
  }

  PetscFunctionReturn(0);
}

typedef enum { TSCP_STORAGE_NONE=0, TSCP_STORAGE_UNLIMITED, TSCP_STORAGE_LIMITED } TSCheckPointStorageType;

const char *TSCheckPointStorageTypes[] = { "none", "unlimited", "limited", "TSCheckPointStorageType", "TSCP_STORAGE_", NULL };

struct _TSCheckPoint {
  char      path_prefix[PETSC_MAX_PATH_LEN];
  char      path_step[PETSC_MAX_PATH_LEN];
  PetscInt  checkpoint_frequency_step;
  PetscReal checkpoint_frequency_cputime_minutes;
  PetscReal checkpoint_frequency_time_physical;
  PetscInt  n, step_last;
  PetscReal cputime_last,time_last;
  TSCheckPointStorageType mode;
  PetscInt limited_size;
};
typedef struct _TSCheckPoint *TSCheckPoint;

/* create directory checkpoint_prefix/stepXXXX */
static PetscErrorCode ts_checkpoint_create_path(TS ts, TSCheckPoint cp, PetscInt step)
{
  size_t L;
  const char *basenames[] = { "none", "step", "chkpnt", 0 };
  char *base = NULL;
  PetscInt fileindex;

  PetscFunctionBeginUser;
  if (cp->mode == TSCP_STORAGE_UNLIMITED) { fileindex = step; }
  if (cp->mode == TSCP_STORAGE_LIMITED) { fileindex = cp->n % cp->limited_size; }
  base = (char*)basenames[(int)cp->mode];
  PetscStrlen(cp->path_prefix,&L);
  if (L == 0) {
    PetscSNPrintf(cp->path_step,PETSC_MAX_PATH_LEN-1,"%s%d",base,(int)fileindex);
  } else {
    if (cp->path_prefix[L-1] == '/') {
      PetscSNPrintf(cp->path_step,PETSC_MAX_PATH_LEN-1,"%s%s%d",cp->path_prefix,base,(int)fileindex);
    } else {
      PetscSNPrintf(cp->path_step,PETSC_MAX_PATH_LEN-1,"%s/%s%d",cp->path_prefix,base,(int)fileindex);
    }
  }
  MPICreateDirectory(PetscObjectComm((PetscObject)ts),cp->path_step);
  PetscFunctionReturn(0);
}

static PetscErrorCode ts_checkpoint_test(TS ts, TSCheckPoint cp, PetscBool *generate)
{
  PetscInt       step;
  PetscLogDouble cputime;
  PetscReal      time_physical;
  MPI_Comm       comm;
  int            bi;
  PetscErrorCode ierr;

  PetscFunctionBeginUser; 
  *generate = PETSC_FALSE;
  comm = PetscObjectComm((PetscObject)ts);
  
  PetscTime(&cputime);
  MPI_Bcast(&cputime,1,MPIU_PETSCLOGDOUBLE,0,comm);
  if (cputime - cp->cputime_last >= cp->checkpoint_frequency_cputime_minutes*60.0) {
    cp->cputime_last = cputime;
    *generate = PETSC_TRUE;
    PetscPrintf(comm,"[TSCheckpoint] Triggered by: \"cputime\"\n");
  }
  
  ierr = TSGetStepNumber(ts, &step);CHKERRQ(ierr);
  if (step%cp->checkpoint_frequency_step == 0) {
    cp->step_last = step;
    *generate = PETSC_TRUE;
    PetscPrintf(comm,"[TSCheckpoint] Triggered by: \"step\"\n");
  }
  
  TSGetTime(ts,&time_physical);
  if (time_physical - cp->time_last >= cp->checkpoint_frequency_time_physical) {
    cp->time_last = time_physical;
    *generate = PETSC_TRUE;
    PetscPrintf(comm,"[TSCheckpoint] Triggered by: \"physical_time\"\n");
  }
  
  bi = (int)(*generate);
  MPI_Bcast(&bi,1,MPI_INT,0,comm);
  if (bi == 0) {  *generate = PETSC_FALSE;
  } else {        *generate = PETSC_TRUE; }
  
  PetscFunctionReturn(0);
}

static PetscErrorCode ts_checkpoint_write(TS ts, TSCheckPoint cp)
{
  PetscInt       step;
  PetscReal      time, dt;
  char           filename[PETSC_MAX_PATH_LEN];
  PetscViewer    viewer;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = TSGetStepNumber(ts,&step);CHKERRQ(ierr);
  ierr = TSGetTime(ts,&time);CHKERRQ(ierr);
  ierr = TSGetTimeStep(ts,&dt);CHKERRQ(ierr);
 
  PetscPrintf(PetscObjectComm((PetscObject)ts),"[TSCheckpoint]   writing checkpoint data for step %d -> path %s\n",(int)step,cp->path_step);
  PetscSNPrintf(filename,PETSC_MAX_PATH_LEN-1,"%s/ts.bin",cp->path_step);
  ierr = PetscViewerBinaryOpen(PetscObjectComm((PetscObject)ts),filename,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = _TSView(ts,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  PetscSNPrintf(filename,PETSC_MAX_PATH_LEN-1,"%s/ts_impl.bin",cp->path_step);
  ierr = PetscViewerBinaryOpen(PetscObjectComm((PetscObject)ts),filename,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = _TSImplView_RK(ts,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr); 

  PetscSNPrintf(filename,PETSC_MAX_PATH_LEN-1,"%s/state.vec",cp->path_step);
  ierr = PetscViewerBinaryOpen(PetscObjectComm((PetscObject)ts),filename,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = VecView(ts->vec_sol,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  {
    int commrank;
    MPI_Comm_rank(PetscObjectComm((PetscObject)ts),&commrank);
    if (commrank == 0) {
      char infoname[PETSC_MAX_PATH_LEN];

      /* Write a CSV file with basic time info */
      PetscSNPrintf(infoname,PETSC_MAX_PATH_LEN-1,"%s/checkpoint_info.csv",cp->path_step);
      ierr = PetscViewerASCIIOpen(PETSC_COMM_SELF,infoname,&viewer);CHKERRQ(ierr);
      PetscViewerASCIIPrintf(viewer,"# time, time_step, step_number, checkpoint_number, path\n");
      PetscViewerASCIIPrintf(viewer,"%1.18e,%1.18e,%ld,%d,%s\n",(double)time,(double)dt,(long int)step,(int)cp->n,cp->path_step);
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

      /* Write a helper file that enables restating from the last checkpoint written */
      PetscSNPrintf(infoname,PETSC_MAX_PATH_LEN-1,"%s/last_checkpoint.txt",cp->path_prefix);
      ierr = PetscViewerASCIIOpen(PETSC_COMM_SELF,infoname,&viewer);CHKERRQ(ierr);
      PetscViewerASCIIPrintf(viewer,"%s\n",cp->path_step);
      PetscViewerASCIIPrintf(viewer,"# is the last written checkpoint file\n");
      PetscViewerASCIIPrintf(viewer,"# time, time_step, step_number, checkpoint_number, path\n");
      PetscViewerASCIIPrintf(viewer,"# %1.18e,%1.18e,%ld,%d,%s\n",(double)time,(double)dt,(long int)step,(int)cp->n,cp->path_step);

      PetscViewerASCIIPrintf(viewer,"\n# Previously used checkpoint options\n");
      PetscViewerASCIIPrintf(viewer,"#ts_checkpoint.save_directory %s\n",cp->path_prefix);
      PetscViewerASCIIPrintf(viewer,"#ts_checkpoint.freq_step %d\n",(int)cp->checkpoint_frequency_step);
      PetscViewerASCIIPrintf(viewer,"#ts_checkpoint.freq_cputime %1.4e\n",(double)cp->checkpoint_frequency_cputime_minutes);
      PetscViewerASCIIPrintf(viewer,"#ts_checkpoint.freq_physical_time %1.4e\n",(double)cp->checkpoint_frequency_time_physical);
      PetscViewerASCIIPrintf(viewer,"#ts_checkpoint.storage_type %s\n",TSCheckPointStorageTypes[(PetscInt)cp->mode]);
      if (cp->mode == TSCP_STORAGE_LIMITED) {
        PetscViewerASCIIPrintf(viewer,"#ts_checkpoint.storage_limited_size %d\n",(int)cp->limited_size);
      }
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    }
  }

  cp->n++;
  PetscFunctionReturn(0);
}

static PetscErrorCode ts_checkpoint_load(TS ts, const char pathname[])
{
  char           filename[PETSC_MAX_PATH_LEN];
  PetscViewer    viewer;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  PetscPrintf(PetscObjectComm((PetscObject)ts),"[TSCheckpoint]   loading checkpoint data <- path %s\n",pathname);
 
  PetscSNPrintf(filename,PETSC_MAX_PATH_LEN-1,"%s/ts.bin",pathname);
  ierr = PetscViewerBinaryOpen(PetscObjectComm((PetscObject)ts),filename,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = _TSLoad(ts,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  PetscSNPrintf(filename,PETSC_MAX_PATH_LEN-1,"%s/ts_impl.bin",pathname);
  ierr = PetscViewerBinaryOpen(PetscObjectComm((PetscObject)ts),filename,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = _TSImplLoad_RK(ts,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  PetscSNPrintf(filename,PETSC_MAX_PATH_LEN-1,"%s/state.vec",pathname);
  ierr = PetscViewerBinaryOpen(PetscObjectComm((PetscObject)ts),filename,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = VecLoad(ts->vec_sol,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode ts_checkpoint(TS ts)
{
  PetscContainer container = NULL;
  TSCheckPoint   tsc = NULL;
  PetscBool      generate;
  PetscInt       step;
  PetscErrorCode ierr; 
 
  PetscFunctionBeginUser;
  ierr = PetscObjectQuery((PetscObject)ts,"_TSCheckPoint",(PetscObject*)&container);CHKERRQ(ierr);
  if (!container) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_ORDER,"Must call ts_checkpoint_configure() first");
  PetscContainerGetPointer(container,(void**)&tsc);
  ierr = ts_checkpoint_test(ts,tsc,&generate);CHKERRQ(ierr);
  if (generate) {
    /* create directory checkpoint_prefix/stepXXXX/ */
    TSGetStepNumber(ts,&step);
    ierr = ts_checkpoint_create_path(ts,tsc,step);CHKERRQ(ierr);
    
    /* write files ts.bin, state.vec, ts_impl.bin */
    ierr = ts_checkpoint_write(ts,tsc);CHKERRQ(ierr);
  }
  tsc->path_step[0] = '\0';
  PetscFunctionReturn(0);
}

PetscErrorCode ts_checkpoint_configure(TS ts, const char* tsCheckpointSaveDirectory,
                                       int tsCheckpointFrequencyStep,
                                       double tsCheckpointFrequencyCputimeMinutes,
                                       double tsCheckpointFrequencyPhysical, int storageType,
                                       int tsCheckpointStorageLimitedSize)
{
  PetscContainer container = NULL;
  TSCheckPoint   tsc = NULL;
  MPI_Comm       comm;
  PetscBool      found;
  
  PetscFunctionBeginUser;
  comm = PetscObjectComm((PetscObject)ts);
  /* Create checkpoint helper object, initialize values */
  PetscCalloc1(1,&tsc);
  tsc->checkpoint_frequency_step = tsCheckpointFrequencyStep;
  tsc->checkpoint_frequency_cputime_minutes = tsCheckpointFrequencyCputimeMinutes;
  tsc->checkpoint_frequency_time_physical = tsCheckpointFrequencyPhysical;
  tsc->mode = (TSCheckPointStorageType)storageType;
  tsc->limited_size = tsCheckpointStorageLimitedSize;

  tsc->path_prefix[0] = '\0';
  PetscSNPrintf(tsc->path_prefix, PETSC_MAX_PATH_LEN, "%s", tsCheckpointSaveDirectory);
  tsc->path_step[0] = '\0';

  PetscPrintf(comm,"TS ts_checkpoint.storage_type %s\n",TSCheckPointStorageTypes[(PetscInt)tsc->mode]);
  if (tsc->mode != TSCP_STORAGE_NONE) {
    PetscPrintf(comm,"TS ts_checkpoint.save_directory %s\n",tsc->path_prefix);
    PetscPrintf(comm,"TS ts_checkpoint.freq_step %d\n",(int)tsc->checkpoint_frequency_step);
    PetscPrintf(comm,"TS ts_checkpoint.freq_cputime %1.4e\n",(double)tsc->checkpoint_frequency_cputime_minutes);
    PetscPrintf(comm,"TS ts_checkpoint.freq_physical_time %1.4e\n",(double)tsc->checkpoint_frequency_time_physical);
  }
  if (tsc->mode == TSCP_STORAGE_LIMITED) {
    PetscPrintf(comm,"TS ts_checkpoint.storage_limited_size %d\n",(int)tsc->limited_size);
  }

  PetscTime(&tsc->cputime_last);
  if (tsc->mode != TSCP_STORAGE_NONE) {
    size_t len;

    PetscStrlen(tsc->path_prefix,&len);
    if (len != 0) {
      MPICreateDirectory(comm,tsc->path_prefix);
    }
  }

  if (tsc->mode != TSCP_STORAGE_NONE) {
    /* Push checkpoint object into PetscContainer. Attach container to TS */
    PetscContainerCreate(PetscObjectComm((PetscObject)ts),&container);
    PetscContainerSetPointer(container,tsc);
    PetscContainerSetUserDestroy(container,PetscContainerUserDestroyDefault);
    PetscObjectCompose((PetscObject)ts,"_TSCheckPoint",(PetscObject)container);
    PetscObjectDereference((PetscObject)container); /* Let ts free container */

    /* Set function to be called after each TS time step */
    TSSetPostStep(ts,ts_checkpoint);
  } else {
    PetscFree(tsc);
  }

  PetscFunctionReturn(0);
}

PetscErrorCode ts_checkpoint_restart(TS ts, const char* tsCheckpointLoadDirectory)
{
  PetscBool found = PETSC_FALSE;
  char      load_path_prefix[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  load_path_prefix[0] = '\0';
  PetscSNPrintf(load_path_prefix, PETSC_MAX_PATH_LEN, "%s", tsCheckpointLoadDirectory);
  ierr = ts_checkpoint_load(ts,load_path_prefix);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

