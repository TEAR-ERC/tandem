
#include <petsc.h>
#include <petscts.h>
#include <petsc/private/tsimpl.h>

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


static PetscErrorCode _TSView(TS ts,PetscViewer viewer)
{
  PetscErrorCode ierr;
  TSType         type;
  PetscBool      iascii,isstring,isundials,isbinary,isdraw;
  DMTS           sdm;
#if defined(PETSC_HAVE_SAWS)
  PetscBool      issaws;
#endif

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
#if defined(PETSC_HAVE_SAWS)
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERSAWS,&issaws);CHKERRQ(ierr);
#endif

  if (iascii) {
    ierr = PetscObjectPrintClassNamePrefixType((PetscObject)ts,viewer);CHKERRQ(ierr);
    if (ts->ops->view) {
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = (*ts->ops->view)(ts,viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
    if (ts->max_steps < PETSC_MAX_INT) {
      ierr = PetscViewerASCIIPrintf(viewer,"  maximum steps=%D\n",ts->max_steps);CHKERRQ(ierr);
    }
    if (ts->max_time < PETSC_MAX_REAL) {
      ierr = PetscViewerASCIIPrintf(viewer,"  maximum time=%g\n",(double)ts->max_time);CHKERRQ(ierr);
    }
    if (ts->usessnes) {
      PetscBool lin;
      if (ts->problem_type == TS_NONLINEAR) {
        ierr = PetscViewerASCIIPrintf(viewer,"  total number of nonlinear solver iterations=%D\n",ts->snes_its);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer,"  total number of linear solver iterations=%D\n",ts->ksp_its);CHKERRQ(ierr);
      ierr = PetscObjectTypeCompareAny((PetscObject)ts->snes,&lin,SNESKSPONLY,SNESKSPTRANSPOSEONLY,"");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  total number of %slinear solve failures=%D\n",lin ? "" : "non",ts->num_snes_failures);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  total number of rejected steps=%D\n",ts->reject);CHKERRQ(ierr);
    if (ts->vrtol) {
      ierr = PetscViewerASCIIPrintf(viewer,"  using vector of relative error tolerances, ");CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  using relative error tolerance of %g, ",(double)ts->rtol);CHKERRQ(ierr);
    }
    if (ts->vatol) {
      ierr = PetscViewerASCIIPrintf(viewer,"  using vector of absolute error tolerances\n");CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  using absolute error tolerance of %g\n",(double)ts->atol);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = TSAdaptView(ts->adapt,viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  } else if (isstring) {
    ierr = TSGetType(ts,&type);CHKERRQ(ierr);
    ierr = PetscViewerStringSPrintf(viewer," TSType: %-7.7s",type);CHKERRQ(ierr);
    if (ts->ops->view) {ierr = (*ts->ops->view)(ts,viewer);CHKERRQ(ierr);}
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
    //if (ts->adapt) {ierr = TSAdaptView(ts->adapt,viewer);CHKERRQ(ierr);}
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
    PetscDraw draw;
    char      str[36];
    PetscReal x,y,bottom,h;

    ierr   = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
    ierr   = PetscDrawGetCurrentPoint(draw,&x,&y);CHKERRQ(ierr);
    ierr   = PetscStrcpy(str,"TS: ");CHKERRQ(ierr);
    ierr   = PetscStrcat(str,((PetscObject)ts)->type_name);CHKERRQ(ierr);
    ierr   = PetscDrawStringBoxed(draw,x,y,PETSC_DRAW_BLACK,PETSC_DRAW_BLACK,str,NULL,&h);CHKERRQ(ierr);
    bottom = y - h;
    ierr   = PetscDrawPushCurrentPoint(draw,x,bottom);CHKERRQ(ierr);
    if (ts->ops->view) {
      ierr = (*ts->ops->view)(ts,viewer);CHKERRQ(ierr);
    }
    if (ts->adapt) {ierr = TSAdaptView(ts->adapt,viewer);CHKERRQ(ierr);}
    if (ts->snes)  {ierr = SNESView(ts->snes,viewer);CHKERRQ(ierr);}
    ierr = PetscDrawPopCurrentPoint(draw);CHKERRQ(ierr);
#if defined(PETSC_HAVE_SAWS)
  } else if (issaws) {
    PetscMPIInt rank;
    const char  *name;

    ierr = PetscObjectGetName((PetscObject)ts,&name);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
    if (!((PetscObject)ts)->amsmem && !rank) {
      char       dir[1024];

      ierr = PetscObjectViewSAWs((PetscObject)ts,viewer);CHKERRQ(ierr);
      ierr = PetscSNPrintf(dir,1024,"/PETSc/Objects/%s/time_step",name);CHKERRQ(ierr);
      PetscStackCallSAWs(SAWs_Register,(dir,&ts->steps,1,SAWs_READ,SAWs_INT));
      ierr = PetscSNPrintf(dir,1024,"/PETSc/Objects/%s/time",name);CHKERRQ(ierr);
      PetscStackCallSAWs(SAWs_Register,(dir,&ts->ptime,1,SAWs_READ,SAWs_DOUBLE));
    }
    if (ts->ops->view) {
      ierr = (*ts->ops->view)(ts,viewer);CHKERRQ(ierr);
    }
#endif
  }
  if (ts->snes && ts->usessnes)  {
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = SNESView(ts->snes,viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  ierr = DMGetDMTS(ts->dm,&sdm);CHKERRQ(ierr);
  ierr = DMTSView(sdm,viewer);CHKERRQ(ierr);

  ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)ts,TSSUNDIALS,&isundials);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode _TSLoad(TS ts, PetscViewer viewer)
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
    // The reason why we create/load here is because _TSView() is gaurnteed to write out a TSAdapt object.
    // However, depending on the TSType TSAdapt may not be created by default.
    // Also, some implementations of TS will actually create / load TSAdapt _within_ ts->ops->load().
    // Hence if we create the TSAdapt (because it didn't exist), we must also load it.
    // Putting TSAdaptLoad() outside of this if statement may sometimes result in TSAdapt being attempted to be loaded twice.
    ierr = TSAdaptCreate(PetscObjectComm((PetscObject)ts),&ts->adapt);CHKERRQ(ierr);
    ierr = TSAdaptLoad(ts->adapt,viewer);CHKERRQ(ierr);
  }
  /* PETSC BUG FIX END */
  ierr = DMCreate(PetscObjectComm((PetscObject)ts),&dm);CHKERRQ(ierr);
  ierr = DMLoad(dm,viewer);CHKERRQ(ierr);
  ierr = TSSetDM(ts,dm);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(ts->dm,&ts->vec_sol);CHKERRQ(ierr);
  ierr = VecLoad(ts->vec_sol,viewer);CHKERRQ(ierr);
  ierr = DMGetDMTS(ts->dm,&sdm);CHKERRQ(ierr);
  ierr = DMTSLoad(sdm,viewer);CHKERRQ(ierr);

  ierr = _TSAdaptMembersLoad(ts->adapt,viewer);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode tandem_TSView(TS ts,const char filename[])
{ 
  PetscErrorCode ierr;
  PetscViewer viewer;

  ierr = PetscViewerBinaryOpen(PetscObjectComm((PetscObject)ts),filename,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = _TSView(ts,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode tandem_TSLoad(TS ts,const char filename[])
{
  PetscErrorCode ierr;
  PetscViewer viewer;
  PetscBool flg = PETSC_FALSE;

  ierr = PetscTestFile(filename,'r',&flg);CHKERRQ(ierr); 
  if (!flg) PetscFunctionReturn(0);

  ierr = PetscViewerBinaryOpen(PetscObjectComm((PetscObject)ts),filename,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = _TSLoad(ts,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}                                                     




