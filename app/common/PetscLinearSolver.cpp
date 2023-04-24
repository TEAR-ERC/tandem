#include "PetscLinearSolver.h"
#include "common/PetscUtil.h"
#include <petscpc.h>

namespace tndm {

PetscLinearSolver::PetscLinearSolver(AbstractDGOperator<DomainDimension>& dgop, bool matrix_free,
                                     MGConfig const& mg_config) {
    auto const& topo = dgop.topo();
    //auto stype = DGOpSparsityType::BLOCK_DIAGONAL;
    auto stype = DGOpSparsityType::DIAGONAL;

    if (matrix_free) {
        A_ = std::make_unique<PetscDGShell>(dgop);
        P_ = std::make_unique<PetscDGMatrix>(dgop.block_size(), topo, stype);
        dgop.assemble(*P_);
    } else {
        P_ = std::make_unique<PetscDGMatrix>(dgop.block_size(), topo, DGOpSparsityType::FULL);
        dgop.assemble(*P_);
    }

    b_ = std::make_unique<PetscVector>(dgop.block_size(), topo.numLocalElements(), topo.comm());
    x_ = std::make_unique<PetscVector>(*b_);
    dgop.rhs(*b_);

    CHKERRTHROW(KSPCreate(topo.comm(), &ksp_));
    CHKERRTHROW(KSPSetType(ksp_, KSPCG));
    if (matrix_free) {
        CHKERRTHROW(KSPSetOperators(ksp_, A_->mat(), P_->mat() ));
        //-testing-//CHKERRTHROW(KSPSetOperators(ksp_, P_->mat(), P_->mat() ));
    } else {
        CHKERRTHROW(KSPSetOperators(ksp_, P_->mat(), P_->mat() ));
    }
    CHKERRTHROW(KSPSetTolerances(ksp_, 1.0e-12, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));

    PC pc;
    CHKERRTHROW(KSPGetPC(ksp_, &pc));
    CHKERRTHROW(PCSetFromOptions(pc));
    PCType type;
    PCGetType(pc, &type);
    switch (fnv1a(type)) {
        case HASH_DEF(PCMG):
            if (!matrix_free) {
                setup_mg_default_asm(dgop, pc, mg_config);
            } else {
                setup_mg_default_mf(dgop, pc, mg_config);
            }
          break;
        default:
          break;
    };

    CHKERRTHROW(KSPSetFromOptions(ksp_));
}

PetscLinearSolver::~PetscLinearSolver() {
    for (auto&& A : mat_cleanup) {
        MatDestroy(&A);
    }
    KSPDestroy(&ksp_);
}

void PetscLinearSolver::setup_mg_default_asm(AbstractDGOperator<DomainDimension>& dgop, PC pc,
                                 MGConfig const& mg_config) {
    auto i_op = dgop.interpolation_operator();
    auto level_degree = mg_config.levels(i_op->max_degree());
    unsigned nlevels = level_degree.size();

    int rank;
    MPI_Comm_rank(dgop.topo().comm(), &rank);
    if (rank == 0) {
        std::cout << "Multigrid P-levels: ";
        for (auto level : level_degree) {
            std::cout << level << " ";
        }
        std::cout << std::endl;
    }

    CHKERRTHROW(PCMGSetLevels(pc, nlevels, nullptr));
    CHKERRTHROW(PCMGSetGalerkin(pc, PC_MG_GALERKIN_NONE));
    KSP smooth;
    CHKERRTHROW(PCMGGetSmoother(pc, nlevels - 1, &smooth));
    CHKERRTHROW(KSPSetOperators(smooth, P_->mat(), P_->mat()));

    Mat interp[] = { NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL };
    for (int k=nlevels-2; k>=0; --k) {
        unsigned to_degree = level_degree[k+1];
        unsigned from_degree = level_degree[k];
        auto interp_k = PetscInterplMatrix(i_op->block_size(to_degree),
                                           i_op->block_size(from_degree),
                                           dgop.topo());

        i_op->assemble(to_degree, from_degree, interp_k);
        PetscObjectReference((PetscObject)interp_k.mat());
        interp[k+1] = interp_k.mat();
    }
    for (int k=nlevels-1; k>=0; k--) {
        if (interp[k]) {
            char pname[PETSC_MAX_PATH_LEN];

            PetscSNPrintf(pname,PETSC_MAX_PATH_LEN-1,"interp_%d%d",k,k-1);
            PetscObjectSetName((PetscObject)interp[k],pname);
        }
   }

    for (int l=nlevels-1; l>=0; --l) {
        unsigned degree = level_degree[l];

	PetscPrintf(dgop.topo().comm(),"level[%d]: degree %zu\n",l,degree);
    }

    PetscPrintf(dgop.topo().comm(),"Interpolation matrices:\n");
    for (int k=nlevels-1; k>=1; k--) {
        unsigned to_degree = level_degree[k];
        unsigned from_degree = level_degree[k-1];

        if (interp[k]) {
            PetscPrintf(dgop.topo().comm(),"Level[%d]:\n",k);
            PetscPrintf(dgop.topo().comm(),"  degree %zu --> degree %zu\n",to_degree, from_degree);
            PetscViewerASCIIPushTab(PETSC_VIEWER_STDOUT_WORLD);
            PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO);
            MatView(interp[k],PETSC_VIEWER_STDOUT_WORLD);
            PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD);
            PetscViewerASCIIPopTab(PETSC_VIEWER_STDOUT_WORLD);
        }
    }

    Mat A_lp1 = P_->mat();
    for (int l = nlevels - 2; l >= 0; --l) {
        Mat interp_l = interp[l+1];
        Mat A_l;

	CHKERRTHROW(PCMGSetInterpolation(pc, l + 1, interp_l));

        CHKERRTHROW(MatPtAP(A_lp1, interp_l, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &A_l));

        CHKERRTHROW(PCMGGetSmoother(pc, l, &smooth));
        CHKERRTHROW(KSPSetOperators(smooth, A_l, A_l));
        mat_cleanup.push_back(A_l);
        A_lp1 = A_l;
    }

    for (int k=nlevels-1; k>=0; --k) {
        unsigned from_degree = level_degree[k];
        Mat Amat,Bmat;

        CHKERRTHROW(PCMGGetSmoother(pc,k,&smooth));
        CHKERRTHROW(KSPGetOperators(smooth,&Amat,&Bmat));

	PetscPrintf(dgop.topo().comm(),"level[%d]: degree %zu\n",k,from_degree);
        PetscPrintf(dgop.topo().comm(),"  Amat, Bmat:\n");
        PetscViewerASCIIPushTab(PETSC_VIEWER_STDOUT_WORLD);
        PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO);
        MatView(Amat,PETSC_VIEWER_STDOUT_WORLD);
        PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD);
        PetscViewerASCIIPopTab(PETSC_VIEWER_STDOUT_WORLD);
    }

    for (int k=nlevels-1; k>=0; k--) {
        if (interp[k]) {
            CHKERRTHROW(MatDestroy(&interp[k]));
        }
    }
}

void PetscLinearSolver::setup_mg_default_mf(AbstractDGOperator<DomainDimension>& dgop, PC pc,
                                 MGConfig const& mg_config) {
    auto i_op = dgop.interpolation_operator();

    auto level_degree = mg_config.levels(i_op->max_degree());
    unsigned nlevels = level_degree.size();

    int rank;
    MPI_Comm_rank(dgop.topo().comm(), &rank);
    if (rank == 0) {
        std::cout << "Multigrid P-levels: ";
        for (auto level : level_degree) {
            std::cout << level << " ";
        }
        std::cout << std::endl;
    }

    for (int l=nlevels-1; l>=0; --l) {
        unsigned degree = level_degree[l];

	PetscPrintf(dgop.topo().comm(),"level[%d]: degree %zu\n",l,degree);
    }

    CHKERRTHROW(PCMGSetLevels(pc, nlevels, nullptr));
    CHKERRTHROW(PCMGSetGalerkin(pc, PC_MG_GALERKIN_NONE));
    KSP smooth;
    CHKERRTHROW(PCMGGetSmoother(pc, nlevels - 1, &smooth));
    CHKERRTHROW(KSPSetOperators(smooth, A_->mat(), P_->mat()));

    Mat interp[] = { NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL };
    for (int k=nlevels-2; k>=0; --k) {
        unsigned to_degree = level_degree[k+1];
        unsigned from_degree = level_degree[k];
        auto interp_k = PetscInterplMatrix(i_op->block_size(to_degree),
		                        i_op->block_size(from_degree),
					dgop.topo());

        i_op->assemble(to_degree, from_degree, interp_k);
        PetscObjectReference((PetscObject)interp_k.mat());
        interp[k+1] = interp_k.mat();
    }

    for (int k=nlevels-1; k>=0; k--) {
        if (interp[k]) {
            char pname[PETSC_MAX_PATH_LEN];

            PetscSNPrintf(pname,PETSC_MAX_PATH_LEN-1,"interp_%d%d",k,k-1);
            PetscObjectSetName((PetscObject)interp[k],pname);
        }
    }

    PetscPrintf(dgop.topo().comm(),"Interpolation matrices:\n");
    for (int k=nlevels-1; k>=0; k--) {
        if (interp[k]) {
            unsigned to_degree = level_degree[k];
            unsigned from_degree = level_degree[k-1];

            PetscPrintf(dgop.topo().comm(),"Level[%d]:\n",k);
            PetscPrintf(dgop.topo().comm(),"  degree %zu -->> degree %zu\n",to_degree,from_degree);
            PetscViewerASCIIPushTab(PETSC_VIEWER_STDOUT_WORLD);
            PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO);
            MatView(interp[k],PETSC_VIEWER_STDOUT_WORLD);
            PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD);
            PetscViewerASCIIPopTab(PETSC_VIEWER_STDOUT_WORLD);
        }
    }

    Mat interp_cat[] = { NULL, NULL, NULL, NULL, NULL };
    //MatDuplicate(interp[nlevels-1],MAT_COPY_VALUES,&interp_cat[nlevels-1]);
    PetscObjectReference((PetscObject)interp[nlevels-1]);
    interp_cat[nlevels-1] = interp[nlevels-1];
    for (int k=nlevels-2; k>0; --k) {
        MatMatMult(interp_cat[k+1],interp[k],MAT_INITIAL_MATRIX,1.0,&interp_cat[k]);
    }

    for (int k=nlevels-1; k>=0; k--) {
        if (interp_cat[k]) {
            char pname[PETSC_MAX_PATH_LEN];

            PetscSNPrintf(pname,PETSC_MAX_PATH_LEN-1,"interp_c_%d%d",nlevels-1,k-1);
            PetscObjectSetName((PetscObject)interp_cat[k],pname);
        }
    }

    PetscPrintf(dgop.topo().comm(),"Interpolation matrices (concatenated):\n");
    for (int k=nlevels-1; k>=0; k--) {
        if (interp_cat[k]) {
            unsigned to_degree = level_degree[nlevels-1];
            unsigned from_degree = level_degree[k-1];

            PetscPrintf(dgop.topo().comm(),"Level[%d]:\n",k);
	    PetscPrintf(dgop.topo().comm(),"  degree %zu -->> degree %zu\n",to_degree,from_degree);

            PetscViewerASCIIPushTab(PETSC_VIEWER_STDOUT_WORLD);
            PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO);
            MatView(interp_cat[k],PETSC_VIEWER_STDOUT_WORLD);
            PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD);
            PetscViewerASCIIPopTab(PETSC_VIEWER_STDOUT_WORLD);
        }
    }

    for (int l = nlevels - 2; l >= 0; --l) {
        Mat interp_l = interp[l+1];

	CHKERRTHROW(PCMGSetInterpolation(pc, l + 1, interp_l));
    }

    for (int l = nlevels - 2; l >= 0; --l) {
        unsigned target = l;
        unsigned to_degree = level_degree[target];
        unsigned to_blocksize = i_op->block_size(to_degree);
        unsigned from_degree = level_degree[l+1];
	Mat asmA = NULL;

        CHKERRTHROW(PCMGGetSmoother(pc, l, &smooth));

#if 0
        auto Ap_ = std::make_unique<PetscDGShellPtAP>(dgop,interp_cat[target+1], to_blocksize);
        asmA = Ap_->create_explicit(dgop.topo(), DGOpSparsityType::FULL);
        CHKERRTHROW(KSPSetOperators(smooth, asmA, asmA));
        //CHKERRTHROW(KSPSetOperators(smooth, Ap_->mat(), asmA)); // out of scope
#endif

	//PetscDGShellPtAP *ctx = new PetscDGShellPtAP(dgop, interp_cat[target+1], to_blocksize);
	//asmA = ctx->create_explicit(dgop.topo(), false);
	//printf("ctx %p\n", ctx);
        //CHKERRTHROW(KSPSetOperators(smooth, ctx->mat(), asmA));
#if 1

        if (l == 0) {
            PetscDGShellPtAP *ctx = new PetscDGShellPtAP(dgop, interp_cat[target+1], to_blocksize);
	    asmA = ctx->create_explicit(dgop.topo(), DGOpSparsityType::FULL); // full operator
            CHKERRTHROW(KSPSetOperators(smooth, asmA, asmA));
            delete ctx;
        } else {
            PetscDGShellPtAP *ctx = new PetscDGShellPtAP(dgop, interp_cat[target+1], to_blocksize);

            //asmA = ctx->create_explicit(dgop.topo(), true); // block diag
            // restrict the fine level preconditioner
            CHKERRTHROW(MatPtAP(P_->mat(), interp_cat[target+1], MAT_INITIAL_MATRIX, PETSC_DEFAULT, &asmA));
            CHKERRTHROW(KSPSetOperators(smooth, ctx->mat(), asmA));
            Mat mat = ctx->mat(); MatDestroy(&mat); // pass control of delete to petsc
        }
#endif

        CHKERRTHROW(MatDestroy(&asmA));
    }

    for (int k=nlevels-1; k>=0; --k) {
        unsigned from_degree = level_degree[k];
        Mat Amat,Bmat;

	PetscPrintf(dgop.topo().comm(),"level[%d]: degree %zu\n",k,from_degree);
        CHKERRTHROW(PCMGGetSmoother(pc, k, &smooth));

        KSPGetOperators(smooth, &Amat, &Bmat);
        PetscViewerASCIIPushTab(PETSC_VIEWER_STDOUT_WORLD);
        PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO);
        PetscPrintf(dgop.topo().comm(),"  Amat:\n");
        MatView(Amat,PETSC_VIEWER_STDOUT_WORLD);
        PetscPrintf(dgop.topo().comm(),"  Bmat:\n");
        MatView(Bmat,PETSC_VIEWER_STDOUT_WORLD);
        PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD);
        PetscViewerASCIIPopTab(PETSC_VIEWER_STDOUT_WORLD);
    }

    for (int k=nlevels-1; k>=0; k--) {
        if (interp[k]) { CHKERRTHROW(MatDestroy(&interp[k])); }
        if (interp_cat[k]) { CHKERRTHROW(MatDestroy(&interp_cat[k])); }
    }
}

void PetscLinearSolver::setup_mg(AbstractDGOperator<DomainDimension>& dgop, PC pc,
                                 MGConfig const& mg_config) {
    auto i_op = dgop.interpolation_operator();

    auto level_degree = mg_config.levels(i_op->max_degree());
    unsigned nlevels = level_degree.size();

    int rank;
    MPI_Comm_rank(dgop.topo().comm(), &rank);
    if (rank == 0) {
        std::cout << "Multigrid P-levels: ";
        for (auto level : level_degree) {
            std::cout << level << " ";
        }
        std::cout << std::endl;
    }

    CHKERRTHROW(PCMGSetLevels(pc, nlevels, nullptr));
    CHKERRTHROW(PCMGSetGalerkin(pc, PC_MG_GALERKIN_NONE));
    KSP smooth;
    CHKERRTHROW(PCMGGetSmoother(pc, nlevels - 1, &smooth));
    if (A_) {
        CHKERRTHROW(KSPSetOperators(smooth, A_->mat(), P_->mat()));
    } else {
        CHKERRTHROW(KSPSetOperators(smooth, P_->mat(), P_->mat()));
    }

    Mat interp[] = { NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL };
    for (int k=nlevels-2; k>=0; --k) {
        unsigned to_degree = level_degree[k+1];
        unsigned from_degree = level_degree[k];
        auto interp_k = PetscInterplMatrix(i_op->block_size(to_degree),
                                           i_op->block_size(from_degree),
                                           dgop.topo());
      i_op->assemble(to_degree, from_degree, interp_k);
      PetscObjectReference((PetscObject)interp_k.mat());
      interp[k+1] = interp_k.mat();
      char pname[PETSC_MAX_PATH_LEN];
      PetscSNPrintf(pname,PETSC_MAX_PATH_LEN-1,"interp_%d",k+1);
      PetscObjectSetName((PetscObject)interp[k+1],pname);
   }
   for (int k=nlevels-1; k>=0; k--) {
      if (interp[k]) {
      PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO);
      MatView(interp[k],PETSC_VIEWER_STDOUT_WORLD);
      PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD);}
    }


    Mat interp_cat[] = { NULL, NULL, NULL, NULL, NULL };
    MatDuplicate(interp[nlevels-1],MAT_COPY_VALUES,&interp_cat[nlevels-1]);
    for (int k=nlevels-2; k>0; --k) {
      MatMatMult(interp_cat[k+1],interp[k],MAT_INITIAL_MATRIX,1.0,&interp_cat[k]);
    }
    for (int k=nlevels-1; k>=0; k--) {
      if (interp_cat[k]) {
      char pname[PETSC_MAX_PATH_LEN];
      PetscSNPrintf(pname,PETSC_MAX_PATH_LEN-1,"interp_c_%d",k);
      PetscObjectSetName((PetscObject)interp_cat[k],pname);
      PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO);
      MatView(interp_cat[k],PETSC_VIEWER_STDOUT_WORLD);
      PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD);}
    }



    Mat A_lp1 = P_->mat();
    for (int l = nlevels - 2; l >= 0; --l) {
        unsigned to_degree = level_degree[l + 1];
        unsigned from_degree = level_degree[l];
        //auto I = PetscInterplMatrix(i_op->block_size(to_degree), i_op->block_size(from_degree),
        //                            dgop.topo());
        //i_op->assemble(to_degree, from_degree, I);
	//Map interp_l = I.mat();
	PetscPrintf(dgop.topo().comm(),"%zu -->> %zu\n",to_degree, from_degree);
        Mat interp_l = interp[l+1];

	CHKERRTHROW(PCMGSetInterpolation(pc, l + 1, interp_l));

        Mat A_l;
        CHKERRTHROW(MatPtAP(A_lp1, interp_l, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &A_l));
        CHKERRTHROW(PCMGGetSmoother(pc, l, &smooth));
        CHKERRTHROW(KSPSetOperators(smooth, A_l, A_l));
        mat_cleanup.push_back(A_l);
        A_lp1 = A_l;
	//MatDestroy(&interp_l);
    }


    {
	int target = nlevels - 2;
	Vec xf_,yf_;
	MatCreateVecs(interp_cat[target+1],&xf_,NULL);
	VecDuplicate(xf_,&yf_);

        unsigned to_degree = level_degree[target];
        unsigned to_blocksize = i_op->block_size(to_degree);

        auto Ap_ = std::make_unique<PetscDGShellPtAP>(dgop,interp_cat[target+1], to_blocksize);

        //MatMult(Ap_->mat(), xf_, yf_);

        Mat asmA = Ap_->create_explicit(dgop.topo(), DGOpSparsityType::FULL);
	PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO);
	MatView(asmA,PETSC_VIEWER_STDOUT_WORLD);
	PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD);
        CHKERRTHROW(MatDestroy(&asmA));
    }

}

void PetscLinearSolver::warmup() { warmup_ksp(ksp_); }

void PetscLinearSolver::warmup_ksp(KSP ksp) {
    PC pc;
    CHKERRTHROW(KSPSetUp(ksp));
    CHKERRTHROW(KSPSetUpOnBlocks(ksp));
    CHKERRTHROW(KSPGetPC(ksp, &pc));
    warmup_sub_pcs(pc);
}

void PetscLinearSolver::warmup_sub_pcs(PC pc) {
    PCType type;
    PCGetType(pc, &type);
    switch (fnv1a(type)) {
    case HASH_DEF(PCCOMPOSITE):
        warmup_composite(pc);
        break;
    case HASH_DEF(PCMG):
        warmup_mg(pc);
        break;
    default:
        break;
    };
}

void PetscLinearSolver::warmup_composite(PC pc) {
    PetscInt nc;
    CHKERRTHROW(PCCompositeGetNumberPC(pc, &nc));
    for (PetscInt n = 0; n < nc; ++n) {
        PC sub;
        CHKERRTHROW(PCCompositeGetPC(pc, n, &sub));
        CHKERRTHROW(PCSetUp(sub));
        CHKERRTHROW(PCSetUpOnBlocks(sub));
        warmup_sub_pcs(sub);
    }
}

void PetscLinearSolver::warmup_mg(PC pc) {
    PetscInt levels;
    CHKERRTHROW(PCMGGetLevels(pc, &levels));
    for (PetscInt level = 0; level < levels; ++level) {
        KSP smoother;
        CHKERRTHROW(PCMGGetSmoother(pc, level, &smoother));
        warmup_ksp(smoother);
    }
}

void PetscLinearSolver::dump() const {
    PetscViewer viewer;
    CHKERRTHROW(PetscViewerCreate(PETSC_COMM_WORLD, &viewer));
    CHKERRTHROW(PetscViewerSetType(viewer, PETSCVIEWERBINARY));
    CHKERRTHROW(PetscViewerFileSetMode(viewer, FILE_MODE_WRITE));

    CHKERRTHROW(PetscViewerFileSetName(viewer, "A.bin"));
    CHKERRTHROW(MatView(P_->mat(), viewer));

    CHKERRTHROW(PetscViewerFileSetName(viewer, "b.bin"));
    CHKERRTHROW(VecView(b_->vec(), viewer));

    CHKERRTHROW(PetscViewerDestroy(&viewer));
}

} // namespace tndm
