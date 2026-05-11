#ifndef SEASQDDISCRETEGREENOPERATOR_20210907_H
#define SEASQDDISCRETEGREENOPERATOR_20210907_H

#include "common/HMatrixConfig.h"
#include "common/PetscVector.h"
#include "form/AbstractAdapterOperator.h"
#include "form/AbstractFrictionOperator.h"
#include "form/FacetFunctionalFactory.h"
#include "form/HMatrixGreenFunction.h"
#include "form/SeasQDOperator.h"
#include "mesh/LocalSimplexMesh.h"

#include <mpi.h>
#include <petscmat.h>
#include <petscvec.h>

#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace tndm {

class SeasQDDiscreteGreenOperator;

struct GreensFunctionIndices {
    PetscInt slip_block_size;
    PetscInt num_local_elements;
    PetscInt m_bs; // traction block size
    PetscInt n_bs; // always 1
    PetscInt m;
    PetscInt n;
    PetscInt mb_offset;
    PetscInt nb_offset;
    int rank;
    MPI_Comm comm;

    GreensFunctionIndices(SeasQDDiscreteGreenOperator const& op);
};

class SeasQDDiscreteGreenOperator : public SeasQDOperator {
public:
    using base = SeasQDOperator;

    SeasQDDiscreteGreenOperator(std::unique_ptr<typename base::dg_t> dgop,
                                std::unique_ptr<AbstractAdapterOperator> adapter,
                                std::unique_ptr<AbstractFrictionOperator> friction,
                                LocalSimplexMesh<DomainDimension> const& mesh,
                                std::optional<std::string> prefix, double gf_checkpoint_every_nmins,
                                bool matrix_free = false, MGConfig const& mg_config = MGConfig(),
                                HMatrixConfig const& hmatrix_config = HMatrixConfig());
    ~SeasQDDiscreteGreenOperator();

    void set_boundary(std::unique_ptr<AbstractFacetFunctionalFactory> fun) override;

    inline void initial_condition(BlockVector& state) {
        base::friction().pre_init(state);
        update_traction(0.0, state);
        base::friction().init(0.0, base::traction_, state);
    }

    inline void rhs(double time, BlockVector const& state, BlockVector& result) {
        update_traction(time, state);
        base::friction().rhs(time, base::traction_, state, result);
    }

    void update_internal_state(double time, BlockVector const& state,
                               bool state_changed_since_last_rhs, bool require_traction,
                               bool require_displacement) override;

protected:
    std::string gf_operator_filename_ = "gf_mat.bin";
    std::string gf_traction_filename_ = "gf_vec.bin";
    std::string gf_facet_filename_    = "gf_facet_labels.bin";
    double checkpoint_every_nmins_;

    void update_traction(double time, BlockVector const& state);

private:
    void compute_boundary_traction();
    // Returns local node coords: [N_el_local * nbf * DomainDimension], one per (element, node).
    std::vector<PetscReal> collect_node_coords() const;
    PetscInt create_discrete_greens_function();
    void partial_assemble_discrete_greens_function(LocalSimplexMesh<DomainDimension> const& mesh,
                                                   PetscInt current_gf_, PetscInt n_gf_);
    void write_discrete_greens_operator(LocalSimplexMesh<DomainDimension> const& mesh,
                                        PetscInt current_gf_, PetscInt n_gf_);
    PetscInt load_discrete_greens_operator(LocalSimplexMesh<DomainDimension> const& mesh,
                                           PetscInt n_gf_);
    void get_discrete_greens_function(LocalSimplexMesh<DomainDimension> const& mesh);
    void back_up_file(std::string file_to_backup);
    void write_discrete_greens_traction();
    void load_discrete_greens_traction();
    void get_boundary_traction();

    void write_facet_labels_IS(LocalSimplexMesh<DomainDimension> const& mesh);
    IS load_facet_labels_seq_IS(void);
    void create_permutation_redundant_IS(LocalSimplexMesh<DomainDimension> const& mesh, IS is);
    std::tuple<Mat, Mat> create_row_col_permutation_matrices(bool create_row, bool create_col);

    bool checkpoint_enabled_ = false;
    Mat  G_ = nullptr;
    std::unique_ptr<PetscVector> S_;
    std::unique_ptr<PetscVector> t_boundary_;
    bool repartition_gfs_ = false;
    IS   is_perm_ = nullptr;

    HMatrixConfig hmatrix_config_;

#ifdef PETSC_HAVE_HTOOL
    std::unique_ptr<HMatrixGreenFunction> hmat_op_;
#endif

public:
    Mat dense_gf() const { return G_; }

#ifdef PETSC_HAVE_HTOOL
    struct ValidationResult {
        double err_H_vs_G      = -1.0;
        double err_G_vs_solver = -1.0;
        double err_H_vs_solver = -1.0;
        double time_G_matvec   = -1.0;
        double time_H_matvec   = -1.0;
        double time_solver     = -1.0;
        int    n_matvec_reps   =  0;
        double mem_G_bytes     = -1.0;
        double mem_H_bytes     = -1.0;
        PetscInt global_rows   =  0;
        PetscInt global_cols   =  0;
        int      n_ranks       =  1;
    };
    ValidationResult validate_all();

    void export_h_structure(const std::string& prefix) const;
#endif
};

} // namespace tndm

#endif // SEASQDDISCRETEGREENOPERATOR_20210907_H
