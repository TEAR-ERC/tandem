#ifndef SEASQDDISCRETEGREENOPERATOR_20210907_H
#define SEASQDDISCRETEGREENOPERATOR_20210907_H

#include "common/PetscVector.h"
#include "form/AbstractAdapterOperator.h"
#include "form/AbstractFrictionOperator.h"
#include "form/FacetFunctionalFactory.h"
#include "form/SeasQDOperator.h"
#include "mesh/LocalSimplexMesh.h"

#include <mpi.h>
#include <petscmat.h>
#include <petscvec.h>

#include <iostream>
#include <memory>
#include <utility>

namespace tndm {

class SeasQDDiscreteGreenOperator : public SeasQDOperator {
public:
    using base = SeasQDOperator;

    SeasQDDiscreteGreenOperator(std::unique_ptr<typename base::dg_t> dgop,
                                std::unique_ptr<AbstractAdapterOperator> adapter,
                                std::unique_ptr<AbstractFrictionOperator> friction,
                                LocalSimplexMesh<DomainDimension> const& mesh,
                                std::optional<std::string> prefix, double gf_checkpoint_every_nmins,
                                bool matrix_free = false, MGConfig const& mg_config = MGConfig());
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
    std::string gf_facet_filename_ = "gf_facet_labels.bin";
    double checkpoint_every_nmins_;

    void update_traction(double time, BlockVector const& state);

private:
    void compute_discrete_greens_function();
    void compute_boundary_traction();
    PetscInt create_discrete_greens_function();
    void partial_assemble_discrete_greens_function(LocalSimplexMesh<DomainDimension> const& mesh,
                                                   PetscInt current_gf_, PetscInt n_gf_);
    void write_discrete_greens_operator(LocalSimplexMesh<DomainDimension> const& mesh,
                                        PetscInt current_gf_, PetscInt n_gf_);
    PetscInt load_discrete_greens_operator(LocalSimplexMesh<DomainDimension> const& mesh,
                                           PetscInt n_gf_);
    // all logic associated with matix craetion, loading / partial assembly is done here
    void get_discrete_greens_function(LocalSimplexMesh<DomainDimension> const& mesh);
    void write_discrete_greens_traction();
    void load_discrete_greens_traction();
    void get_boundary_traction();

    void write_facet_labels_IS(LocalSimplexMesh<DomainDimension> const& mesh);
    IS load_facet_labels_seq_IS(void);
    void create_permutation_redundant_IS(LocalSimplexMesh<DomainDimension> const& mesh, IS is);
    std::tuple<Mat, Mat> create_row_col_permutation_matrices(bool create_row, bool create_col);

    bool checkpoint_enabled_ = false;
    Mat G_ = nullptr;
    std::unique_ptr<PetscVector> S_;
    std::unique_ptr<PetscVector> t_boundary_;
    bool repartition_gfs_ = false;
    IS is_perm_ = nullptr;
};

} // namespace tndm

#endif // SEASQDDISCRETEGREENOPERATOR_20210907_H
