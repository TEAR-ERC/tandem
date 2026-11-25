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

#include <functional>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

namespace tndm {

class SeasQDDiscreteGreenOperator;

/**
 * @brief Computes the traction response to a unit slip on its own (sub-)communicator.
 *
 * Used to compute several Green's functions concurrently: the ranks are split into groups, each
 * group owns a complete copy of the discretisation on its sub-communicator and works on
 * different Green's functions. All methods are collective on comm().
 */
class GfSolver : public SeasQDOperator {
public:
    GfSolver(std::unique_ptr<dg_t> dgop, std::unique_ptr<AbstractAdapterOperator> adapter,
             std::unique_ptr<AbstractFrictionOperator> friction, bool matrix_free = false,
             MGConfig const& mg_config = MGConfig());

    /// Solves for unit slip at the given global index of the slip vector (in the layout of this
    /// solver, i.e. numbered by the fault elements of this communicator).
    /// The result is available in traction().
    void unit_slip_response(PetscInt global_slip_index);

    PetscVector& traction() { return traction_; }

private:
    Scatter scatter_;
    SparseBlockVector<double> ghost_;
    PetscVector slip_;
};

/// Everything a group needs to compute Green's functions (mesh, context, solver). Created by a
/// factory, as the construction depends on the configuration and on the problem type.
class GfWorker {
public:
    virtual ~GfWorker() = default;
    virtual GfSolver& solver() = 0;
    /// Vertex indices of the fault facets of the solver, in the order of the local fault elements.
    virtual std::vector<Simplex<DomainDimension - 1u>> const& fault_facets() const = 0;
};

/// Creates a worker whose mesh and solver are distributed over the given communicator.
using GfWorkerFactory = std::function<std::unique_ptr<GfWorker>(MPI_Comm)>;

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
                                GfWorkerFactory worker_factory = nullptr, unsigned gf_groups = 1);
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
    void compute_boundary_traction();
    PetscInt create_discrete_greens_function();
    void partial_assemble_discrete_greens_function(LocalSimplexMesh<DomainDimension> const& mesh,
                                                   PetscInt current_gf_, PetscInt n_gf_);
    // Same as above, but the ranks are split into groups that solve different Green's functions
    // concurrently (see gf_groups).
    void
    partial_assemble_discrete_greens_function_grouped(LocalSimplexMesh<DomainDimension> const& mesh,
                                                      PetscInt current_gf_, PetscInt n_gf_);
    void write_discrete_greens_operator(LocalSimplexMesh<DomainDimension> const& mesh,
                                        PetscInt current_gf_, PetscInt n_gf_);
    PetscInt load_discrete_greens_operator(LocalSimplexMesh<DomainDimension> const& mesh,
                                           PetscInt n_gf_);
    // all logic associated with matix craetion, loading / partial assembly is done here
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
    Mat G_ = nullptr;
    std::unique_ptr<PetscVector> S_;
    std::unique_ptr<PetscVector> t_boundary_;
    bool repartition_gfs_ = false;
    IS is_perm_ = nullptr;
    GfWorkerFactory worker_factory_;
    unsigned gf_groups_ = 1;
};

} // namespace tndm

#endif // SEASQDDISCRETEGREENOPERATOR_20210907_H
