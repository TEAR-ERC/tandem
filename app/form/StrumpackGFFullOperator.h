#ifndef STRUMPACK_GF_FULL_OPERATOR_20260514_H
#define STRUMPACK_GF_FULL_OPERATOR_20260514_H

#include "common/HMatrixConfig.h"

#include <mpi.h>
#include <petscmat.h>
#include <petscvec.h>

#include <misc/MPIWrapper.hpp>
#include <structured/ClusterTree.hpp>
#include <structured/StructuredMatrix.hpp>
#include <structured/StructuredOptions.hpp>

#include <atomic>
#include <cstddef>
#include <memory>
#include <vector>

namespace tndm {

// Compresses the full GF (D*Np x slip_D*Np) as ONE STRUMPACK structured matrix.
// This is the sole GF compression operator: the earlier per-component variant
// (StrumpackGFOperator, one HODLR per (alpha,beta) block) was removed because the
// full operator shares probes across all components — one solve per probe instead
// of one per component per probe, the economics we want once probes are PDE solves.
//
// HODLR needs a square matrix, so the slip space is padded from slip_D*Np to
// D*Np. The padding is *node-interleaved*, giving rows and columns an identical
// layout:
//   permuted row  pr = new_i * D + alpha,  alpha in [0, D)      — all real tractions
//   permuted col  pc = new_j * D + beta,   beta  in [0, slip_D) — real slip
//                                          beta  == slip_D      — fictitious slip
// where perm_[new_i] = old_i (scalar node index).
//
// Two properties matter and both are consequences of interleaving:
//   1. Row index i and column index i refer to the SAME node, so the matrix
//      diagonal is the physical near-field self-interaction and the single
//      cluster tree is genuinely valid for both index spaces.
//   2. The fictitious columns are spread one-per-node instead of forming a
//      contiguous (D-slip_D)*Np band, so no cluster-tree leaf can ever be
//      entirely zero — which would make that diagonal block exactly singular
//      and blow up STRUMPACK's randomized construction.
// The fictitious columns additionally carry a scaled unit diagonal (pad_scale_)
// so each is nonzero on its own; apply() feeds them zeros, so their content
// never reaches the output.
//
// No component splitting: one MatMult call against G_dense covers all stress
// and slip directions simultaneously.
class StrumpackGFFullOperator {
public:
    // G_dense    : fully assembled GF (M x N, Tandem DOF order); not freed by this class
    // local_coords: [N_el_local * nbf * sdim] physical coords, one entry per (element, node)
    // nbf         : basis functions per element
    // D           : number of traction components per node (= DomainDimension)
    // s_proto     : slip Vec in Tandem order — used only to size scatter infrastructure
    // t_proto     : traction Vec in Tandem order — used only to size scatter infrastructure
    StrumpackGFFullOperator(Mat G_dense,
                            const std::vector<PetscReal>& local_coords,
                            PetscInt nbf, int D,
                            Vec s_proto, Vec t_proto,
                            MPI_Comm comm,
                            HMatrixConfig const& config);
    ~StrumpackGFFullOperator();

    // t += G_perm * s   (s and t in original Tandem DOF order; caller zeros t first)
    void apply(Vec s, Vec t) const;

    std::size_t total_matvec_count()  const { return matvec_count_; }
    std::size_t total_adjoint_count() const { return adjoint_count_; }
    double      total_mem_bytes()     const { return mem_bytes_; }

private:
    int      D_, slip_D_;
    PetscInt nbf_, N_el_;
    MPI_Comm comm_;
    HMatrixConfig config_;

    // Owning wrapper around a duplicate of comm_, kept for the lifetime of S_full_.
    // HODLRMatrix does NOT copy the MPIComm it is constructed with — it stores a raw
    // non-owning `const MPIComm* c_` (HODLRMatrix.hpp:575, assigned at HODLRMatrix.cpp:257)
    // and dereferences it in compress(), mult() and memory() (`if (c_->is_null()) return;`).
    // MPIComm also owns its communicator: it MPI_Comm_dup()s on construction and
    // MPI_Comm_free()s on destruction. Passing a temporary here left c_ dangling for the
    // whole life of the matrix and freed the communicator out from under it; whether the
    // abandoned storage still read MPI_COMM_NULL then decided, per run, whether compress()
    // silently did nothing and apply() later crashed. Declared BEFORE S_full_ so it is
    // destroyed AFTER it (members are destroyed in reverse declaration order).
    strumpack::MPIComm mpi_comm_;

    std::unique_ptr<strumpack::structured::StructuredMatrix<double>> S_full_;
    double mem_bytes_{0.0};

    // perm_[new_i] = old_i  — scalar permutation over N_el*nbf nodes
    std::vector<PetscInt> perm_;

    // row_perm_to_tandem_[pr] = Tandem row r for permuted row index pr (length D*Np)
    // col_perm_to_tandem_[pc] = Tandem col c for permuted col index pc (length D*Np,
    //                           padded space); -1 marks a fictitious (padding) column.
    std::vector<PetscInt> row_perm_to_tandem_;
    std::vector<PetscInt> col_perm_to_tandem_;

    // Diagonal value carried by each fictitious column, set to the RMS 2-norm of a
    // real column of G so the padding is neither negligible nor dominant.
    double pad_scale_{1.0};

    // Spatial median-split tree over the Np-node cloud, sizes scaled to D DOFs per
    // node. Non-null only when config_.cluster_tree == "kdtree"; it induces perm_.
    std::unique_ptr<strumpack::structured::ClusterTree> kd_tree_;

    // PETSc PETSC_DECIDE cumulative distributions (length n_ranks+1):
    //   petsc_dist_row_: for D*Np      (G_dense rows  / traction space)
    //   petsc_dist_col_: for slip_D*Np (G_dense cols  / slip     space)
    std::vector<int> petsc_dist_row_;
    std::vector<int> petsc_dist_col_;

    // Work Vecs for MatMult inside the construction callback
    Vec col_work_{nullptr};   // global size slip_D*Np, local = petsc_dist_col_ slice
    Vec row_work_{nullptr};   // global size D*Np,      local = petsc_dist_row_ slice

    // Scatter infrastructure for apply()
    Vec        s_perm_work_{nullptr};   // permuted col order, global slip_D*Np
    Vec        t_perm_work_{nullptr};   // permuted row order, global D*Np
    VecScatter scatter_s_{nullptr};     // s  (Tandem) → s_perm_work_
    VecScatter scatter_t_{nullptr};     // t_perm_work_ → t (Tandem, ADD)

    mutable std::atomic<std::size_t> matvec_count_{0};
    mutable std::atomic<std::size_t> adjoint_count_{0};

    void build_spatial_permutation(const std::vector<PetscReal>& global_coords);
    void build_index_maps();
    void build_s_full(Mat G_dense);
    void build_scatters(Vec s_proto, Vec t_proto);

    // True if permuted column index pc is a fictitious (padding) column. Rows and
    // columns share a layout, so this also identifies the row a fictitious column's
    // diagonal entry lands in.
    bool is_pad_index(PetscInt pc) const { return (pc % D_) == slip_D_; }

    static strumpack::structured::ClusterTree
    build_petsc_tree(const std::vector<int>& dist, int lo, int hi);

    // Spatial median-split tree that INDUCES perm_. Each
    // ClusterTree node reports n_nodes*D DOFs; recursion stops at leaf_size *nodes*,
    // so depth is set by leaf_size and NOT by the MPI rank count.
    strumpack::structured::ClusterTree
    build_kdtree(std::vector<PetscInt>& indices,
                 const std::vector<double>& proj, int eff_dim,
                 int leaf_size, PetscInt& fill);
};

} // namespace tndm
#endif // STRUMPACK_GF_FULL_OPERATOR_20260514_H
