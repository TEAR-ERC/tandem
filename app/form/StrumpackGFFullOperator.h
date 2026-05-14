#ifndef STRUMPACK_GF_FULL_OPERATOR_20260514_H
#define STRUMPACK_GF_FULL_OPERATOR_20260514_H

#include "common/HMatrixConfig.h"

#include <mpi.h>
#include <petscmat.h>
#include <petscvec.h>

#include <structured/ClusterTree.hpp>
#include <structured/StructuredMatrix.hpp>
#include <structured/StructuredOptions.hpp>

#include <atomic>
#include <cstddef>
#include <memory>
#include <vector>

namespace tndm {

// Compresses the full GF (D*Np x slip_D*Np) as ONE STRUMPACK structured matrix
// in node-interleaved spatial (Hilbert-permuted) DOF order:
//   permuted row  pr = new_i * D     + alpha
//   permuted col  pc = new_j * slip_D + beta
// where perm_[new_i] = old_i (scalar node index).
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

    std::unique_ptr<strumpack::structured::StructuredMatrix<double>> S_full_;
    double mem_bytes_{0.0};

    // perm_[new_i] = old_i  — scalar permutation over N_el*nbf nodes
    std::vector<PetscInt> perm_;

    // row_perm_to_tandem_[pr] = Tandem row r  for permuted row index pr  (length D*Np)
    // col_perm_to_tandem_[pc] = Tandem col c  for permuted col index pc  (length slip_D*Np)
    std::vector<PetscInt> row_perm_to_tandem_;
    std::vector<PetscInt> col_perm_to_tandem_;

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

    static strumpack::structured::ClusterTree
    build_petsc_tree(const std::vector<int>& dist, int lo, int hi);
};

} // namespace tndm
#endif // STRUMPACK_GF_FULL_OPERATOR_20260514_H
