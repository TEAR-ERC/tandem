#ifndef STRUMPACK_GF_OPERATOR_20260513_H
#define STRUMPACK_GF_OPERATOR_20260513_H

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

// Owns D*(D-1) STRUMPACK BUTTERFLY structured matrices G_αβ extracted from the
// full dense GF matrix, along with all permutation and scatter infrastructure
// needed for apply().  Mirror of HMatrixGreenFunction but uses matrix-free
// construction: only MatMult/MatMultTranspose callbacks are issued against
// G_perm_ab during construction.  No entry-wise kernel access.
class StrumpackGFOperator {
public:
    // G_dense    : fully assembled GF (M×N, original Tandem DOF order); not freed by this class
    // local_coords: [N_el_local * nbf * sdim] physical coords, one entry per (element, node)
    // nbf         : basis functions per element
    // D           : number of traction components per node (= DomainDimension)
    // s_proto     : slip Vec in Tandem order — used only to size column scatters
    // t_proto     : traction Vec in Tandem order — used only to size row scatters
    StrumpackGFOperator(Mat G_dense,
                        const std::vector<PetscReal>& local_coords,
                        PetscInt nbf, int D,
                        Vec s_proto, Vec t_proto,
                        MPI_Comm comm,
                        HMatrixConfig const& config,
                        strumpack::structured::Type format =
                            strumpack::structured::Type::BUTTERFLY);
    ~StrumpackGFOperator();

    // t += Σ_{α,β} S_αβ * s_β   (s and t in original Tandem DOF order)
    // Caller must zero t before calling apply.
    void apply(Vec s, Vec t) const;

    // Total random vectors used across all sub-matrix constructions.
    // Each forward vector = one PDE solve in the eventual matrix-free build.
    std::size_t total_matvec_count()  const { return matvec_count_; }
    std::size_t total_adjoint_count() const { return adjoint_count_; }

    // Sum of memory stored across all S_αβ matrices (bytes).
    double total_mem_bytes() const;

private:
    int      D_, slip_D_;
    PetscInt nbf_, N_el_;
    MPI_Comm comm_;
    HMatrixConfig config_;
    strumpack::structured::Type format_;

    // S_ab_[α * slip_D_ + β] : STRUMPACK StructuredMatrix, (N_el*nbf) × (N_el*nbf), spatially
    // permuted. nullptr when planar_fault=true and α=0.
    std::vector<std::unique_ptr<
        strumpack::structured::StructuredMatrix<double>>> S_ab_;
    std::vector<double> mem_bytes_ab_;

    // perm_[new_i] = old_i  — shared scalar permutation over N_el*nbf nodes
    std::vector<PetscInt> perm_;

    // PETSc PETSC_DECIDE row/col distribution: petsc_dist_[r] = first row on rank r.
    // Precomputed from Np and n_ranks; used for scatter infrastructure and apply().
    std::vector<int> petsc_dist_;

    // Per-component work vectors in spatial (permuted) order
    std::vector<Vec> slip_spatial_;      // [slip_D_] vecs, global size N_el*nbf
    std::vector<Vec> traction_spatial_;  // [D_]      vecs, global size N_el*nbf

    // scatter_s_[β]  : s β-component slice (original) → slip_spatial_[β]
    // scatter_t_[α]  : traction_spatial_[α] → t α-component slice (original)
    std::vector<VecScatter> scatter_s_;
    std::vector<VecScatter> scatter_t_;

    // Counters accumulate += nvec per callback invocation (vectors, not calls).
    mutable std::atomic<std::size_t> matvec_count_{0};
    mutable std::atomic<std::size_t> adjoint_count_{0};

    void build_spatial_permutation(const std::vector<PetscReal>& global_coords);
    Mat  extract_component_submatrix(Mat G_dense, int alpha, int beta) const;
    Mat  build_spatially_permuted_submatrix(Mat G_ab) const;
    void build_one_s_matrix(int alpha, int beta, Mat G_perm_ab);
    void build_scatters(Vec s_proto, Vec t_proto);

    // Build a binary ClusterTree over rank interval [lo, hi) from cumulative
    // dist[] array (PETSc PETSC_DECIDE layout).  Guarantees that rdist/cdist
    // in the mult_1d_t callback match PETSc's row/col ownership ranges.
    static strumpack::structured::ClusterTree
    build_petsc_tree(const std::vector<int>& dist, int lo, int hi);
};

} // namespace tndm

#endif // STRUMPACK_GF_OPERATOR_20260513_H
