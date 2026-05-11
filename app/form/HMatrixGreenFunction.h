#ifndef HMATRIXGREENFUNCTION_20260511_H
#define HMATRIXGREENFUNCTION_20260511_H

// petscsys.h pulls in petscconf.h which defines PETSC_HAVE_HTOOL.
// Must come before the #ifdef check.
#include <petscsys.h>

#ifdef PETSC_HAVE_HTOOL

#include "common/HMatrixConfig.h"

#include <mpi.h>
#include <petscmat.h>
#include <petscvec.h>

#include <string>
#include <vector>

namespace tndm {

// Owns D*(D-1) HTool H-matrices G_αβ extracted from the full dense GF matrix,
// along with all permutation and scatter infrastructure needed for apply().
// All knowledge of HTool is contained here; SeasQDDiscreteGreenOperator is agnostic.
class HMatrixGreenFunction {
public:
    // G_dense     : fully assembled GF (M×N, original Tandem DOF order); not freed by this class
    // local_coords: [N_el_local * nbf * sdim] physical coords, one entry per (element, node)
    // nbf         : basis functions per element
    // D           : number of traction components per node (= DomainDimension)
    // s_proto     : slip Vec in Tandem order — used only to size column scatters
    // t_proto     : traction Vec in Tandem order — used only to size row scatters
    HMatrixGreenFunction(Mat G_dense,
                         const std::vector<PetscReal>& local_coords,
                         PetscInt nbf, int D,
                         Vec s_proto, Vec t_proto,
                         MPI_Comm comm,
                         HMatrixConfig const& config);
    ~HMatrixGreenFunction();

    // t += Σ_{α,β} H_αβ * s_β   (s and t in original Tandem DOF order)
    // Caller must zero t before calling apply.
    void apply(Vec s, Vec t) const;

    // Sum of memory stored across all H_αβ matrices (bytes).
    double total_mem_bytes() const;

    // Write leaf CSV for each (α,β) sub-matrix.
    // <prefix>_ab<α><β>.csv        — global merged CSV (all ranks)
    // <prefix>_ab<α><β>_rank<r>.csv — per-rank CSV (local offsets, HTool native)
    void export_structure(const std::string& prefix) const;

private:
    int      D_;
    int      slip_D_;   // D - 1
    PetscInt nbf_;
    PetscInt N_el_;     // global element count
    MPI_Comm comm_;
    HMatrixConfig config_;

    // H_ab_[α * slip_D_ + β] : HTool Mat, (N_el*nbf) × (N_el*nbf), spatially permuted order
    std::vector<Mat>    H_ab_;
    std::vector<double> mem_bytes_ab_;

    // perm_[new_i] = old_i   — shared scalar permutation over N_el*nbf nodes
    std::vector<PetscInt> perm_;

    // Per-component work vectors in spatial (permuted) order
    std::vector<Vec> slip_spatial_;      // [slip_D_] vecs, global size N_el*nbf
    std::vector<Vec> traction_spatial_;  // [D_]      vecs, global size N_el*nbf

    // scatter_s_[β]  : s β-component slice (original) → slip_spatial_[β]
    // scatter_t_[α]  : traction_spatial_[α] → t α-component slice (original)
    std::vector<VecScatter> scatter_s_;
    std::vector<VecScatter> scatter_t_;

    void build_spatial_permutation(const std::vector<PetscReal>& global_coords);
    Mat  extract_component_submatrix(Mat G_dense, int alpha, int beta) const;
    Mat  build_spatially_permuted_submatrix(Mat G_ab) const;
    void build_one_h_matrix(int alpha, int beta, Mat G_perm_ab,
                            const std::vector<PetscReal>& global_coords);
    void build_scatters(Vec s_proto, Vec t_proto);

#ifdef HAVE_PETSC_HTOOL_PRIVATE
    void export_one_h_structure(Mat H, const std::string& subprefix) const;
#endif
};

} // namespace tndm

#endif // PETSC_HAVE_HTOOL
#endif // HMATRIXGREENFUNCTION_20260511_H
