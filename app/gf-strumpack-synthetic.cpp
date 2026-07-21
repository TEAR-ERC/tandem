#include "common/CmdLine.h"
#include "common/MeshConfig.h"
#include "config.h"
#include "pc/register.h"
#include "tandem/SeasConfig.h"
#include "form/StrumpackGFFullOperator.h"

#include <argparse.hpp>
#include <mpi.h>
#include <petscsys.h>
#include <petscmat.h>
#include <petscvec.h>
#include <petscviewer.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <vector>
#include <cmath>

using namespace tndm;

namespace {

// Deterministic probe vectors.
//
// The entry value is a pure function of (seed, probe index, GLOBAL dof index), never of the
// local index or the owning rank. That makes a probe set bit-identical across MPI rank counts
// and across runs, so two configurations can be compared on exactly the same vectors and any
// change in the reported error is attributable to the compression alone.
uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

// Standard normal via Box-Muller. iid unit variance is what makes the mean over probes a
// consistent estimator of ||H-G||_F / ||G||_F (see report_probe_stats).
double probe_gaussian(uint64_t seed, int probe, PetscInt gidx) {
    const uint64_t h1 = splitmix64(seed ^ (0x9E3779B9ULL * static_cast<uint64_t>(probe + 1))
                                        ^ (static_cast<uint64_t>(gidx) * 0x2545F4914F6CDD1DULL));
    const uint64_t h2 = splitmix64(h1);
    // (0,1] uniforms; the +0.5 keeps u1 strictly positive so log() is finite.
    const double u1 = (static_cast<double>(h1 >> 11) + 0.5) * (1.0 / 9007199254740992.0);
    const double u2 = (static_cast<double>(h2 >> 11) + 0.5) * (1.0 / 9007199254740992.0);
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
}

struct ProbeStats {
    double mean{-1.0}, median{-1.0}, min{-1.0}, max{-1.0}, stddev{-1.0}, frobenius{-1.0};
};

ProbeStats summarize(std::vector<double> e, double sum_diff2, double sum_ref2) {
    ProbeStats s;
    if (e.empty()) return s;
    std::sort(e.begin(), e.end());
    const std::size_t n = e.size();
    s.min = e.front();
    s.max = e.back();
    s.median = (n % 2) ? e[n / 2] : 0.5 * (e[n / 2 - 1] + e[n / 2]);
    s.mean = std::accumulate(e.begin(), e.end(), 0.0) / static_cast<double>(n);
    double var = 0.0;
    for (double v : e) var += (v - s.mean) * (v - s.mean);
    s.stddev = (n > 1) ? std::sqrt(var / static_cast<double>(n - 1)) : 0.0;
    // For iid unit-variance probes, E||(H-G)v||^2 = ||H-G||_F^2 and E||Gv||^2 = ||G||_F^2,
    // so this ratio estimates ||H-G||_F / ||G||_F -- the aggregate, vector-independent number.
    s.frobenius = (sum_ref2 > 0.0) ? std::sqrt(sum_diff2 / sum_ref2) : -1.0;
    return s;
}

} // namespace

int main(int argc, char** argv) {
    int pArgc = 0;
    char** pArgv = nullptr;
    for (int i = 0; i < argc; ++i) {
        if (strcmp(argv[i], "--petsc") == 0) {
            pArgc = argc - i;
            pArgv = argv + i;
            argc = i;
            break;
        }
    }

    argparse::ArgumentParser program("gf-strumpack-synthetic");
    program.add_argument("--petsc").help("PETSc options, must be passed last!");
    program.add_argument("--mat").default_value(std::string("gf_mat.bin")).help("Path to gf_mat.bin");
    program.add_argument("--coords").default_value(std::string("coords.bin")).help("Path to coords.bin");
    program.add_argument("config").help("Configuration file (.toml) to read hmatrix settings");
    program.add_argument("--butterfly")
        .help("Also build and compare BUTTERFLY (slower, more MatVecs than HODLR)")
        .default_value(false)
        .implicit_value(true);
    program.add_argument("--probes")
        .help("Number of deterministic random probe vectors used for the accuracy report "
              "(default 10). Probes depend only on the global dof index, so they are "
              "identical across MPI rank counts and across runs.")
        .default_value(10)
        .scan<'i', int>();

    auto makePathRelativeToConfig =
        MakePathRelativeToOtherPath([&program]() { return program.get<std::string>("config"); });

    TableSchema<Config> schema;
    setConfigSchema(schema, makePathRelativeToConfig);

    std::optional<Config> cfg = readFromConfigurationFileAndCmdLine(schema, program, argc, argv);
    if (!cfg) return -1;

    CHKERRQ(PetscInitialize(&pArgc, &pArgv, nullptr, nullptr));
    CHKERRQ(register_PCs());
    CHKERRQ(register_KSPs());
    int exit_code = 0;
    {
    MPI_Comm comm = PETSC_COMM_WORLD;
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    std::string mat_path = program.get<std::string>("--mat");
    std::string coords_path = program.get<std::string>("--coords");

    // 1. MatLoad the dense G from gf_mat.bin (skipping the 2 tandem prefix ints)
    PetscViewer viewer;
    CHKERRQ(PetscViewerBinaryOpen(comm, mat_path.c_str(), FILE_MODE_READ, &viewer));
    
    // Skip the first two int32s in tandem prefix (ranks, ngfs)
    PetscInt commsize_checkpoint, current_gf;
    CHKERRQ(PetscViewerBinaryRead(viewer, &commsize_checkpoint, 1, NULL, PETSC_INT));
    CHKERRQ(PetscViewerBinaryRead(viewer, &current_gf, 1, NULL, PETSC_INT));

    Mat G_dense;
    CHKERRQ(MatCreate(comm, &G_dense));
    CHKERRQ(MatSetType(G_dense, MATDENSE));
    CHKERRQ(MatLoad(G_dense, viewer));
    CHKERRQ(PetscViewerDestroy(&viewer));

    PetscInt M_gf, N_gf;
    CHKERRQ(MatGetSize(G_dense, &M_gf, &N_gf));
    
    int D = DomainDimension;
    int slip_D = DomainDimension - 1;
    PetscInt Np = N_gf / slip_D;

    if (rank == 0) {
        std::cout << "Loaded G_dense: " << M_gf << " x " << N_gf << "\n";
        std::cout << "Np = " << Np << ", D = " << D << ", slip_D = " << slip_D << "\n";
    }

    // 2. Read coords.bin -> distributed local_coords
    std::vector<PetscReal> global_coords(Np * 3);
    PetscViewer coords_viewer;
    CHKERRQ(PetscViewerBinaryOpen(comm, coords_path.c_str(), FILE_MODE_READ, &coords_viewer));
    CHKERRQ(PetscViewerBinaryRead(coords_viewer, global_coords.data(), Np * 3, NULL, PETSC_DOUBLE));
    CHKERRQ(PetscViewerDestroy(&coords_viewer));

    PetscInt local_Np = Np / size + (rank < (Np % size) ? 1 : 0);
    PetscInt start_Np = rank * (Np / size) + std::min(static_cast<PetscInt>(rank), Np % size);
    std::vector<PetscReal> local_coords(local_Np * 3);
    std::copy(global_coords.begin() + start_Np * 3,
              global_coords.begin() + (start_Np + local_Np) * 3,
              local_coords.begin());

    // 3. Build slip/traction proto Vecs
    Vec s_proto, t_proto;
    CHKERRQ(VecCreateMPI(comm, PETSC_DECIDE, N_gf, &s_proto));
    CHKERRQ(VecCreateMPI(comm, PETSC_DECIDE, M_gf, &t_proto));



    // Benchmark the full-matrix operator (StrumpackGFFullOperator)
    if (rank == 0) {
        std::cout << "\n=== Benchmarking StrumpackGFFullOperator ===\n";
    }
    PetscLogDouble t0, t1;
    CHKERRQ(PetscTime(&t0));
    StrumpackGFFullOperator op_full(G_dense, local_coords, 1, D, s_proto, t_proto, comm, cfg->hmatrix_config);
    CHKERRQ(PetscTime(&t1));
    double build_time_full = t1 - t0;

    Vec x, y_G, y_S, diff;
    CHKERRQ(VecDuplicate(s_proto, &x));
    CHKERRQ(VecDuplicate(t_proto, &y_G));
    CHKERRQ(VecDuplicate(t_proto, &y_S));
    CHKERRQ(VecDuplicate(t_proto, &diff));

    // Populate x
    PetscInt local_size;
    CHKERRQ(VecGetLocalSize(x, &local_size));
    PetscInt start_idx;
    CHKERRQ(VecGetOwnershipRange(x, &start_idx, nullptr));
    
    PetscScalar* array;
    CHKERRQ(VecGetArray(x, &array));
    for (PetscInt i = 0; i < local_size; ++i) {
        array[i] = std::sin(static_cast<double>(start_idx + i));
    }
    CHKERRQ(VecRestoreArray(x, &array));

    // Time the dense matrix-vector product
    constexpr int N_REPS = 50;
    CHKERRQ(MatMult(G_dense, x, y_G)); // Warmup
    PetscLogDouble t_g0, t_g1;
    CHKERRQ(PetscTime(&t_g0));
    for (int r = 0; r < N_REPS; ++r) {
        CHKERRQ(MatMult(G_dense, x, y_G));
    }
    CHKERRQ(PetscTime(&t_g1));
    double dense_apply_time = (t_g1 - t_g0) / N_REPS;

    // Time the compressed matrix-vector product
    CHKERRQ(VecZeroEntries(y_S));
    op_full.apply(x, y_S); // Warmup
    PetscLogDouble t_s0, t_s1;
    CHKERRQ(PetscTime(&t_s0));
    for (int r = 0; r < N_REPS; ++r) {
        CHKERRQ(VecZeroEntries(y_S));
        op_full.apply(x, y_S);
    }
    CHKERRQ(PetscTime(&t_s1));
    double comp_apply_time = (t_s1 - t_s0) / N_REPS;

    // ---- Accuracy ----------------------------------------------------------------
    // A single fixed probe is too noisy to tune against: across the 64-rank BP5 sweep the
    // per-level construction errors were near-identical while this one number swung ~20x.
    // Measure on a deterministic probe SET instead and report the distribution.

    // Reusable single-probe measurement: err = ||H v - G v|| / ||G v||.
    // Returns PetscErrorCode (not the error value) so CHKERRQ's "return ierr" stays type-correct.
    auto measure = [&](Vec v, double* out_err, double* out_diff2, double* out_ref2) -> PetscErrorCode {
        CHKERRQ(MatMult(G_dense, v, y_G));
        CHKERRQ(VecZeroEntries(y_S));
        op_full.apply(v, y_S);
        CHKERRQ(VecCopy(y_S, diff));
        CHKERRQ(VecAXPY(diff, -1.0, y_G));
        PetscReal dn, rn;
        CHKERRQ(VecNorm(diff, NORM_2, &dn));
        CHKERRQ(VecNorm(y_G, NORM_2, &rn));
        if (out_diff2) *out_diff2 = static_cast<double>(dn) * static_cast<double>(dn);
        if (out_ref2)  *out_ref2  = static_cast<double>(rn) * static_cast<double>(rn);
        if (out_err)   *out_err   = (rn > 0.0) ? static_cast<double>(dn) / static_cast<double>(rn) : -1.0;
        return 0;
    };

    // (a) Legacy probe: x[i] = sin(i). x still holds it from the timing loop above. Kept so
    //     the "Relative Error" line below stays comparable with every previously logged run.
    double rel_err_full = -1.0;
    CHKERRQ(measure(x, &rel_err_full, nullptr, nullptr));

    // (b) Deterministic Gaussian probe set.
    constexpr uint64_t PROBE_SEED = 0x5EA5C0DEULL;
    const int n_probes = std::max(0, program.get<int>("--probes"));
    std::vector<double> probe_errs;
    double sum_diff2 = 0.0, sum_ref2 = 0.0;
    for (int p = 0; p < n_probes; ++p) {
        CHKERRQ(VecGetArray(x, &array));
        for (PetscInt i = 0; i < local_size; ++i)
            array[i] = probe_gaussian(PROBE_SEED, p, start_idx + i);
        CHKERRQ(VecRestoreArray(x, &array));
        double e = -1.0, d2 = 0.0, r2 = 0.0;
        CHKERRQ(measure(x, &e, &d2, &r2));
        probe_errs.push_back(e);
        sum_diff2 += d2;
        sum_ref2  += r2;
    }
    const ProbeStats stats = summarize(probe_errs, sum_diff2, sum_ref2);

    // (c) Smooth slip field -- the production regime. G is a smoothing operator, so a
    //     flat-spectrum probe is close to a worst case for relative error; a physically
    //     plausible slip distribution is the number that actually matters downstream.
    double rel_err_smooth = -1.0;
    {
        double xmin = 0.0, xmax = 0.0, zmin = 0.0, zmax = 0.0;
        for (PetscInt e = 0; e < Np; ++e) {
            const double cx = global_coords[e * 3 + 0], cz = global_coords[e * 3 + 2];
            if (e == 0) { xmin = xmax = cx; zmin = zmax = cz; }
            xmin = std::min(xmin, cx); xmax = std::max(xmax, cx);
            zmin = std::min(zmin, cz); zmax = std::max(zmax, cz);
        }
        const double mx = 0.5 * (xmin + xmax), mz = 0.5 * (zmin + zmax);
        const double sx = std::max(1e-30, 0.25 * (xmax - xmin));
        const double sz = std::max(1e-30, 0.25 * (zmax - zmin));
        CHKERRQ(VecGetArray(x, &array));
        for (PetscInt i = 0; i < local_size; ++i) {
            const PetscInt c = start_idx + i;          // global slip dof
            const PetscInt e = c / slip_D;             // element
            const PetscInt beta = c % slip_D;          // slip component
            const double dx = (global_coords[e * 3 + 0] - mx) / sx;
            const double dz = (global_coords[e * 3 + 2] - mz) / sz;
            const double bump = std::exp(-0.5 * (dx * dx + dz * dz));
            array[i] = bump * ((beta == 0) ? 1.0 : 0.5);   // strike-dominated smooth patch
        }
        CHKERRQ(VecRestoreArray(x, &array));
        CHKERRQ(measure(x, &rel_err_smooth, nullptr, nullptr));
    }

    // Worst number anywhere -- what the guard rail below judges.
    double worst_err = std::max(rel_err_full, std::max(stats.max, rel_err_smooth));

    const std::size_t total_applies =
        op_full.total_matvec_count() + op_full.total_adjoint_count();

    double mem_G = static_cast<double>(M_gf) * static_cast<double>(N_gf) * sizeof(PetscScalar);

    if (rank == 0) {
        std::cout << "\n==================================================\n"
                  << "  BENCHMARK RESULTS: StrumpackGFFullOperator\n"
                  << "==================================================\n"
                  << "  Matrix Dimensions: " << M_gf << " x " << N_gf << "\n"
                  << "  Build Time: " << build_time_full << " s\n"
                  << "  Dense Apply Time (avg): " << dense_apply_time << " s\n"
                  << "  Compressed Apply Time (avg): " << comp_apply_time << " s\n"
                  << "  Speedup: " << dense_apply_time / comp_apply_time << "x\n"
                  // Construction cost is forward AND adjoint applies. Elastostatics is
                  // self-adjoint, so in the matrix-free build a G^T*w costs the same
                  // elastostatic solve as a G*v -- both must be counted. The baseline
                  // N_assemble = slip_D*Np = N_gf is the number of unit-slip solves to
                  // assemble G column-by-column, so numerator and denominator are both
                  // in units of solves. (Reporting the forward count alone understates
                  // the true cost by ~2x; SEAS.cpp:633 already sums them.)
                  << "  Operator Applies (forward Gv): " << op_full.total_matvec_count() << "\n"
                  << "  Operator Applies (adjoint G^Tw): " << op_full.total_adjoint_count() << "\n"
                  << "  Total Operator Applies: " << total_applies << "\n"
                  << "  Ratio Applies to N_assemble (2*Np): "
                  << static_cast<double>(total_applies) / N_gf << "\n"
                  << "  Relative Error (||Hv - Gv||/||Gv||): " << rel_err_full << "\n"
                  << "  Dense Memory Size: " << mem_G / (1024.0 * 1024.0) << " MB\n"
                  << "  Compressed Memory Size: " << op_full.total_mem_bytes() / (1024.0 * 1024.0) << " MB\n"
                  << "  Compression Ratio: " << op_full.total_mem_bytes() / mem_G << "x\n"
                  << "--------------------------------------------------\n"
                  << "  Probe Count: " << n_probes << "\n"
                  << "  Probe Error Mean: "      << stats.mean      << "\n"
                  << "  Probe Error Median: "    << stats.median    << "\n"
                  << "  Probe Error Max: "       << stats.max       << "\n"
                  << "  Probe Error Min: "       << stats.min       << "\n"
                  << "  Probe Error Stddev: "    << stats.stddev    << "\n"
                  << "  Probe Error Frobenius: " << stats.frobenius << "\n"
                  << "  Smooth Slip Error: "     << rel_err_smooth  << "\n"
                  << "  Legacy sin(i) Error: "   << rel_err_full    << "\n"
                  << "==================================================\n";

        // Guard rail: a valid low-rank approximation cannot have relative error >= 1 --
        // even H == 0 gives exactly 1. Anything above that is a structural failure of the
        // construction (singular or misaligned diagonal blocks), not a coarse tolerance,
        // and must never be silently tabulated as a data point.
        if (!(worst_err < 1.0)) {
            std::cout << "\n*** ERROR: relative error " << worst_err
                      << " >= 1.0 — the compressed operator is structurally invalid.\n"
                      << "*** This is a construction failure, not an accuracy setting.\n"
                      << "*** Check the cluster tree report above (levels / all-padding leaves).\n"
                      << std::flush;
        }
    }
    // Non-zero exit so a sweep driver cannot silently tabulate a structurally invalid run.
    if (!(worst_err < 1.0)) exit_code = 1;

    CHKERRQ(VecDestroy(&x));
    CHKERRQ(VecDestroy(&y_G));
    CHKERRQ(VecDestroy(&y_S));
    CHKERRQ(VecDestroy(&diff));

    CHKERRQ(VecDestroy(&s_proto));
    CHKERRQ(VecDestroy(&t_proto));
    CHKERRQ(MatDestroy(&G_dense));
    }

    PetscFinalize();
    return exit_code;
}
