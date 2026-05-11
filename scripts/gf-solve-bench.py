#!/usr/bin/env python3
"""
Empirical MPI scaling benchmark for the SEAS QD solve(v) step.

Runs gf-solve-bench with each requested MPI rank count, collects the
wall-clock solve timings reported on stdout, saves a CSV, and plots the
results.

Usage example:
    python scripts/gf-solve-bench.py \\
        --binary build/app/gf-solve-bench \\
        --config examples/my_problem.toml \\
        --ranks 1 2 4 8 16 32 \\
        --nreps 10 \\
        --output results/scaling.csv \\
        --plot   results/scaling.png \\
        --petsc "-ksp_type gmres -pc_type bjacobi"
"""

import argparse
import csv
import os
import subprocess
import sys
import time


def parse_args():
    p = argparse.ArgumentParser(
        description="MPI scaling benchmark for gf-solve-bench",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--binary",
        required=True,
        help="Path to the gf-solve-bench executable.",
    )
    p.add_argument(
        "--config",
        required=True,
        help="TOML configuration file (mode must be QDGreen). "
             "A [gf_checkpoint] section pointing to a precomputed GF is "
             "strongly recommended so setup time is not included in the benchmark.",
    )
    p.add_argument(
        "--ranks",
        nargs="+",
        type=int,
        required=True,
        metavar="N",
        help="List of MPI rank counts to benchmark (e.g. 1 2 4 8 16 32).",
    )
    p.add_argument(
        "--nreps",
        type=int,
        default=5,
        help="Number of timed solve repetitions per rank count (default: 5).",
    )
    p.add_argument(
        "--mpirun",
        default="mpirun",
        help="MPI launcher command (default: mpirun). "
             "Use 'srun' on Slurm clusters, or the full path if not in PATH.",
    )
    p.add_argument(
        "--output",
        default="solve_bench_results.csv",
        help="Output CSV file (default: solve_bench_results.csv).",
    )
    p.add_argument(
        "--plot",
        default="solve_bench_plot.png",
        help="Output plot file (default: solve_bench_plot.png). "
             "Set to '' to skip plotting.",
    )
    p.add_argument(
        "--petsc",
        default="",
        metavar="OPTIONS",
        help="PETSc options string passed after --petsc to the binary "
             "(e.g. \"-ksp_type gmres -pc_type bjacobi\").",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Per-run timeout in seconds (default: no timeout).",
    )
    return p.parse_args()


def build_command(args, n_ranks):
    cmd = [args.mpirun, "-n", str(n_ranks), args.binary,
           args.config, "--nreps", str(args.nreps)]
    if args.petsc.strip():
        cmd += ["--petsc"] + args.petsc.split()
    return cmd


def parse_output(stdout_text):
    """Extract key=value lines from gf-solve-bench stdout."""
    result = {}
    for line in stdout_text.splitlines():
        line = line.strip()
        if "=" in line and line.startswith("solve_bench_"):
            key, _, val = line.partition("=")
            result[key.strip()] = val.strip()
    return result


def run_one(cmd, timeout):
    t_start = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            text=True,
        )
    except subprocess.TimeoutExpired:
        return None, None, "TIMEOUT"
    elapsed = time.monotonic() - t_start

    if proc.returncode != 0:
        print("  STDERR:", proc.stderr[-2000:], file=sys.stderr)
        return None, elapsed, f"FAILED (exit {proc.returncode})"

    parsed = parse_output(proc.stdout)
    return parsed, elapsed, None


def save_csv(rows, path):
    if not rows:
        print("No results to save.", file=sys.stderr)
        return
    fieldnames = list(rows[0].keys())
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results saved to: {path}")


def plot_results(rows, path):
    if not path:
        return
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("matplotlib not available — skipping plot.", file=sys.stderr)
        return

    ranks     = [r["n_ranks"]      for r in rows]
    avg_times = [r["time_avg_s"]   for r in rows]
    min_times = [r["time_min_s"]   for r in rows]
    max_times = [r["time_max_s"]   for r in rows]

    err_lo = [a - mn for a, mn in zip(avg_times, min_times)]
    err_hi = [mx - a for mx, a  in zip(max_times, avg_times)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Left: absolute solve time ---
    ax = axes[0]
    ax.errorbar(ranks, avg_times,
                yerr=[err_lo, err_hi],
                fmt="o-", capsize=4, label="solve(v) wall time")
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("Wall-clock time per solve (s)")
    ax.set_title("Solve time vs MPI ranks")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xticks(ranks)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend()

    # --- Right: speedup relative to smallest rank count ---
    ax2 = axes[1]
    base_time = avg_times[0]
    base_rank = ranks[0]
    speedups = [base_time / t for t in avg_times]
    ideal    = [n / base_rank for n in ranks]

    ax2.plot(ranks, speedups, "o-", label="Measured speedup")
    ax2.plot(ranks, ideal,    "k--", alpha=0.5, label=f"Ideal (linear from {base_rank})")
    ax2.set_xlabel("MPI ranks")
    ax2.set_ylabel(f"Speedup (relative to {base_rank} rank(s))")
    ax2.set_title("Parallel efficiency")
    ax2.set_xscale("log", base=2)
    ax2.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax2.set_xticks(ranks)
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)
    ax2.legend()

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=150)
    print(f"Plot saved to: {path}")


def main():
    args = parse_args()

    if not os.path.isfile(args.binary):
        sys.exit(f"Binary not found: {args.binary}")
    if not os.path.isfile(args.config):
        sys.exit(f"Config not found: {args.config}")

    rank_list = sorted(set(args.ranks))
    print(f"Benchmarking gf-solve-bench with ranks: {rank_list}")
    print(f"  config : {args.config}")
    print(f"  nreps  : {args.nreps}")
    print(f"  mpirun : {args.mpirun}")
    if args.petsc.strip():
        print(f"  petsc  : {args.petsc}")
    print()

    rows = []
    for n in rank_list:
        cmd = build_command(args, n)
        print(f"Running: {' '.join(cmd)}")
        parsed, elapsed, error = run_one(cmd, args.timeout)

        if error:
            print(f"  ERROR ({error}) after {elapsed:.1f}s — skipping.\n")
            continue

        # Extract numeric results
        try:
            row = {
                "n_ranks":        int(parsed["solve_bench_n_ranks"]),
                "n_reps":         int(parsed["solve_bench_n_reps"]),
                "time_avg_s":     float(parsed["solve_bench_time_avg_s"]),
                "time_min_s":     float(parsed["solve_bench_time_min_s"]),
                "time_max_s":     float(parsed["solve_bench_time_max_s"]),
                "wall_elapsed_s": round(elapsed, 3),
            }
        except (KeyError, ValueError) as exc:
            print(f"  Failed to parse output ({exc}) — skipping.\n", file=sys.stderr)
            continue

        rows.append(row)
        print(f"  n_ranks={row['n_ranks']}  "
              f"avg={row['time_avg_s']:.4f}s  "
              f"min={row['time_min_s']:.4f}s  "
              f"max={row['time_max_s']:.4f}s  "
              f"(total elapsed {elapsed:.1f}s)\n")

    if not rows:
        sys.exit("No successful runs — nothing to save.")

    save_csv(rows, args.output)
    plot_results(rows, args.plot)


if __name__ == "__main__":
    main()
