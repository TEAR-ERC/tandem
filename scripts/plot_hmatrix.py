#!/usr/bin/env python3
"""
Visualize H-matrix leaf structure from CSV files produced by gf-hmatrix-visualize.

CSV format:
  row 0 : M,N          -- global matrix dimensions
  row k : row_start, row_count, col_start, col_count, compression_rank
           compression_rank == -1  =>  dense block
           compression_rank >= 0   =>  low-rank compressed block

Single-matrix mode (default):
    python scripts/plot_hmatrix.py --inputfile hmatrix_structure.csv [options]

Split mode (component sub-matrices from HMatrixGreenFunction):
    python scripts/plot_hmatrix.py --prefix hmat_bp1 --split --D 2 [options]
    Reads hmat_bp1_ab<alpha><beta>.csv for all (alpha, beta) pairs.
    Produces a D x (D-1) subplot grid.

Options:
    --inputfile FILE   CSV file (single-matrix mode, required unless --split)
    --prefix STR       Filename prefix (split mode, required with --split)
    --split            Enable split mode (component sub-matrices)
    --D N              Number of traction components: 2 for 2D, 3 for 3D  (default: 2)
    --show             Open an interactive matplotlib window  (default: on)
    --no-show          Suppress interactive window
    --save FILE        Save figure to FILE (PNG, PDF, SVG, ...)
    --ranks N          Draw row-partition lines (assumes N equal-size blocks)
    --dpi N            Figure DPI for raster output  (default: 150)
    --annotate         Annotate large blocks with compression rank
"""

import argparse
import sys

import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputfile", default="", metavar="FILE",
                   help="Leaf-structure CSV (single-matrix mode)")
    p.add_argument("--prefix", default="", metavar="STR",
                   help="Filename prefix for split mode")
    p.add_argument("--split", action="store_true",
                   help="Split mode: read one CSV per (alpha, beta) component pair")
    p.add_argument("--D", type=int, default=2, metavar="N",
                   help="Number of traction components (2=2D, 3=3D; used with --split)")
    p.add_argument("--show", dest="show", action="store_true", default=True)
    p.add_argument("--no-show", dest="show", action="store_false")
    p.add_argument("--save", default="", metavar="FILE", help="Output image path")
    p.add_argument("--ranks", type=int, default=0, metavar="N",
                   help="Number of MPI ranks (draws row-partition lines)")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--annotate", action="store_true",
                   help="Annotate large blocks with compression rank")
    return p.parse_args()


def truncate_colormap(cmap, lo=0.0, hi=1.0, n=256):
    return mcolors.LinearSegmentedColormap.from_list(
        f"trunc({cmap.name},{lo:.2f},{hi:.2f})",
        cmap(np.linspace(lo, hi, n))
    )


def load_csv(path, allow_missing=False):
    """Load a leaf-structure CSV.  Returns (nr, nc, data) or None if missing and allow_missing."""
    try:
        size_row = pd.read_csv(path, nrows=1, header=None)
    except FileNotFoundError:
        if allow_missing:
            return None
        sys.exit(f"Error: file not found: {path}")
    nr = int(size_row.iloc[0, 0])
    nc = int(size_row.iloc[0, 1])
    data = pd.read_csv(path, skiprows=1, header=None,
                       names=["row0", "nrows", "col0", "ncols", "crank"])
    if data.empty:
        sys.exit(f"Error: no leaf data in {path}")
    return nr, nc, data


def draw_hmatrix(ax, nr, nc, data, args, title=""):
    """Draw one H-matrix onto ax."""
    max_rank = max(int(data["crank"].max()), 1)
    green_cmap = truncate_colormap(plt.get_cmap("YlGn"), lo=0.35, hi=1.0)
    green_cmap.set_bad(color="#d62728")

    rank_img = np.full((nr, nc), np.nan)

    for _, row in data.iterrows():
        r0, nr_b = int(row["row0"]), int(row["nrows"])
        c0, nc_b = int(row["col0"]), int(row["ncols"])
        crank    = int(row["crank"])

        if crank >= 0:
            rank_img[r0:r0 + nr_b, c0:c0 + nc_b] = float(crank)

        rect = patches.Rectangle(
            (c0 - 0.5, r0 - 0.5), nc_b, nr_b,
            linewidth=0.5, edgecolor="black", facecolor="none"
        )
        ax.add_patch(rect)

        if args.annotate and crank >= 0:
            if nc_b / nc > 0.05 and nr_b / nr > 0.05:
                ax.annotate(str(crank),
                            (c0 + nc_b / 2.0, r0 + nr_b / 2.0),
                            color="white", fontsize=7, va="center", ha="center",
                            fontweight="bold")

    masked = np.ma.masked_invalid(rank_img)
    im = ax.imshow(masked, cmap=green_cmap, vmin=0, vmax=max_rank,
                   aspect="auto", interpolation="none", origin="upper")

    ax.set_xlim(-0.5, nc - 0.5)
    ax.set_ylim(nr - 0.5, -0.5)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_title(title, fontsize=9)

    if args.ranks > 1:
        rows_per_rank = nr / args.ranks
        for r in range(1, args.ranks):
            y = r * rows_per_rank - 0.5
            ax.axhline(y, color="white", linewidth=1.0, linestyle="--", alpha=0.8)

    dense_patch = patches.Patch(facecolor="#d62728", label="dense")
    ax.legend(handles=[dense_patch], loc="lower right", fontsize=6)
    return im


def main():
    args = parse_args()

    if args.split:
        # ---- Split mode: D x (D-1) subplot grid ----
        if not args.prefix:
            sys.exit("Error: --prefix is required with --split")
        D     = args.D
        slip_D = D - 1

        traction_labels = {0: "normal", 1: "shear₁", 2: "shear₂"}
        slip_labels     = {0: "slip₁",  1: "slip₂"}

        fig, axes = plt.subplots(D, slip_D,
                                 figsize=(5 * slip_D, 5 * D),
                                 squeeze=False)
        fig.suptitle(f"H-matrix component split  ({D}D, prefix={args.prefix})", fontsize=11)

        last_im = None
        for alpha in range(D):
            for beta in range(slip_D):
                path = f"{args.prefix}_ab{alpha}{beta}.csv"
                result = load_csv(path, allow_missing=True)
                ax = axes[alpha][beta]
                tl = traction_labels.get(alpha, f"α={alpha}")
                sl = slip_labels.get(beta, f"β={beta}")
                if result is None:
                    # Component was skipped (planar_fault=true) — show as zero matrix
                    ax.set_facecolor("#e8e8e8")
                    ax.set_title(
                        f"G_{alpha}{beta}: {tl} ← {sl}\n(skipped — zero by planar-fault symmetry)",
                        fontsize=8)
                    ax.text(0.5, 0.5, "zero\n(not built)",
                            transform=ax.transAxes,
                            ha="center", va="center", fontsize=11, color="#666666")
                    ax.set_xticks([])
                    ax.set_yticks([])
                else:
                    nr, nc, data = result
                    title = f"G_{alpha}{beta}: {tl} ← {sl}\n({nr}×{nc})"
                    last_im = draw_hmatrix(ax, nr, nc, data, args, title=title)

        if last_im is not None:
            fig.colorbar(last_im, ax=axes.ravel().tolist(),
                         fraction=0.02, pad=0.02, label="Compression rank  (red = dense)")
        plt.tight_layout()

    else:
        # ---- Single-matrix mode ----
        if not args.inputfile:
            sys.exit("Error: --inputfile is required (or use --split for component mode)")

        nr, nc, data = load_csv(args.inputfile)
        max_rank = max(int(data["crank"].max()), 1)
        green_cmap = truncate_colormap(plt.get_cmap("YlGn"), lo=0.35, hi=1.0)
        green_cmap.set_bad(color="#d62728")

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title(f"H-matrix leaf structure  ({nr} × {nc})")
        im = draw_hmatrix(ax, nr, nc, data, args,
                          title=f"H-matrix  ({nr} × {nc})")
        ax.set_xlabel("column DOF index")
        ax.set_ylabel("row DOF index")
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02,
                     label="Compression rank  (red = dense)")
        plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved to {args.save}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
