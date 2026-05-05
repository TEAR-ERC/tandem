#!/usr/bin/env python3
"""
Visualize an H-matrix leaf structure from a CSV produced by gf-hmatrix-visualize.

CSV format (written by SeasQDDiscreteGreenOperator::export_h_structure):
  row 0 : M,N          -- global matrix dimensions
  row k : row_start, row_count, col_start, col_count, compression_rank
           compression_rank == -1  =>  dense (full) block
           compression_rank >= 0   =>  low-rank compressed block

Usage:
    python scripts/plot_hmatrix.py --inputfile hmatrix_structure.csv [options]

Options:
    --inputfile FILE   CSV file produced by gf-hmatrix-visualize  (required)
    --show             Open an interactive matplotlib window        (default: on)
    --no-show          Suppress interactive window
    --save FILE        Save figure to FILE (PNG, PDF, SVG, ...)
    --ranks N          Draw horizontal dashed lines at MPI row-block boundaries
                       (assumes equal-size blocks: N ranks splitting M rows)
    --dpi N            Figure DPI for raster output (default: 150)
    --annotate         Annotate large blocks with their compression rank
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
    p.add_argument("--inputfile", required=True, help="Leaf-structure CSV file")
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


def main():
    args = parse_args()

    # --- Load CSV ---
    try:
        size_row = pd.read_csv(args.inputfile, nrows=1, header=None)
    except FileNotFoundError:
        sys.exit(f"Error: file not found: {args.inputfile}")

    nr = int(size_row.iloc[0, 0])   # global rows  (M)
    nc = int(size_row.iloc[0, 1])   # global cols  (N)

    data = pd.read_csv(args.inputfile, skiprows=1, header=None,
                       names=["row0", "nrows", "col0", "ncols", "crank"])

    if data.empty:
        sys.exit("Error: no leaf data found in CSV.")

    max_rank = int(data["crank"].max())

    # --- Colormap ---
    # Green gradient for compressed blocks; clipped so very low ranks still show colour
    green_cmap = truncate_colormap(plt.get_cmap("YlGn"), lo=0.35, hi=1.0)
    # Dense blocks are shown in red (rank == -1 treated as NaN -> bad-colour)
    green_cmap.set_bad(color="#d62728")   # matplotlib red

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-0.5, nc - 0.5)
    ax.set_ylim(nr - 0.5, -0.5)      # row 0 at top
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("column DOF index")
    ax.set_ylabel("row DOF index")
    ax.set_title(f"H-matrix leaf structure  ({nr} × {nc})")

    # Background (zero-initialised; dense blocks will be NaN)
    rank_img = np.zeros((nr, nc), dtype=float)
    rank_img[:] = np.nan    # start everything as "dense/unknown" (red)

    # --- Draw blocks ---
    for _, row in data.iterrows():
        r0, nr_b = int(row["row0"]), int(row["nrows"])
        c0, nc_b = int(row["col0"]), int(row["ncols"])
        crank    = int(row["crank"])

        if crank >= 0:
            rank_img[r0:r0 + nr_b, c0:c0 + nc_b] = float(crank)

        # Rectangle outline
        rect = patches.Rectangle(
            (c0 - 0.5, r0 - 0.5), nc_b, nr_b,
            linewidth=0.6, edgecolor="black", facecolor="none"
        )
        ax.add_patch(rect)

        # Annotate large blocks with their rank value
        if args.annotate and crank >= 0:
            if nc_b / nc > 0.05 and nr_b / nr > 0.05:
                ax.annotate(
                    str(crank),
                    (c0 + nc_b / 2.0, r0 + nr_b / 2.0),
                    color="white", fontsize=8, va="center", ha="center",
                    fontweight="bold"
                )

    # Render the image
    masked = np.ma.masked_invalid(rank_img)
    im = ax.imshow(
        masked,
        cmap=green_cmap,
        vmin=0,
        vmax=max(max_rank, 1),
        aspect="auto",
        interpolation="none",
        origin="upper"
    )

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Compression rank  (red = dense)")

    # --- MPI rank partition lines ---
    if args.ranks > 1:
        rows_per_rank = nr / args.ranks
        for r in range(1, args.ranks):
            y = r * rows_per_rank - 0.5
            ax.axhline(y, color="white", linewidth=1.2, linestyle="--", alpha=0.8)
            ax.text(-0.01 * nc, y, f"rank {r}", color="white",
                    fontsize=7, va="center", ha="right")

    # --- Legend patch for dense blocks ---
    dense_patch = patches.Patch(facecolor="#d62728", label="dense block")
    ax.legend(handles=[dense_patch], loc="lower right", fontsize=8)

    plt.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved to {args.save}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
