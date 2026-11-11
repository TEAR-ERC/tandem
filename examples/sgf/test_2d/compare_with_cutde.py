"""Compare the 2D sgf test GFs with cutde in the plane-strain limit.

A 2D dislocation equals a 3D rectangular fault that is very long along strike,
so each 2D patch is modelled in cutde as a 2 x 10^4 km long rectangle (two
triangles) and the surface displacement is taken on the cross-section y = 0.

    python compare_with_cutde.py <directory with sgf_2d_test_grid.h5>

Reports, for each patch, the sign that matches (cutde's dip-slip convention is
opposite to the sgf fault basis: -1 expected) and the relative and maximum
absolute error for 0.5 km < |x| <= 100 km.
"""
import sys

import cutde.halfspace as hs
import h5py
import numpy as np

directory = sys.argv[1] if len(sys.argv) > 1 else "."
dip = np.deg2rad(60.0)
L = 1.0e4  # half length along strike (km)
patches = {10: (0, 10), 11: (10, 20), 12: (20, 30), 13: (30, 40)}  # km down dip

with h5py.File(f"{directory}/sgf_2d_test_grid.h5") as f:
    x = f["x"][:]
    G = {t: f[str(t)][:, 0, :] for t in patches}  # (x, component)

# 3D frame: x3 = x, y3 = strike, z3 = 2D y
obs = np.c_[x, np.zeros_like(x), np.zeros_like(x)]
mask = (np.abs(x) <= 100.0) & (np.abs(x) > 0.5)


def point(s, y):
    return [s * np.cos(dip), y, -s * np.sin(dip)]


for t, (s1, s2) in patches.items():
    A, B, C, D = point(s1, -L), point(s1, L), point(s2, L), point(s2, -L)
    tris = np.array([[A, B, C], [A, C, D]])
    u = hs.disp_matrix(obs, tris, 0.25)[:, :, :, 1].sum(axis=2)  # unit dip slip
    uc = np.c_[u[:, 0], u[:, 2]]
    rel = {s: np.linalg.norm(G[t][mask] - s * uc[mask]) / np.linalg.norm(uc[mask])
           for s in (1, -1)}
    sign = min(rel, key=rel.get)
    err = G[t][mask] - sign * uc[mask]
    print(f"tag {t} ({s1}-{s2} km): sign {sign:+d}  rel L2 err {rel[sign]:.2e}  "
          f"max abs err {np.abs(err).max():.2e}  max|u| {np.abs(uc[mask]).max():.3f}")
