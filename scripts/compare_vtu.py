#!/usr/bin/env python3
"""
Compare per-step VTU domain output between tandem and tandem_main.

tandem naming:      savage_prescott__N_0.vtu
tandem_main naming: savage_prescott_step_N.vtu_0.vtu

Common displacement fields: u0, u1, u2
Reports L-inf and L2 norms of the pointwise difference for each step/field.
Optionally writes per-step diff VTU files.
"""

import argparse
import os
import re
import sys
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy


def read_vtu(path):
    r = vtk.vtkXMLUnstructuredGridReader()
    r.SetFileName(path)
    r.Update()
    return r.GetOutput()


def get_point_array(ug, name):
    arr = ug.GetPointData().GetArray(name)
    if arr is None:
        return None
    return vtk_to_numpy(arr)


def write_diff_vtu(ref_ug, fields_diff, out_path):
    out = vtk.vtkUnstructuredGrid()
    out.DeepCopy(ref_ug)
    out.GetPointData().Initialize()
    for name, diff in fields_diff.items():
        a = vtk.vtkDoubleArray()
        a.SetName(f"diff_{name}")
        a.SetNumberOfTuples(len(diff))
        for i, v in enumerate(diff):
            a.SetValue(i, v)
        out.GetPointData().AddArray(a)
    w = vtk.vtkXMLUnstructuredGridWriter()
    w.SetFileName(out_path)
    w.SetInputData(out)
    w.Write()


def find_steps(tandem_dir, tandem_main_dir):
    pattern_new = re.compile(r"savage_prescott__(\d+)_0\.vtu$")
    pattern_old = re.compile(r"savage_prescott_step_(\d+)\.vtu_0\.vtu$")

    steps_new = {}
    for f in os.listdir(tandem_dir):
        m = pattern_new.match(f)
        if m:
            steps_new[int(m.group(1))] = os.path.join(tandem_dir, f)

    steps_old = {}
    for f in os.listdir(tandem_main_dir):
        m = pattern_old.match(f)
        if m:
            steps_old[int(m.group(1))] = os.path.join(tandem_main_dir, f)

    common = sorted(set(steps_new) & set(steps_old))
    only_new = sorted(set(steps_new) - set(steps_old))
    only_old = sorted(set(steps_old) - set(steps_new))

    if only_new:
        print(f"[warn] steps only in tandem:      {only_new}")
    if only_old:
        print(f"[warn] steps only in tandem_main: {only_old}")

    return common, steps_new, steps_old


def compare_step(step, path_new, path_old, fields, write_diff, diff_dir):
    ug_new = read_vtu(path_new)
    ug_old = read_vtu(path_old)

    n_pts_new = ug_new.GetNumberOfPoints()
    n_pts_old = ug_old.GetNumberOfPoints()
    if n_pts_new != n_pts_old:
        print(f"  [MISMATCH] step {step}: point counts differ ({n_pts_new} vs {n_pts_old})")
        return None

    results = {}
    diff_arrays = {}
    for field in fields:
        a_new = get_point_array(ug_new, field)
        a_old = get_point_array(ug_old, field)
        if a_new is None:
            print(f"  [skip] step {step}: '{field}' missing in tandem")
            continue
        if a_old is None:
            print(f"  [skip] step {step}: '{field}' missing in tandem_main")
            continue

        diff = a_new - a_old
        linf = np.max(np.abs(diff))
        l2 = np.sqrt(np.mean(diff ** 2))
        ref_linf = np.max(np.abs(a_old))
        rel_linf = linf / ref_linf if ref_linf > 0 else float("nan")

        results[field] = dict(linf=linf, l2=l2, rel_linf=rel_linf,
                              max_new=np.max(np.abs(a_new)),
                              max_old=np.max(np.abs(a_old)))
        diff_arrays[field] = np.abs(diff)

    if write_diff and diff_dir:
        os.makedirs(diff_dir, exist_ok=True)
        out_path = os.path.join(diff_dir, f"diff_step_{step}.vtu")
        write_diff_vtu(ug_old, diff_arrays, out_path)

    return results


def print_table_header():
    print(f"{'step':>5}  {'field':>6}  {'L-inf diff':>14}  {'rel L-inf':>12}  {'L2 diff':>14}  "
          f"{'max |new|':>12}  {'max |old|':>12}")
    print("-" * 90)


def print_row(step, field, r):
    print(f"{step:>5}  {field:>6}  {r['linf']:>14.6e}  {r['rel_linf']:>12.4e}  "
          f"{r['l2']:>14.6e}  {r['max_new']:>12.6e}  {r['max_old']:>12.6e}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tandem-dir", default="/export/dump/pkarki/build-test-new/app",
                        help="Directory with tandem VTU files")
    parser.add_argument("--tandem-main-dir", default="/export/dump/pkarki/build-VE-old-3d/app",
                        help="Directory with tandem_main VTU files")
    parser.add_argument("--fields", nargs="+", default=["u0", "u1", "u2"],
                        help="Point-data field names to compare (default: u0 u1 u2)")
    parser.add_argument("--steps", type=int, nargs="+", default=None,
                        help="Only compare these step indices (default: all common steps)")
    parser.add_argument("--write-diff", action="store_true",
                        help="Write per-step diff VTU files")
    parser.add_argument("--diff-dir", default="vtu_diff",
                        help="Output directory for diff VTU files (default: vtu_diff)")
    args = parser.parse_args()

    common, steps_new, steps_old = find_steps(args.tandem_dir, args.tandem_main_dir)
    if not common:
        print("No matching steps found between the two directories.")
        sys.exit(1)

    if args.steps is not None:
        requested = set(args.steps)
        missing = requested - set(common)
        if missing:
            print(f"[warn] requested steps not found in both dirs: {sorted(missing)}")
        common = sorted(requested & set(common))

    print(f"Comparing {len(common)} steps: {common}")
    print(f"Fields: {args.fields}\n")
    print_table_header()

    summary = {}  # step -> {field -> results}
    for step in common:
        res = compare_step(step, steps_new[step], steps_old[step],
                           args.fields, args.write_diff, args.diff_dir)
        if res is None:
            continue
        summary[step] = res
        for field, r in res.items():
            print_row(step, field, r)

    # Per-field summary across all steps
    print("\n" + "=" * 90)
    print("Per-field summary (max L-inf across all steps):")
    for field in args.fields:
        vals = [summary[s][field]["linf"] for s in summary if field in summary[s]]
        rels = [summary[s][field]["rel_linf"] for s in summary if field in summary[s]]
        if not vals:
            continue
        worst_step = common[np.argmax(vals)]
        print(f"  {field}: max L-inf = {max(vals):.6e}  (rel {max(rels):.4e})  "
              f"at step {worst_step}")

    if args.write_diff:
        print(f"\nDiff VTU files written to: {os.path.abspath(args.diff_dir)}/")


if __name__ == "__main__":
    main()
