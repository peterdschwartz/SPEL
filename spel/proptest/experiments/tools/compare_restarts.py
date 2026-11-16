"""Where do two ELM restart files differ? Every variable on a subgrid dimension is compared
bit for bit (NaN = NaN), and each differing entry is mapped to its gridcell (the *1d_gridcell_index
fields); prints the gridcells that differ (index, lat, lon) and, per variable, how many entries.

    python compare_restarts.py A.elm.r.nc B.elm.r.nc [--expect LAT,LON ...]
"""
import argparse
from collections import defaultdict

import netCDF4
import numpy as np

GRID_INDEX = {"gridcell": None, "topounit": "topo1d_gridcell_index", "landunit": "land1d_gridcell_index",
              "column": "cols1d_gridcell_index", "pft": "pfts1d_gridcell_index"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--expect", nargs="*", default=[], help="lat,lon of the gridcells expected to differ")
    args = ap.parse_args()
    A, B = netCDF4.Dataset(args.a), netCDF4.Dataset(args.b)
    for d in (A, B):
        d.set_auto_mask(False)
    lat, lon = np.asarray(A["grid1d_lat"][:]), np.asarray(A["grid1d_lon"][:])
    owner = {dim: (np.arange(len(lat)) if v is None else np.asarray(A[v][:]) - 1) for dim, v in GRID_INDEX.items()
             if v is None or v in A.variables}
    per_var, cells, nonsub = {}, defaultdict(set), []
    for name, va in A.variables.items():
        if name not in B.variables:
            continue
        vb = B[name]
        x, y = np.asarray(va[:]), np.asarray(vb[:])
        if x.shape != y.shape:
            per_var[name] = f"shape {x.shape} vs {y.shape}"
            continue
        if x.dtype.kind not in "fiu":
            continue
        diff = (x != y) & ~(np.isnan(x) & np.isnan(y)) if x.dtype.kind == "f" else (x != y)
        if not diff.any():
            continue
        dim = next((dname for dname in va.dimensions if dname in owner), None)
        if dim is None:
            nonsub.append(name)
            continue
        ax = va.dimensions.index(dim)
        rows = np.nonzero(np.moveaxis(diff, ax, 0).reshape(diff.shape[ax], -1).any(axis=1))[0]
        g = np.unique(owner[dim][rows])
        per_var[name] = f"{int(diff.sum())} entries in {len(g)} gridcells"
        for gi in g:
            cells[int(gi)].add(name)
    expect = {tuple(round(float(v), 2) for v in e.split(",")) for e in args.expect}
    print(f"{len(per_var)} variables differ; {len(cells)} gridcells differ")
    for gi in sorted(cells):
        key = (round(float(lat[gi]), 2), round(float(lon[gi]), 2))
        mark = "" if not expect else ("  expected" if key in expect or (key[0], round(key[1] - 360, 2)) in expect else "  UNEXPECTED")
        print(f"  grid1d {gi + 1:6d}  lat {lat[gi]:7.2f}  lon {lon[gi]:8.2f}  {len(cells[gi]):3d} variables{mark}")
    for name in sorted(per_var):
        print(f"  {name}: {per_var[name]}")
    if nonsub:
        print("differ without a subgrid dimension:", ", ".join(sorted(nonsub)))


if __name__ == "__main__":
    main()
