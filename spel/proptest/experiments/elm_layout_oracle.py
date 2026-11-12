"""Compare hierarchy.Layout with what ELM built, gridcell by gridcell, from a restart file.

usage: python -m spel.proptest.experiments.elm_layout_oracle RESTART [nsample] [seed]

For each sampled gridcell, a declaration is rebuilt from the restart's land1d/cols1d/pfts1d
records and laid out twice:
  full     : every element ELM created, with its weight (no subgrid);
  implicit : only elements of weight > 0 (plus nothing else), with a Subgrid giving ELM's ranges.
Both must reproduce, in ELM's order, the itype, weight and active flag of every landunit, column
and patch of that gridcell.
"""
import sys
from collections import Counter

import netCDF4
import numpy as np

from spel.proptest.hierarchy import (TYPE_CODES, Column, Gridcell, HierarchyError, Landunit, LandunitKind, Layout,
                                     Patch, Subgrid, Topounit, UrbanDensity)

KIND = {1: LandunitKind.SOIL, 2: LandunitKind.CROP, 3: LandunitKind.GLACIER, 4: LandunitKind.GLACIER_MEC,
        5: LandunitKind.LAKE, 6: LandunitKind.WETLAND, 7: LandunitKind.URBAN, 8: LandunitKind.URBAN,
        9: LandunitKind.URBAN}
DENS = {7: UrbanDensity.TBD, 8: UrbanDensity.HD, 9: UrbanDensity.MD}


def load(path):
    d = netCDF4.Dataset(path)
    d.set_auto_mask(False)
    r = {k: np.asarray(d[k][:]) for k in d.variables if k.split("_")[0] in ("land1d", "cols1d", "pfts1d", "topo1d")}
    return r


def group(owner, n):
    order = np.argsort(owner, kind="stable")
    bounds = np.searchsorted(owner[order], np.arange(1, n + 2))
    return [order[bounds[k]:bounds[k + 1]] for k in range(n)]


def main():
    r = load(sys.argv[1])
    nsample = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    rng = np.random.default_rng(int(sys.argv[3]) if len(sys.argv) > 3 else 0)
    ng = int(r["land1d_gridcell_index"].max())
    nt = len(r["topo1d_gridcell_index"])
    tops_of_g = group(r["topo1d_gridcell_index"], ng)
    lus_of_t = group(r["land1d_topounit_index"], nt)
    nl = len(r["land1d_ityplun"])
    cols_of_l = group(r["cols1d_landunit_index"], nl)
    pats_of_c = group(r["pfts1d_column_index"], len(r["cols1d_ityp"]))
    natpft = (int(r["pfts1d_itypveg"][r["pfts1d_ityplun"] == 1].min()),
              int(r["pfts1d_itypveg"][r["pfts1d_ityplun"] == 1].max()))
    print("natpft range in the restart:", natpft)

    # every special topounit kind, then a random sample
    kinds_t = {}
    for t in range(nt):
        ls = lus_of_t[t]
        pos = tuple(sorted(int(r["land1d_ityplun"][l]) for l in ls if r["land1d_wttopounit"][l] > 0))
        kinds_t[t] = pos
    special = Counter(kinds_t.values())
    print("topounit kinds (types of weight>0 landunits):", special.most_common(15))
    picks = set(rng.choice(ng, size=min(nsample, ng), replace=False).tolist())
    seen = set()
    for t, k in kinds_t.items():
        if k not in seen:
            seen.add(k)
            picks.add(int(r["topo1d_gridcell_index"][t]) - 1)
    stats = Counter()
    bad = []
    for g in sorted(picks):
        tops_full, tops_imp, elm = [], [], {"l": [], "c": [], "p": []}
        tw_sum = []
        for t in tops_of_g[g]:
            lus_full, lus_imp = [], []
            for l in lus_of_t[t]:
                lt = int(r["land1d_ityplun"][l])
                kind = KIND[lt]
                cols_full, cols_imp = [], []
                for c in cols_of_l[l]:
                    pats = [Patch(int(r["pfts1d_itypveg"][p]), float(r["pfts1d_wtcol"][p])) for p in pats_of_c[c]]
                    cw = float(r["cols1d_wtlnd"][c])
                    cols_full.append(Column(int(r["cols1d_ityp"][c]), tuple(pats), cw))
                    imp = tuple(p for p in pats if p.weight > 0)
                    if cw > 0 or kind is LandunitKind.URBAN:
                        cols_imp.append(Column(int(r["cols1d_ityp"][c]), imp or tuple(pats[:1]), cw))
                lw = float(r["land1d_wttopounit"][l])
                lus_full.append(Landunit(kind, tuple(cols_full), lw, DENS.get(lt)))
                if lw > 0 or kind in (LandunitKind.URBAN, LandunitKind.SOIL):
                    lus_imp.append(Landunit(kind, tuple(cols_imp), lw, DENS.get(lt)))
            tw = float(sum(r["land1d_wtxy"][l] for l in lus_of_t[t]))
            tw_sum.append(tw)
            tops_full.append(Topounit(tuple(lus_full), tw, 0.0))
            tops_imp.append(Topounit(tuple(lus_imp), tw, 0.0))
        for l in sorted(int(l) for t in tops_of_g[g] for l in lus_of_t[t]):
            elm["l"].append((int(r["land1d_ityplun"][l]), float(r["land1d_wttopounit"][l]), bool(r["land1d_active"][l])))
            for c in cols_of_l[l]:
                elm["c"].append((int(r["cols1d_ityp"][c]), float(r["cols1d_wtlnd"][c]), bool(r["cols1d_active"][c])))
                for p in pats_of_c[c]:
                    elm["p"].append((int(r["pfts1d_itypveg"][p]), float(r["pfts1d_wtcol"][p]),
                                     bool(r["pfts1d_active"][p])))
        s = sum(tw_sum)
        tops_full = [Topounit(t.landunits, t.weight / s) for t in tops_full]
        tops_imp = [Topounit(t.landunits, t.weight / s) for t in tops_imp]
        for label, tops, sub in (("full", tops_full, None), ("implicit", tops_imp, Subgrid(natpft))):
            try:
                lay = Layout([Gridcell(tuple(tops))], subgrid=sub)
            except HierarchyError as exc:
                stats[f"{label}: declaration refused"] += 1
                bad.append((g, label, str(exc)[:200]))
                continue
            ok = True
            for lev in ("l", "c", "p"):
                it = lay.itype(lev)
                w = lay.weight(lev, {"l": "t", "c": "l", "p": "c"}[lev])
                act = lay.active(lev)
                mine = list(zip(it.tolist(), w.tolist(), act.tolist()))
                theirs = elm[lev]
                if len(mine) != len(theirs):
                    ok = False
                    bad.append((g, label, f"{lev}: {len(mine)} elements, ELM {len(theirs)}"))
                    break
                for k, (a, b) in enumerate(zip(mine, theirs)):
                    if a[0] != b[0] or abs(a[1] - b[1]) > 1e-6 or a[2] != b[2]:
                        ok = False
                        bad.append((g, label, f"{lev}[{k}]: layout {a}, ELM {b}"))
                        break
                if not ok:
                    break
            stats[f"{label}: {'same' if ok else 'DIFFERENT'}"] += 1
    print(f"{len(picks)} gridcells:", dict(stats))
    for b in bad[:25]:
        print("  ", b)


if __name__ == "__main__":
    main()
