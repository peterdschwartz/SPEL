"""ELM's set_topounit downhill_ti (dn_index bug included) vs the intended one, per gridcell."""
import sys, numpy as np, netCDF4 as nc
from collections import Counter

d = nc.Dataset(sys.argv[1])
n = np.ma.filled(d["topoPerGrid"][:], 0).astype(int).ravel()
elv = np.ma.filled(d["TopounitAveElv"][:], np.nan).reshape(d["TopounitAveElv"].shape[0], -1)
wt = np.ma.filled(d["TopounitFracArea"][:], 0.0).reshape(d["TopounitFracArea"].shape[0], -1)
lat = np.ma.filled(d["LATIXY"][:]).ravel(); lon = np.ma.filled(d["LONGXY"][:]).ravel()

def elm(e):  # exact copy of the Fortran loop (0-based), dn_index set once
    mn = e.min(); out = []; dn = -1
    for i in range(len(e)):
        de = mn
        for j in range(len(e)):
            if e[j] < e[i] and e[j] >= de:
                de = e[j]; dn = j
        out.append(dn)
    return out

def intended(e):
    out = []
    for i in range(len(e)):
        de = e.min(); dn = -1
        for j in range(len(e)):
            if e[j] < e[i] and e[j] >= de:
                de = e[j]; dn = j
        out.append(dn)
    return out

c = Counter(); examples = {}
for g in np.nonzero(n > 1)[0]:
    k = n[g]; e = elv[:k, g]; w = wt[:k, g]
    c["gridcells with >1 topounit"] += 1
    if np.all(np.diff(e) > 0): c["order: strictly ascending"] += 1
    elif np.all(np.diff(e) < 0): c["order: strictly descending"] += 1
    else: c["order: other"] += 1
    if len(set(e.tolist())) < k: c["has equal elevations"] += 1
    if np.argmin(e) != 0: c["lowest not first"] += 1
    a, b = elm(e), intended(e)
    kinds = set()
    for i, (x, y) in enumerate(zip(a, b)):
        if x == y: continue
        if x == i: kinds.add("self-loop")
        elif x >= 0 and e[x] > e[i]: kinds.add("uphill")
        elif x >= 0 and e[x] == e[i]: kinds.add("level")
        else: kinds.add("other")
    for i in range(k):  # cycles longer than a self-loop
        seen = []; j = i
        while j != -1 and j not in seen: seen.append(j); j = a[j]
        if j != -1 and a[j] != j: kinds.add("cycle"); break
    if kinds:
        c["H1: wrong downhill_ti"] += 1
        for kk in kinds: c["H1 " + kk] += 1
        examples.setdefault(tuple(sorted(kinds)), []).append(
            (g, round(float(lat[g]), 2), round(float(lon[g]), 2), np.round(e, 1).tolist(), np.round(w, 3).tolist(), a, b))
    if (w <= 0).any(): c["has a weight-0 topounit"] += 1
    if any(w[x] <= 0 for x in a if x >= 0): c["H2: weight-0 downhill target (ELM)"] += 1
    if any(w[x] <= 0 for x in b if x >= 0): c["H2: weight-0 downhill target (intended)"] += 1
for key in sorted(c): print(f"{key:45s} {c[key]}")
print("topoPerGrid histogram:", dict(sorted(Counter(n.tolist()).items())))
for kinds, ex in examples.items():
    print(f"\n{kinds}: {len(ex)} gridcells; first 3:")
    for row in ex[:3]: print("  g=%d lat=%s lon=%s elv=%s wt=%s elm=%s intended=%s" % row)
