"""
Filter relations from the sites that build them.

The static trace records every assignment that fills a member of a
`clumpfilter` instance (`this_filter%lakec(fl) = c`): the member, the loop it
runs in (over subgrid bounds, or over another filter: the parent) and the
conditions of its enclosing ifs. From those:

- A filter built inside a loop over another filter is a subset of it
  (`nolakeurban_vegp` of `nolakeurbanp`, `snowc` of `nolakec`).
- Two filters filled in the same routine and loop domain are disjoint when
  every pair of their sites sits in different branches of one if (one guard
  is the `not(...)` of the other, as registrations.iter_statements writes the
  else and else-if branches): `lakec`/`nolakec`, `pcropp`/`ppercropp`.
- One is a subset of the other when each of its sites carries the other's
  guards and more (`hydrononsoic` of `hydrologyc`).
- Disjointness is inherited by subsets.

Facts that no syntax gives (lake, urban, soil and glacier landunits are
exclusive) are listed in CURATED_DISJOINT and marked `source=curated`.
Anything not derived is assumed to overlap: a dependency is only ever dropped
when both filters are known and known to be disjoint.
"""

from __future__ import annotations

import re
from collections import defaultdict

# Subgrid levels, coarsest first: an element of one level belongs to exactly one
# element of each coarser level, so a condition on a coarser level's element
# (`lun_pp%lakpoi(l)`) decides the same way for a column and for its patches.
LEVELS = "gtlcp"

# Families of filters that cannot share an element at any level, from ELM's
# landunit types (lun_pp%itype is one of soil/crop, lake, urban, glacier-mec,
# ...) and column types (a pervious road column is urban, not soil). Every
# member of one family is disjoint from every member of the other, across
# levels as well (lakec vs soilp: no soil patch sits on a lake column).
FAMILIES: dict[str, tuple[str, ...]] = {
    "lake": ("lakec", "lakep"),
    "soil": ("soilc", "soilp"),
    "urban": ("urbanl", "urbanc", "urbanp"),
    "icemec": ("icemecc",),
    "hydrology": ("hydrologyc",),  # soil, crop and pervious-road columns
    "roadperv": ("hydrononsoic",),
    "smb": ("do_smb_c",),  # glacier-mec, or soil with an ice mask
}
CURATED_DISJOINT: list[tuple[str, str, str]] = [
    ("lake", "soil", "lake vs soil/crop landunits"),
    ("lake", "urban", "lake vs urban landunits"),
    ("lake", "icemec", "lake vs glacier-mec landunits"),
    ("lake", "hydrology", "lake vs soil/crop/pervious-road columns"),
    ("lake", "roadperv", "lake vs pervious-road columns"),
    ("lake", "smb", "lake vs glacier-mec/soil columns"),
    ("soil", "urban", "soil/crop vs urban landunits"),
    ("soil", "icemec", "soil/crop vs glacier-mec landunits"),
    ("soil", "roadperv", "soil/crop vs pervious-road columns"),
    ("urban", "icemec", "urban vs glacier-mec landunits"),
    ("urban", "smb", "urban vs glacier-mec/soil columns"),
    ("icemec", "roadperv", "glacier-mec vs pervious-road columns"),
    ("icemec", "hydrology", "glacier-mec vs soil/crop/pervious-road columns"),
]


def guard_levels(text: str) -> set[str]:
    """The subgrid index letters a condition subscripts with: `lun_pp%lakpoi(l)` -> {'l'}."""
    return set(re.findall(r"\(([gtlcp])\)", text))


def coarser_than_both(text: str, level_a: str, level_b: str) -> bool:
    """
    A condition on elements of a level at or above the coarser of the two
    filters' levels: a landunit property decides the same way for the landunit,
    its columns and their patches.
    """
    levels = guard_levels(text)
    if not levels or level_a not in LEVELS or level_b not in LEVELS:
        return False
    top = min(LEVELS.index(level_a), LEVELS.index(level_b))
    return all(LEVELS.index(v) <= top for v in levels)


def contradiction(ga: tuple[str, ...], gb: tuple[str, ...]) -> str:
    """A guard of one site whose negation guards the other, else ''."""
    sa, sb = set(ga), set(gb)
    for t in sa:
        if f"not({t})" in sb:
            return t
    for t in sb:
        if f"not({t})" in sa:
            return t
    return ""


def derive(sites: list[dict], curated=CURATED_DISJOINT) -> tuple[list[dict], list[dict]]:
    """`filters` and `filter_relations` rows (schema.TABLES) from the trace's filter sites."""
    by_name: dict[str, list[dict]] = defaultdict(list)
    seen: set[tuple] = set()
    for s in sites:
        key = (s["name"], s["subroutine"], s["ln"])
        if key in seen:
            continue
        seen.add(key)
        by_name[s["name"]].append(s)
    names = sorted(by_name)
    filter_rows = []
    for name in names:
        first = by_name[name][0]
        filter_rows.append({
            "name": name, "subroutine": first["subroutine"], "ln": first["ln"], "level": first["level"],
            "parent": first["parent"], "guards": " & ".join(first["guards"]),
        })

    relations: dict[tuple[str, str, str], dict] = {}

    def add(a: str, b: str, relation: str, source: str, subroutine: str = "", condition: str = "") -> None:
        if relation == "disjoint":
            a, b = sorted((a, b))
        key = (a, b, relation)
        if a != b and key not in relations:
            relations[key] = {"filter_a": a, "filter_b": b, "relation": relation, "source": source,
                              "subroutine": subroutine, "condition": condition}

    def group(s: dict) -> tuple:
        return (s["subroutine"], s["domain"]) if s["domain"] else None

    for name in names:
        for s in by_name[name]:
            if s["parent"]:
                add(name, s["parent"], "subset", "parent", s["subroutine"], f"loop over {s['parent']}")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pairs = [(sa, sb) for sa in by_name[a] for sb in by_name[b]]
            same = [(sa, sb) for sa, sb in pairs if group(sa) is not None and group(sa) == group(sb)]
            if len(same) != len(pairs):
                # different loops: still disjoint when every pair is decided by one condition on a
                # coarser level (lakec and nolakep both hinge on the landunit's lakpoi)
                cross = [contradiction(sa["guards"], sb["guards"]) for sa, sb in pairs
                         if sa["subroutine"] == sb["subroutine"] and sa["domain"].startswith("range:") and sb["domain"].startswith("range:")]
                if len(cross) == len(pairs) and all(cross) and all(
                    coarser_than_both(c, sa["level"], sb["level"]) for c, (sa, sb) in zip(cross, pairs)
                ):
                    add(a, b, "disjoint", "if/else", pairs[0][0]["subroutine"], cross[0] + " (across levels)")
                continue  # otherwise not comparable: assume overlap
            conds = [contradiction(sa["guards"], sb["guards"]) for sa, sb in same]
            if all(conds):
                add(a, b, "disjoint", "if/else", same[0][0]["subroutine"], conds[0])
                continue
            for x, y in ((a, b), (b, a)):
                if all(any(set(sy["guards"]) < set(sx["guards"]) for sy in by_name[y] if group(sy) == group(sx)) for sx in by_name[x]):
                    extra = sorted(set(by_name[x][0]["guards"]) - set(by_name[y][0]["guards"]))
                    add(x, y, "subset", "nested", by_name[x][0]["subroutine"], " & ".join(extra))
    known = set(names)
    for fa, fb, why in curated:
        for a in FAMILIES.get(fa, (fa,)):
            for b in FAMILIES.get(fb, (fb,)):
                if a in known and b in known:
                    add(a, b, "disjoint", "curated", "", why)

    # closure: subsets of disjoint filters are disjoint; subset is transitive
    parents: dict[str, set[str]] = defaultdict(set)
    for (a, b, rel) in list(relations):
        if rel == "subset":
            parents[a].add(b)
    changed = True
    while changed:
        changed = False
        for a in list(parents):
            for p in list(parents[a]):
                for gp in parents.get(p, ()):
                    if gp not in parents[a]:
                        parents[a].add(gp)
                        add(a, gp, "subset", "closure", "", f"via {p}")
                        changed = True
    descendants: dict[str, set[str]] = defaultdict(set)
    for a, ps in parents.items():
        for p in ps:
            descendants[p].add(a)
    for (a, b, rel) in [k for k in relations if k[2] == "disjoint"]:
        for x in {a} | descendants[a]:
            for y in {b} | descendants[b]:
                add(x, y, "disjoint", "closure", "", f"{a} vs {b}")
    rows = sorted(relations.values(), key=lambda r: (r["relation"], r["filter_a"], r["filter_b"]))
    return filter_rows, rows


def disjoint_pairs(rows) -> set[tuple[str, str]]:
    """Unordered disjoint pairs from `filter_relations` rows (either order present)."""
    out = set()
    for r in rows:
        if r["relation"] == "disjoint":
            out.add((r["filter_a"], r["filter_b"]))
            out.add((r["filter_b"], r["filter_a"]))
    return out


def filter_members(filt: str) -> str:
    """`filter%lakec|filter_inactive_and_active%lakec` -> 'lakec'; '' (any element) -> '*'; several -> joined."""
    if not filt:
        return "*"
    members = sorted({f.rsplit("%", 1)[-1] for f in filt.split("|")})
    return "|".join(members)


def filters_overlap(wkeys: set[str], rkeys: set[str], disjoint: set[tuple[str, str]]) -> bool:
    """
    Whether some write and some read may touch the same element: only known
    filters that are all pairwise disjoint rule it out ('*' = any element, and a
    multi-filter key counts as overlapping unless each of its members is disjoint).
    """
    for w in wkeys:
        for r in rkeys:
            if w == "*" or r == "*":
                return True
            if any((a, b) not in disjoint for a in w.split("|") for b in r.split("|")):
                return True
    return False
