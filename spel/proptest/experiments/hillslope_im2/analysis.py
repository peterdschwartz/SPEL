"""Analyses of the IM2 hillslope experiment (hillslope_im2.ipynb), as named functions: each takes the
case, spec, session, layout or result it needs and returns a table, an array or text."""

from __future__ import annotations

import copy
import re
import tomllib
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from spel.proptest import coupling, runner
from spel.proptest.hierarchy import Gridcell, HierarchyError, Layout

ROUTED_TOPOUNIT_STORE = "top_ws%from_uphill"
RUNOFF = "col_wf%qflx_runoff"


def boundary_table(
    case, namelist: dict, show: Optional[Iterable[str]] = None
) -> pd.DataFrame:
    """The variables the case's routines read that the rest of ELM's step writes (coupling.boundary,
    from elm_drv's static trace), with how many writers and where; `show`: only these rows.
    """
    values = {k.lower(): v for k, v in namelist.items()}
    live = case.inv.configured(values)
    found = coupling.boundary(case.inv, values)
    rows = [
        dict(
            variable=n,
            kind=live.vars[n].kind,
            writers=len(w),
            where=coupling.describe(w, 1),
        )
        for n, w in sorted(found.items())
        if live.vars[n].kind in ("input", "inout")
    ]
    table = pd.DataFrame(rows).set_index("variable")
    return table.loc[list(show)] if show is not None else table


def boundary_declarations(spec, boundary: Iterable[str]) -> pd.DataFrame:
    """Per scenario of a validated spec, how it states each boundary variable: forcing (set before
    every step), between (assigned after every step), keep (left to the process)."""
    rows = []
    for name, sc in spec.scenarios.items():
        counts = {"forcing": 0, "between": 0, "keep": 0, "UNDECLARED": 0}
        for n in boundary:
            if n in sc.forcing or any(v == n for v, _ in sc.forcing_exprs):
                counts["forcing"] += 1
            elif any(v == n for v, _ in sc.between):
                counts["between"] += 1
            elif sc.boundary_declared(n):
                counts["keep"] += 1
            else:
                counts["UNDECLARED"] += 1
        rows.append(dict(scenario=name, steps=sc.steps, **counts))
    return pd.DataFrame(rows).set_index("scenario")


# -- outcomes of a run


def _domain_number(outcome) -> Optional[int]:
    m = re.match(r"domain (\d+)/", outcome.domain or "")
    return int(m.group(1)) if m else None


def check_table(spec) -> pd.DataFrame:
    """A validated spec's checks: kind, scenarios and expression."""
    return pd.DataFrame(
        [
            dict(
                check=c.name,
                kind=c.table,
                scenarios=", ".join(c.scenarios),
                expression=c.judged.text,
            )
            for c in spec.checks
        ]
    ).set_index("check")


def outcome_table(outcomes) -> pd.DataFrame:
    """pass/FAIL per check: one column per scenario, or per drawn domain for a run on drawn domains."""
    rows = [
        dict(
            check=o.check,
            where=(f"domain {_domain_number(o)}" if _domain_number(o) else o.scenario),
            result="pass" if o.passed else "FAIL",
        )
        for o in outcomes
    ]
    return (
        pd.DataFrame(rows)
        .pivot(index="check", columns="where", values="result")
        .fillna("")
    )


def compare_outcomes(**runs) -> pd.DataFrame:
    """pass/FAIL per check and drawn domain (or scenario) of several runs side by side, e.g.
    `compare_outcomes(elm=outcomes, intended=fixed)`."""
    return pd.concat(
        {label: outcome_table(o) for label, o in runs.items()}, axis=1
    ).fillna("")


def failure_summaries(outcomes: list[runner.Outcome], lines: int = 7) -> str:
    """The first lines of every failure's report."""
    return "\n\n".join(
        "\n".join(o.message.splitlines()[:lines]) for o in outcomes if not o.passed
    )


def _landunits(topounit) -> str:
    return "+".join(
        lu.label + ("(0)" if lu.weight == 0.0 else "") for lu in topounit.landunits
    )


def domain_label(gridcells) -> str:
    """A domain in one line: its gridcells' topounits (`|`) and landunits (`+`, `(0)`: weight 0,
    `[w0]`: a topounit of weight 0)."""
    return "; ".join(
        f"{g.name}: "
        + " | ".join(
            _landunits(t) + (" [w0]" if t.weight == 0.0 else "") for t in g.topounits
        )
        for g in gridcells
    )


def shrunk_gridcells(outcome) -> list[Gridcell]:
    """The gridcells of a failure's saved domain (`domain.toml`, written for drawn domains)."""
    table = tomllib.loads((Path(outcome.failure) / runner.DOMAIN_FILE).read_text())
    return [
        Gridcell.from_table(g, f"{outcome.failure} gridcell {k}")
        for k, g in enumerate(table["gridcell"])
    ]


def shrunk_domain_layout(outcome) -> Layout:
    """A failure's shrunk domain laid out (one set), for `topounit_table`."""
    return Layout(shrunk_gridcells(outcome))


def shrunk_domains(outcomes) -> pd.DataFrame:
    """Every failure of a drawn-domain run: the domain it was drawn in, how it failed (its
    signature) and the smallest domain it still fails on, in that way."""
    rows = []
    for o in outcomes:
        if o.passed or not o.failure:
            continue
        rows.append(
            dict(
                domain=_domain_number(o),
                check=o.check,
                signature=o.signature.text() if o.signature is not None else "",
                shrunk_to=domain_label(shrunk_gridcells(o)),
            )
        )
    return pd.DataFrame(rows).set_index(["domain", "check"])


def topounit_table(layout: Layout) -> pd.DataFrame:
    """The first set's topounits: gridcell, number, weight, elevation, landunits, the downhill
    neighbour the domain gives the routines (`downhill_ti`, its number in the gridcell), how many
    topounits send to it, and what is wrong with the neighbour."""
    tt = layout.table("t")
    tt["landunits"] = [_landunits(t) for t in layout.topounits]
    try:
        dn = layout.downhill()
    except HierarchyError:  # a gridcell of several topounits without elevations
        return tt[tt["set"] == 0][
            ["name", "topo_number", "weight", "active", "landunits"]
        ]
    tt["downhill_ti"] = dn
    tt["points at"] = layout.downhill(local=True)
    tt["uphill_count"] = np.bincount(
        [d - 1 for t, d in enumerate(dn) if d > 0 and d - 1 != t], minlength=len(dn)
    )
    tt["problem"] = layout.downhill_problems()
    cols = [
        "name",
        "topo_number",
        "weight",
        "active",
        "elevation",
        "landunits",
        "downhill_ti",
        "points at",
        "uphill_count",
        "problem",
    ]
    return tt[tt["set"] == 0][[c for c in cols if c in tt]]


def first_failures(session, scenario: str, checks, example) -> pd.DataFrame:
    """For each check, the step it first fails at on `example` and the elements that fail then, by
    hillslope and topounit."""
    sc = session.spec.scenarios[scenario]
    rows = []
    for check in checks:
        v = session.evaluate(sc, check, check.prepare(session, sc, example))
        tab = session.layout.table(v.level)
        where = []
        for k in v.failing:
            row = tab.loc[int(k)]
            topo = f" t{row['topo_number']}" if "topo_number" in tab else ""
            where.append(f"{row['name']}{topo} (set {row['set']})")
        rows.append(
            dict(
                check=check.name,
                first_failing_step=v.step if len(v.failing) else None,
                failing=len(v.failing),
                where=", ".join(sorted(set(where))),
            )
        )
    return pd.DataFrame(rows).set_index("check")


# -- the code behind the confirmed hypotheses


def source_excerpt(
    srcroot: Path, relpath: str, marker: str, before: int = 0, lines: int = 12
) -> str:
    """Lines of an ELM source file (with their file line numbers) around the first line containing
    `marker` (plain text: it only places the excerpt)."""
    src = (Path(srcroot) / relpath).read_text().splitlines()
    k = next(i for i, line in enumerate(src) if marker in line) - before
    return "\n".join(
        f"{k + 1 + j:5d}  {line}" for j, line in enumerate(src[k : k + lines])
    )


def set_topounit_source(srcroot: Path) -> str:
    """initGridCellsMod's downhill-neighbour search (set_topounit): `dn_index` is reset once per
    gridcell, before the loop over its topounits."""
    return source_excerpt(
        srcroot,
        "components/elm/src/main/initGridCellsMod.F90",
        "find the closest downhill neighbor",
        lines=13,
    )


def send_to_downhill_source(srcroot: Path) -> str:
    """HydrologyDrainage's send: 
    the water is scaled by the target topounit's weight in the gridcell (0 for an inactive target).
    """
    return source_excerpt(
        srcroot,
        "components/elm/src/biogeophys/HydrologyDrainageMod.F90",
        "temp_mass = (qflx_to_downhill(c) * dtime)",
        before=6,
        lines=9,
    )


def release_from_store_source(srcroot: Path) -> str:
    """SurfaceRunoff's release: each hydrology column's share is its weight over the summed weight
    of the topounit's hydrology columns (0 when they all weigh 0)."""
    return source_excerpt(
        srcroot,
        "components/elm/src/biogeophys/SoilHydrologyMod.F90",
        "top_pp%uphill_wt(t) = 0._r8",
        before=2,
        lines=17,
    )


def downhill_table(layout: Layout) -> pd.DataFrame:
    """Per hillslope of the first set: its elevations, the downhill neighbours ELM's set_topounit
    gives them and the intended ones (numbers within the gridcell, -1: none), and what is wrong with
    ELM's."""
    tt = layout.table("t")
    tt["elm"] = layout.downhill(rule="elm", local=True)
    tt["intended"] = layout.downhill(rule="intended", local=True)
    tt["problem"] = layout.downhill_problems(rule="elm")

    def summary(g: pd.DataFrame) -> pd.Series:
        bad = [f"{n} {p}" for n, p in zip(g["topo_number"], g["problem"]) if p]
        return pd.Series(
            {
                "elevations": g["elevation"].tolist(),
                "elm": g["elm"].tolist(),
                "intended": g["intended"].tolist(),
                "problem": ", ".join(bad) or "ok",
            }
        )

    return (
        tt[tt["set"] == 0]
        .groupby("name", sort=False)
        .apply(summary)
        .rename_axis("hillslope")
    )


def captured_downhill(case) -> pd.DataFrame:
    """ELM's top_pp%downhill_ti in the case's reference capture (a real run), one row per topounit,
    with the gridcell's global index and coordinates."""
    ref = case.reference_inputs()
    if ref is None:
        raise FileNotFoundError(f"{case.name}: no reference input data")
    with ref:
        dn = np.asarray(ref["top_pp__downhill_ti"][0])
        k = np.asarray(ref["top_pp__topo_grc_ind"][0])
        w = np.asarray(ref["top_pp__wtgcell"][0])
        g = np.cumsum(
            k == 1
        )  # topo_grc_ind restarts at 1 in each gridcell: g = each topounit's gridcell
        out = pd.DataFrame(
            {
                "gindex": np.asarray(ref.spel_subset_gindex)[g - 1],
                "lat": np.asarray(ref.spel_subset_latdeg)[g - 1],
                "lon": np.asarray(ref.spel_subset_londeg)[g - 1],
                "topounit": np.arange(1, len(dn) + 1),
                "number in gridcell": k,
                "weight": w,
                "downhill_ti": dn,
            }
        )
    out["points at itself"] = out["downhill_ti"] == out["topounit"]
    return out


def store_series(
    result,
    layout: Layout,
    hillslopes: Iterable[str],
    every: int = 4,
) -> pd.DataFrame:
    """Each topounit's store (`top_ws%from_uphill`, kg/m2 of topounit) of the named hillslopes of
    the first set, every `every` steps."""
    store = result.series(ROUTED_TOPOUNIT_STORE)
    tt = layout.table("t")
    want = set(hillslopes)
    cols = {
        f"{n} t{k}": i
        for i, (n, k, st) in enumerate(zip(tt["name"], tt["topo_number"], tt["set"]))
        if st == 0 and n in want
    }
    frame = pd.DataFrame(
        {c: store[:, i] for c, i in cols.items()},
        index=pd.RangeIndex(1, store.shape[0] + 1, name="step"),
    )
    return frame.iloc[every - 1 :: every]


def river_runoff(result, layout: Layout, dtime: float) -> np.ndarray:
    """Each gridcell's runoff to the river so far (kg/m2 of gridcell), after each step: shape
    (steps, gridcells)."""
    w_c, g_c = layout.weight("c", "g"), layout.ancestor_of("c", "g")
    per_step = result.series(RUNOFF) * w_c * dtime
    per_g = np.stack([np.bincount(g_c, row, layout.count("g")) for row in per_step])
    return np.cumsum(per_g, axis=0)


def river_table(on, off, layout: Layout, dtime: float) -> pd.DataFrame:
    """The first set's gridcells: the day's runoff to the river (kg/m2 of gridcell) with IM2 on and
    off, and their ratio."""
    gt = layout.table("g")
    r_on, r_off = river_runoff(on, layout, dtime), river_runoff(off, layout, dtime)
    last = pd.DataFrame(
        {
            "hillslope": gt["name"],
            "set": gt["set"],
            "IM2 on": r_on[-1],
            "IM2 off": r_off[-1],
        }
    )
    last["on / off"] = last["IM2 on"] / last["IM2 off"]
    return last[last["set"] == 0].drop(columns="set").set_index("hillslope")


def mirror_example(example, layout: Layout, inv, src: str, dst: str):
    """A copy of `example` in which hillslope `dst` has the drawn inputs of hillslope `src`, element
    by element of the same elevation (in each set): the same hillslope listed in another order.
    """
    out = copy.deepcopy(example)
    tt, ct = layout.table("t"), layout.table("c")
    elev = tt["elevation"].to_numpy()
    maps = {}
    for level, table in (("t", tt), ("c", ct)):
        e = elev if level == "t" else elev[layout.ancestor_of("c", "t")]
        idx = np.arange(len(table))
        m = {}
        for k in sorted(set(table["set"])):
            a = idx[(table["name"] == src).to_numpy() & (table["set"] == k).to_numpy()]
            b = idx[(table["name"] == dst).to_numpy() & (table["set"] == k).to_numpy()]
            m.update({j: a[np.nonzero(e[a] == e[j])[0][0]] for j in b})
        maps[level] = m
    for name, arr in out.inputs.items():
        var = inv.vars.get(name)
        if var is None or var.level not in maps:
            continue
        x = np.moveaxis(np.asarray(arr), var.subgrid_axis, 0)
        for j, i in maps[var.level].items():
            x[j] = x[i]
        out.inputs[name] = np.moveaxis(x, 0, var.subgrid_axis)
    return out


def mirrored_storms(
    session,
    off_session,
    scenario: str = "storm",
    seeds: Iterable[int] = range(20),
    src: str = "lowest first",
    dst: str = "lowest last",
    dtime: float = 1800.0,
) -> pd.DataFrame:
    """Storms drawn from `seeds`, each given to `dst` as `src`'s inputs (mirror_example), run with IM2
    on (`session`) and off (`off_session`). Per twin: the day's river runoff on and off, and the step
    by which half of it had arrived."""
    layout = session.layout
    gt = layout.table("g")
    twins = np.nonzero(gt["name"].isin([src, dst]).to_numpy())[0]
    rows = []
    for seed in seeds:
        ex = mirror_example(
            session.sample(scenario, seed=seed), layout, session.inv, src, dst
        )
        on = river_runoff(session.execute(scenario, ex), layout, dtime)
        off = river_runoff(off_session.execute(scenario, ex), layout, dtime)
        half_on = (on >= 0.5 * on[-1]).argmax(axis=0) + 1
        half_off = (off >= 0.5 * off[-1]).argmax(axis=0) + 1
        rows += [
            dict(
                seed=seed,
                set=gt["set"].iloc[i],
                hillslope=gt["name"].iloc[i],
                on=on[-1, i],
                off=off[-1, i],
                half_on=half_on[i],
                half_off=half_off[i],
            )
            for i in twins
        ]
    storms = pd.DataFrame(rows)
    storms["on / off"] = storms["on"] / storms["off"]
    return storms


def twins_agree_without_im2(
    storms: pd.DataFrame, src: str = "lowest first", dst: str = "lowest last"
) -> bool:
    """Whether the mirrored twins give the same river runoff with IM2 off (to rounding): then their
    difference with IM2 on is the topology alone."""
    a = storms[storms.hillslope == src]["off"].to_numpy()
    b = storms[storms.hillslope == dst]["off"].to_numpy()
    return bool(np.allclose(a, b, rtol=1e-12, atol=1e-15))


def storm_summary(storms: pd.DataFrame) -> pd.DataFrame:
    """Mean, min and max per hillslope of the day's river runoff (on, off, on/off) and of the step by
    which half of it had arrived."""
    return (
        storms.groupby("hillslope")[["off", "on", "on / off", "half_off", "half_on"]]
        .describe()
        .loc[:, (slice(None), ["mean", "min", "max"])]
        .T
    )
