"""`spel proptest stats`: what a reference capture says about a case's inputs and outputs.

A design aid for scenarios: it never feeds values into a test run. For every input of the
inventory it pools the captured values over the steps and the active elements (optionally only
the patches on landunits of some site kinds, `--sites urban`, and those selected by a patch-level
Fortran expression, `--where "frac_veg_nosno(p) > 0"`; a coarser element counts when one of its
patches is selected, through the reference's index fields), and writes

- `unit-tests/proptest-data/<case>/stats/<case>.inputs.toml`: an `[inputs]` table with one rule per input (range
  from the 1st/99th percentiles, log_range when positive over decades, values for integers,
  value for inputs that never vary), units and per-pft quantiles as comments. Scenarios use it
  through `extends` and override what needs care.
- `<case>.stats.md`: the same numbers, outputs too, inputs that never vary, missing units and
  strongly correlated input pairs (candidates for `derive` rules).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from spel.proptest.domain import INPUTS_FILE, SiteKind, landunit_types
from spel.proptest.fexpr import Evaluator, ExprError, FArray, parse_expr
from spel.proptest.inventory import LEVEL_NAMES, LEVELS, NC_DIM, Inventory, VarInfo
from spel.proptest.paths import data_dir

OUTPUTS_FILE = "spel-outputs0001.nc"
SPVAL = 1.0e35
MAX_VALUES = 12  # integer inputs with at most this many distinct values get `values`
LOG_DECADES = 2.0  # positive inputs spanning more decades than this get `log_range`
CORRELATED = 0.95
INDEX = {
    "t": "topounit",
    "l": "landunit",
    "c": "column",
    "g": "gridcell",
}  # veg_pp%<x>: patch -> level


@dataclass
class VarStats:
    name: str
    kind: str
    units: str
    count: int = 0
    missing: str = ""  # why there are no values
    q: dict[float, float] = field(default_factory=dict)
    mean: float = math.nan
    std: float = math.nan
    zero: float = 0.0
    spval: float = 0.0
    distinct: Optional[list] = None
    by_pft: dict[int, tuple[float, float, int]] = field(
        default_factory=dict
    )  # pft -> (q05, q95, n)

    @property
    def constant(self) -> bool:
        return self.count > 0 and self.q[0.0] == self.q[1.0]


QUANTILES = (0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0)


class Reference:
    """Per-step access to a capture in Fortran index order (elements first)."""

    def __init__(self, path: Path):
        import netCDF4

        self.ds = netCDF4.Dataset(path)
        self.ds.set_auto_mask(False)
        self.nsteps = (
            self.ds.dimensions["time"].size if "time" in self.ds.dimensions else 1
        )
        self._cache: dict[str, np.ndarray] = {}

    def close(self) -> None:
        self.ds.close()

    def has(self, nc_name: str) -> bool:
        return nc_name in self.ds.variables

    def series(self, nc_name: str, steps: np.ndarray) -> np.ndarray:
        """(step, Fortran dims...) at `steps`; variables without a time dim are repeated."""
        key = nc_name
        if key not in self._cache:
            v = self.ds.variables[nc_name]
            data = (
                v[steps]
                if v.dimensions and v.dimensions[0] == "time"
                else np.broadcast_to(v[...], (len(steps),) + v.shape)
            )
            data = np.asarray(data)
            # netCDF holds the Fortran dims reversed after time
            self._cache[key] = np.ascontiguousarray(
                np.transpose(data, (0,) + tuple(range(data.ndim - 1, 0, -1)))
            )
        return self._cache[key]


def site_mask(ref: Reference, steps: np.ndarray, sites: set[SiteKind]) -> np.ndarray:
    """(step, patch): whether the patch's landunit is of one of these site kinds (its lun_pp%itype)."""
    for nc in ("veg_pp__landunit", "lun_pp__itype"):
        if not ref.has(nc):
            raise ExprError(f"--sites: the reference has no {nc.replace('__', '%')}")
    types = np.asarray(sorted({t for k in sites for t in landunit_types(k)}))
    lun = ref.series("veg_pp__landunit", steps)
    ltype = ref.series("lun_pp__itype", steps)
    out = np.zeros(lun.shape, dtype=bool)
    for k in range(len(steps)):
        j = lun[k] - 1
        ok = (j >= 0) & (j < ltype.shape[1])
        out[k, ok] = np.isin(ltype[k][j[ok]], types)
    return out


def element_masks(
    ref: Reference,
    inv: Inventory,
    steps: np.ndarray,
    where: Optional[str],
    sites: Optional[set[SiteKind]] = None,
) -> dict[str, np.ndarray]:
    """Per level: (step, element) mask of active elements, on landunits of `sites` and selected by
    `where` when given."""

    masks: dict[str, np.ndarray] = {}
    for lev, inst in (
        ("g", None),
        ("t", "top_pp"),
        ("l", "lun_pp"),
        ("c", "col_pp"),
        ("p", "veg_pp"),
    ):
        n = (
            ref.ds.dimensions[NC_DIM[lev]].size
            if NC_DIM[lev] in ref.ds.dimensions
            else 0
        )
        name = f"{inst}__active" if inst else None
        if name and ref.has(name):
            masks[lev] = ref.series(name, steps) != 0
        else:
            masks[lev] = np.ones((len(steps), n), dtype=bool)
    if sites:
        masks["p"] &= site_mask(ref, steps, sites)
    if where is None:
        return _coarse(ref, steps, masks) if sites else masks
    node = parse_expr(where)
    sel = np.zeros_like(masks["p"])
    n_p = sel.shape[1]
    for k, step in enumerate(steps):

        def resolve(path, k=k):
            name = inv.canonical(path)
            if name is None:
                if len(path) == 1 and inv.parameters.get(path[0]) is not None:
                    return inv.parameters[path[0]]
                raise KeyError("%".join(path))
            nc = inv.vars[name].nc_name
            if not ref.has(nc):
                raise ExprError(f"{name} is not in the reference")
            data = ref.series(nc, steps)[k]
            return FArray(data, (1,) * data.ndim, name)

        p = np.arange(1, n_p + 1)
        try:
            r = Evaluator(resolve)(node, {"p": p}, n_p)
        except ExprError as exc:
            raise ExprError(f"--where {where!r}: {exc}") from None
        sel[k] = np.broadcast_to(np.asarray(r, dtype=bool), (n_p,))
    masks["p"] &= sel
    return _coarse(ref, steps, masks)


def _coarse(
    ref: Reference, steps: np.ndarray, masks: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Coarser elements count when one of their selected patches does."""
    for lev, idx_field in INDEX.items():
        nc = f"veg_pp__{idx_field}"
        if not ref.has(nc) or lev not in masks:
            continue
        idx = ref.series(nc, steps)
        coarse = np.zeros_like(masks[lev])
        for k in range(len(steps)):
            j = idx[k][masks["p"][k]] - 1
            coarse[k, j[(j >= 0) & (j < coarse.shape[1])]] = True
        masks[lev] &= coarse
    return masks


def pooled(
    ref: Reference, var: VarInfo, steps: np.ndarray, masks: dict[str, np.ndarray]
) -> np.ndarray:
    data = ref.series(var.nc_name, steps)
    axis = var.subgrid_axis
    if axis is None:
        return data.reshape(-1)
    d = np.moveaxis(data, axis + 1, 1)  # (step, element, ...)
    m = masks[var.level]
    if m.shape[1] != d.shape[1]:
        return d.reshape(-1)
    return d[m].reshape(-1)


def var_stats(
    ref: Reference,
    var: VarInfo,
    steps: np.ndarray,
    masks: dict[str, np.ndarray],
    pft: Optional[np.ndarray],
) -> VarStats:
    s = VarStats(var.name, var.kind, var.units)
    if not ref.has(var.nc_name):
        s.missing = "not in the reference"
        return s
    vals = pooled(ref, var, steps, masks).astype(float)
    if vals.size == 0:
        s.missing = "no selected elements"
        return s
    spv = np.abs(vals) >= SPVAL
    s.spval = float(spv.mean())
    vals = vals[np.isfinite(vals) & ~spv]
    s.count = int(vals.size)
    if not s.count:
        s.missing = "only fill values"
        return s
    s.q = dict(zip(QUANTILES, np.quantile(vals, QUANTILES).tolist()))
    s.mean, s.std = float(vals.mean()), float(vals.std())
    s.zero = float((vals == 0).mean())
    uniq = np.unique(vals)
    if uniq.size <= MAX_VALUES:
        s.distinct = uniq.tolist()
    if var.level == "p" and pft is not None:
        data = np.moveaxis(ref.series(var.nc_name, steps), var.subgrid_axis + 1, 1)
        m = masks["p"]
        for p in np.unique(pft[m]):
            sel = m & (pft == p)
            v = data[sel].reshape(-1).astype(float)
            v = v[np.isfinite(v) & (np.abs(v) < SPVAL)]
            if v.size:
                lo, hi = np.quantile(v, (0.05, 0.95))
                s.by_pft[int(p)] = (float(lo), float(hi), int(v.size))
    return s


def suggest(var: VarInfo, s: VarStats) -> Optional[dict[str, Any]]:
    if not s.count:
        return None
    if s.constant:
        return {"value": _num(s.q[0.0], var.type)}
    if var.type in ("integer", "logical"):
        if s.distinct is not None:
            return {"values": [_num(v, var.type) for v in s.distinct]}
        return {"range": [_num(s.q[0.01], "integer"), _num(s.q[0.99], "integer")]}
    lo, hi = s.q[0.01], s.q[0.99]
    if lo == hi:
        lo, hi = s.q[0.0], s.q[1.0]
    if lo > 0 and math.log10(hi / lo) > LOG_DECADES:
        return {"log_range": [_round(lo), _round(hi)]}
    return {"range": [_round(lo), _round(hi)]}


def _num(v: float, ftype: str) -> Any:
    if ftype in ("integer", "logical"):
        return int(round(v))
    return _round(v)


def _round(v: float) -> float:
    return float(f"{v:.6g}")


def correlations(
    ref: Reference,
    inv: Inventory,
    names: list[str],
    steps: np.ndarray,
    masks: dict[str, np.ndarray],
) -> list[tuple[str, str, float]]:
    by_level: dict[str, dict[str, np.ndarray]] = {}
    for n in names:
        v = inv.vars[n]
        if v.rank != 1 or v.level is None or v.type != "real" or not ref.has(v.nc_name):
            continue
        x = pooled(ref, v, steps, masks).astype(float)
        if (
            x.size > 2
            and np.isfinite(x).all()
            and (np.abs(x) < SPVAL).all()
            and x.std() > 0
        ):
            by_level.setdefault(v.level, {})[n] = x
    out = []
    for vs in by_level.values():
        keys = sorted(vs)
        for i, a in enumerate(keys):
            for b in keys[i + 1 :]:
                if vs[a].size != vs[b].size:
                    continue
                r = float(np.corrcoef(vs[a], vs[b])[0, 1])
                if abs(r) >= CORRELATED:
                    out.append((a, b, r))
    return sorted(out, key=lambda t: -abs(t[2]))


@dataclass
class StatsResult:
    inputs: list[VarStats]
    outputs: list[VarStats]
    correlated: list[tuple[str, str, float]]
    steps: int
    where: Optional[str]
    reference: Path
    # pft -> selected patches (over the steps)
    pfts: dict[int, int] = field(default_factory=dict)
    sites: Optional[set[SiteKind]] = None
    # lun_pp%itype -> selected landunits (over the steps)
    landunits: dict[int, int] = field(default_factory=dict)

    def selection(self) -> str:
        """What the statistics cover, for headers."""
        parts = []
        if self.sites:
            parts.append("sites " + ", ".join(sorted(k.value for k in self.sites)))
        if self.where:
            parts.append(f"where {self.where}")
        return "; ".join(parts)


def compute(
    inv: Inventory,
    reference: Path,
    where: Optional[str] = None,
    stride: int = 1,
    constants: Optional[dict[str, Any]] = None,
    sites: Optional[set[SiteKind]] = None,
) -> StatsResult:
    """
    Computes statistics on the input and output space of the function of interest.
    - inv: Inventory
    """
    ref = Reference(reference / INPUTS_FILE)
    try:
        steps = np.arange(0, ref.nsteps, max(1, stride))
        masks = element_masks(ref, inv, steps, where, sites)
        pft = ref.series("veg_pp__itype", steps) if ref.has("veg_pp__itype") else None
        consts = constants or {}
        ins = [
            v
            for v in inv.of_kind("input", "inout")
            if v.instance != "bounds" and (not consts or inv.guard_active(v, consts))
        ]
        inputs = [var_stats(ref, v, steps, masks, pft) for v in ins]
        corr = correlations(ref, inv, [v.name for v in ins], steps, masks)
        pfts: dict[int, int] = {}
        if pft is not None and pft.shape == masks["p"].shape:
            kinds, counts = np.unique(pft[masks["p"]], return_counts=True)
            pfts = {int(k): int(c) for k, c in zip(kinds, counts)}
        landunits: dict[int, int] = {}
        if ref.has("lun_pp__itype"):
            ltype = ref.series("lun_pp__itype", steps)
            if ltype.shape == masks["l"].shape:
                kinds, counts = np.unique(ltype[masks["l"]], return_counts=True)
                landunits = {int(k): int(c) for k, c in zip(kinds, counts)}
    finally:
        ref.close()
    outputs = []
    out_path = reference / OUTPUTS_FILE
    if out_path.exists():
        ref = Reference(out_path)
        try:
            # same steps as the inputs: same element masks
            if ref.nsteps >= int(steps[-1]) + 1:
                for v in inv.of_kind("output", "inout"):
                    if v.instance != "bounds" and (
                        not consts or inv.guard_active(v, consts)
                    ):
                        outputs.append(var_stats(ref, v, steps, masks, pft))
        finally:
            ref.close()
    return StatsResult(
        inputs,
        outputs,
        corr,
        len(steps),
        where,
        reference,
        pfts,
        set(sites or ()) or None,
        landunits,
    )


# -- writing


def _toml_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    if isinstance(v, float):
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return repr(v)
    return str(v)


def rules_toml(inv: Inventory, res: StatsResult, case: str) -> str:
    lines = [
        f"# Input rules for {case} from the reference {res.reference.name} ({res.steps} steps"
        + (f"; {res.selection()}" if res.selection() else "")
        + ").",
        "# Generated by `spel proptest stats`: a starting point. Ranges are the 1st-99th percentiles of the",
        "# captured values; check each one and make dependent inputs `derive` rules in the scenario.",
        "",
        "[inputs]",
    ]
    for s in sorted(res.inputs, key=lambda s: s.name):
        var = inv.vars[s.name]
        dims = ", ".join(d.text for d in var.dims)
        head = f"# {s.name}({dims}) {var.type}" + (
            f" [{s.units}]" if s.units else " [units?]"
        )
        if var.long_name:
            head += f" {var.long_name}"
        lines.append(head)
        rule = suggest(var, s)
        if rule is None:
            lines.append(f"# no rule: {s.missing}")
            lines.append(f'# "{s.name}" = {{ range = [?, ?] }}')
            lines.append("")
            continue
        q = s.q
        lines.append(
            f"#   n={s.count} min={q[0.0]:.6g} p50={q[0.5]:.6g} max={q[1.0]:.6g} "
            f"mean={s.mean:.6g} std={s.std:.6g}"
            + (f" zero={s.zero:.0%}" if s.zero else "")
            + (f" spval={s.spval:.0%}" if s.spval else "")
        )
        for p, (lo, hi, n) in sorted(s.by_pft.items()):
            lines.append(f"#   pft {p:2d}: p5..p95 = {lo:.6g} .. {hi:.6g} (n={n})")
        body = ", ".join(f"{k} = {_toml_value(v)}" for k, v in rule.items())
        lines.append(f'"{s.name}" = {{ {body} }}')
        lines.append("")
    return "\n".join(lines)


def report_md(inv: Inventory, res: StatsResult, case: str) -> str:
    def table(items: list[VarStats]) -> list[str]:
        out = [
            "| variable | units | n | min | p1 | p50 | p99 | max | zero | spval |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for s in sorted(items, key=lambda s: s.name):
            if not s.count:
                out.append(f"| `{s.name}` | {s.units} | 0 | {s.missing} | | | | | | |")
                continue
            q = s.q
            out.append(
                f"| `{s.name}` | {s.units} | {s.count} | {q[0.0]:.4g} | {q[0.01]:.4g} | {q[0.5]:.4g} | "
                f"{q[0.99]:.4g} | {q[1.0]:.4g} | {s.zero:.0%} | {s.spval:.0%} |"
            )
        return out

    mask_str = f"Applied mask: {res.where}" if res.where else ""
    lines = [
        f"# Reference statistics: {case}",
        "",
        f"Reference `{res.reference}`, {res.steps} steps"
        + (f"; {res.selection()}" if res.selection() else "")
        + mask_str,
        "",
        "## Inputs",
        "",
        *table(res.inputs),
        "",
    ]
    const = [s.name for s in res.inputs if s.constant]
    if const:
        lines += [
            "## Inputs that never vary in the reference",
            "",
            ", ".join(f"`{n}`" for n in sorted(const)),
            "",
        ]
    nounits = sorted(s.name for s in [*res.inputs, *res.outputs] if not s.units)
    if nounits:
        lines += [
            "## Without units (add them under `[units]`)",
            "",
            ", ".join(f"`{n}`" for n in nounits),
            "",
        ]
    if res.correlated:
        lines += [
            f"## Correlated inputs (|r| >= {CORRELATED}, same level): `derive` candidates",
            "",
            "| a | b | r |",
            "|---|---|---|",
        ]
        lines += [f"| `{a}` | `{b}` | {r:.3f} |" for a, b, r in res.correlated]
        lines.append("")
    if res.outputs:
        lines += ["## Outputs", "", *table(res.outputs), ""]
    lines += ["Levels: " + ", ".join(f"{k} = {LEVEL_NAMES[k]}" for k in LEVELS), ""]
    return "\n".join(lines)


def write(inv: Inventory, res: StatsResult, case_dir: Path) -> tuple[Path, Path]:
    out = data_dir(case_dir) / "stats"
    out.mkdir(parents=True, exist_ok=True)
    case = Path(case_dir).name
    rules = out / f"{case}.inputs.toml"
    md = out / f"{case}.stats.md"
    rules.write_text(rules_toml(inv, res, case))
    md.write_text(report_md(inv, res, case))
    return rules, md
