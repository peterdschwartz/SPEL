"""Active subspaces of a routine's outputs over a scenario's input space.

For an output f (a Fortran expression per element, e.g. `t_veg(p)`) and the scenario's sampled
real inputs x (each normalized to [-1, 1] through its rule's quantile map, so a log-range input is
measured on its log scale), estimate C = E[grad f grad f^T] and its eigendecomposition. The leading
eigenvectors span the directions along which f changes most; the activity score of input i,
sum_j<=r lambda_j W_ij^2, ranks the inputs.

Gradients are forward differences in batches: the domain gets m+1 sets (m perturbed inputs per
call, chunked when there are more inputs than `max_sets - 1`); set 0 holds the base draw, set k a
copy of it with input k moved by h (backwards when the base is within h of the end of the range).
Every input (and the derive rules) is re-evaluated per set, so each element of set 0 is a base
point whose whole gradient one call (or one call per chunk) gives. All levels of a layered input
move together (one direction per variable). Inputs without a subgrid dimension (ruled parameters,
elm_drv's scalar arguments) have one value for every set, so they can't differ between sets:
`parameters=True` (`--parameters`) gives each of them a direction with one extra call per
direction and base draw. Integer inputs and discrete or fixed rules are not perturbed.

Gradients across a threshold (an `if` in the routine) are spikes: rows with non-finite values are
dropped (counted in the report); `--bootstrap` resamples base points to show how stable the
eigenvalues are.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from spel.proptest.errors import SpecError
from spel.proptest.fexpr import ExprError
from spel.proptest.inventory import Inventory
from spel.proptest.paths import data_dir, slug
from spel.proptest.rules import KINDS as RULE_KINDS
from spel.proptest.spec import Names, Scenario, Spec

SUBSPACE_DIR = "subspace"


@dataclass
class Directions:
    """The scenario inputs that get a gradient component."""
    names: list[str]  # inputs with a subgrid dimension first, then `whole`
    skipped: dict[str, str]  # name -> reason
    whole: list[str] = field(default_factory=list)  # no subgrid dimension: one call per direction


@dataclass
class SubspaceResult:
    output: str
    scenario: str
    inputs: list[str]
    skipped: dict[str, str]
    eigenvalues: np.ndarray  # descending
    eigenvectors: np.ndarray  # columns, inputs x inputs
    eig_ci: np.ndarray  # (2, m): 5% / 95% bootstrap percentiles
    rank: int  # active dimension (largest eigenvalue gap)
    activity: np.ndarray  # per input
    n_points: int
    n_dropped: int
    h: float
    gradients: Optional[np.ndarray] = None

    def to_json(self) -> dict:
        return {
            "output": self.output, "scenario": self.scenario, "inputs": self.inputs, "skipped": self.skipped,
            "eigenvalues": self.eigenvalues.tolist(), "eigenvectors": self.eigenvectors.tolist(),
            "eigenvalue_ci": self.eig_ci.tolist(), "rank": self.rank, "activity": self.activity.tolist(),
            "n_points": self.n_points, "n_dropped": self.n_dropped, "h": self.h,
            "normalization": "x = 2u - 1, u in [0, 1] the rule's quantile coordinate",
        }


def directions(scenario: Scenario, inv: Inventory, only: Optional[list[str]] = None,
               parameters: bool = False) -> Directions:
    names, whole, skipped = [], [], {}
    for name in sorted(scenario.inputs):
        rule = scenario.inputs[name]
        var = inv.vars[name]
        if only is not None and name not in only:
            continue
        fixed = [RULE_KINDS[r.kind].fixed for r in [rule, *rule.by_pft.values(), *(s for _, s in rule.by_level)]]
        if all(fixed):  # nothing continuous to step, in the rule or its overrides
            skipped[name] = fixed[0]
        elif var.type != "real":
            skipped[name] = f"{var.type} input"
        elif var.subgrid_axis is None:
            if parameters:
                whole.append(name)
            else:
                skipped[name] = "no subgrid dimension (one value for every set; --parameters steps it)"
        else:
            names.append(name)
    return Directions(names + whole, skipped, whole)


def analyze(spec: Spec, inv: Inventory, scenario: str, output: str, samples: int = 4, h: float = 1.0e-3,
            max_sets: int = 33, seed: int = 0, bootstrap: int = 200, inputs: Optional[list[str]] = None,
            fc: Optional[str] = None, parameters: bool = False, session=None) -> SubspaceResult:
    """session: a `runner.Session` factory `spec -> Session` (default: the spec's case)."""
    from spel.proptest.runner import Session

    if scenario not in spec.scenarios:
        raise SpecError(f"no scenario {scenario!r} (have {sorted(spec.scenarios)})")
    scen = spec.scenarios[scenario]
    expr = Names(inv, set(spec.functions)).expr(output, f"output {output!r}")
    only = [_canonical(inv, n) for n in inputs] if inputs else None
    from spel.proptest import spec as specmod

    live = inv.configured(specmod.constants(spec, inv, scen))  # inputs dead under the scenario's options: no direction
    scen_live = dataclasses.replace(scen, inputs={n: r for n, r in scen.inputs.items()
                                                  if live.vars[n].kind in specmod.INPUT_KINDS})
    dirs = directions(scen_live, inv, only, parameters)
    if not dirs.names:
        raise SpecError(f"[scenario.{scenario}]: no input to perturb ({dirs.skipped})")
    m = len(dirs.names)
    batched = [n for n in dirs.names if n not in dirs.whole]
    nsets = max(2, min(len(batched) + 1, max_sets))
    chunk = nsets - 1
    sspec = dataclasses.replace(spec, domain=dataclasses.replace(spec.domain, nsets=nsets))
    rng = np.random.default_rng(seed)
    rows = []
    with (session(sspec) if session is not None else Session(sspec, inv, fc=fc, namelist=scen.namelist)) as s:
        # Set j > 0 steps input j of a block; every gridcell of the set moves as a whole (all its
        # elements and levels by one step), and an output element's gradient is its gridcell's.
        layout = s.layout
        level = scen.level
        per_set = layout.per_set(level)
        n_cells = layout.per_set_gridcells
        out_cell = layout.gridcell_of(level)[:per_set]  # gridcell (within set 0) of each output element
        n = layout.count(level)
        idx = np.arange(1, n + 1)
        base_elems = np.asarray([k for k in s.elements((scen.level, scen.filter)) if k <= per_set], dtype=np.int64)
        outputs = s.needed([expr, *scen.assume])

        def evaluate(uu: dict[str, np.ndarray]) -> np.ndarray:
            ex = s.inputs_at(scen, uu)
            arrays = s.arrays(ex, s.compute(scen, ex, outputs))
            try:
                f = np.asarray(s.evaluator(arrays)(expr.node, s.locals_for(idx, scen.level), n), dtype=float)
            except ExprError as exc:
                raise SpecError(f"output {output!r}: {exc}") from None
            if f.shape != (n,):
                raise SpecError(f"output {output!r}: one value per element expected, got shape {f.shape}")
            return f

        for _ in range(samples):
            u = _base_u(s, scen, inv, rng)
            f0 = None
            grads = np.full((per_set, m), np.nan)
            for start in range(0, len(batched), chunk):
                block = batched[start:start + chunk]
                uu = {k: v.copy() for k, v in u.items()}
                steps = {}
                for j, name in enumerate(block, start=1):
                    var = inv.vars[name]
                    t = np.moveaxis(uu[name], var.subgrid_axis, 0)
                    sel = layout.set_of(var.level) == j
                    cell = layout.gridcell_of(var.level)[sel] - j * n_cells
                    elem_top = t[sel].reshape(int(sel.sum()), -1).max(axis=1)
                    top = np.full(n_cells, -np.inf)
                    np.maximum.at(top, cell, elem_top)
                    step = np.where(top + h <= 1.0, h, -h)  # one step per gridcell, all its elements and levels
                    t[sel] = np.clip(t[sel] + step[cell].reshape((len(cell),) + (1,) * (t.ndim - 1)), 0.0, 1.0)
                    steps[name] = step[out_cell]
                f = evaluate(uu)
                if f0 is None:
                    f0 = f[:per_set].copy()
                for j, name in enumerate(block, start=1):
                    fk = f[j * per_set:(j + 1) * per_set]
                    # d f / d x with x = 2u - 1
                    grads[:, dirs.names.index(name)] = (fk - f0) / steps[name] * 0.5
            if dirs.whole and f0 is None:
                f0 = evaluate(u)[:per_set].copy()
            for name in dirs.whole:  # one value for every set: its own call, the whole variable stepped
                uu = {k: v.copy() for k, v in u.items()}
                step = h if float(np.max(uu[name])) + h <= 1.0 else -h
                uu[name] = np.clip(uu[name] + step, 0.0, 1.0)
                grads[:, dirs.names.index(name)] = (evaluate(uu)[:per_set] - f0) / step * 0.5
            mask = s.assume_mask(scen, s.inputs_at(scen, u), base_elems)
            rows.append(grads[base_elems[mask] - 1])
    G = np.concatenate(rows) if rows else np.empty((0, m))
    finite = np.isfinite(G).all(axis=1)
    dropped = int((~finite).sum())
    G = G[finite]
    if G.shape[0] < 2:
        raise SpecError(f"output {output!r}: fewer than two finite gradients ({dropped} dropped)")
    lam, W = _eig(G)
    rank = _gap_rank(lam)
    activity = (W[:, :rank] ** 2 * lam[:rank]).sum(axis=1)
    ci = _bootstrap(G, bootstrap, rng)
    return SubspaceResult(output, scenario, dirs.names, dirs.skipped, lam, W, ci, rank, activity,
                          G.shape[0], dropped, h, G)


def _canonical(inv: Inventory, name: str) -> str:
    from spel.proptest.fexpr import designator, parse_expr

    path = designator(parse_expr(name))
    canon = inv.canonical(path) if path else None
    if canon is None:
        raise SpecError(f"--input {name!r} is not a variable of the case")
    return canon


def _base_u(s, scen: Scenario, inv: Inventory, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """u in [0, 1) per input element, set 0's values copied to every set."""
    out = {}
    for name in sorted(scen.inputs):
        shape, _ = s.rule_shape(name, scen.inputs[name])
        u = rng.random(shape) if shape else np.asarray(rng.random())
        var = inv.vars[name]
        axis = var.subgrid_axis
        if axis is not None and shape:
            set_of = s.layout.set_of(var.level)
            t = np.moveaxis(u, axis, 0)
            base = t[set_of == 0]
            for k in range(1, int(set_of.max()) + 1):
                t[set_of == k] = base
        out[name] = u
    return out


def _eig(G: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    C = G.T @ G / G.shape[0]
    lam, W = np.linalg.eigh(C)
    order = np.argsort(lam)[::-1]
    lam, W = np.clip(lam[order], 0.0, None), W[:, order]
    # sign convention: the largest component of each eigenvector is positive
    signs = np.sign(W[np.abs(W).argmax(axis=0), np.arange(W.shape[1])])
    return lam, W * np.where(signs == 0, 1.0, signs)


def _gap_rank(lam: np.ndarray, max_rank: int = 6) -> int:
    """The active dimension: where log(lambda_k / lambda_k+1) is largest among the first few."""
    if lam.size < 2 or lam[0] <= 0:
        return 1
    tiny = lam[0] * 1e-14
    k = min(max_rank, lam.size - 1)
    gaps = np.log(np.maximum(lam[:k], tiny)) - np.log(np.maximum(lam[1:k + 1], tiny))
    return int(np.argmax(gaps)) + 1


def _bootstrap(G: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    m = G.shape[1]
    if n <= 0:
        return np.full((2, m), np.nan)
    lams = np.empty((n, m))
    for i in range(n):
        idx = rng.integers(0, G.shape[0], G.shape[0])
        lams[i] = _eig(G[idx])[0]
    return np.percentile(lams, [5, 95], axis=0)


# -- output


def report_md(res: SubspaceResult, top: int = 10) -> str:
    lines = [
        f"# Active subspace of `{res.output}` (scenario {res.scenario})",
        "",
        f"{res.n_points} base points ({res.n_dropped} dropped: non-finite gradient), {len(res.inputs)} inputs, "
        f"forward differences h = {res.h:g} in u-space; inputs normalized to [-1, 1].",
        f"Active dimension (largest eigenvalue gap): {res.rank}.",
        "",
        "## Eigenvalues",
        "",
        "| k | eigenvalue | 5% | 95% | fraction |",
        "|---|---|---|---|---|",
    ]
    total = res.eigenvalues.sum() or 1.0
    for k, lam in enumerate(res.eigenvalues[:top]):
        lines.append(f"| {k + 1} | {lam:.4g} | {res.eig_ci[0, k]:.4g} | {res.eig_ci[1, k]:.4g} | {lam / total:.3f} |")
    lines += ["", "## Activity scores", "", "| input | activity | w1 |" + "".join(f" w{k + 2} |" for k in range(min(res.rank, 3) - 1)),
              "|---|---|---|" + "---|" * (min(res.rank, 3) - 1)]
    for i in np.argsort(res.activity)[::-1]:
        ws = "".join(f" {res.eigenvectors[i, k]:+.3f} |" for k in range(min(res.rank, 3)))
        lines.append(f"| `{res.inputs[i]}` | {res.activity[i]:.4g} |{ws}")
    if res.skipped:
        lines += ["", "## Not perturbed", ""] + [f"- `{n}`: {why}" for n, why in sorted(res.skipped.items())]
    return "\n".join(lines) + "\n"


def write(res: SubspaceResult, case_dir: Path) -> tuple[Path, Path]:
    out = data_dir(case_dir) / SUBSPACE_DIR
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{slug(res.scenario)}.{slug(res.output)}"
    j = out / f"{stem}.json"
    md = out / f"{stem}.md"
    j.write_text(json.dumps(res.to_json(), indent=1))
    md.write_text(report_md(res))
    return j, md


def load(case_dir: Path, scenario: str, output: str) -> dict:
    path = data_dir(case_dir) / SUBSPACE_DIR / f"{slug(scenario)}.{slug(output)}.json"
    if not path.is_file():
        raise SpecError(f"no subspace analysis {path} (run `spel proptest subspace`)")
    return json.loads(path.read_text())
