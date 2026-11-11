"""What a session can show, to make results checkable by eye.

- `BuiltDomain` (`Session.built()`): the state the Fortran case built at init (the filters
  setFilters made, the topology values, the bounds of every allocation), and `check()` against
  what the declaration says (`Domain.topology`, `Domain.extents`, `domain.expected_filters`).
- `Result` (`Session.execute(scenario, example)`): one run's inputs and outputs, as pandas tables
  per level (`frame`), with ancestors' variables joined and descendants' aggregated.
- `Verdict` (`Session.evaluate(...)`, `Result.verdict(check)`): a check's per-element outcome
  and the values its expression used, without raising.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from spel.proptest.domain import expected_filters
from spel.proptest.fexpr import FArray
from spel.proptest.hierarchy import LEVEL_NAMES, LEVELS

if TYPE_CHECKING:
    import pandas as pd

    from spel.proptest.checks import Check
    from spel.proptest.runner import Example, Session
    from spel.proptest.spec import Scenario


@dataclass(frozen=True)
class Mismatch:
    what: str  # "filter urbanp", "topology lun_pp%coli", "bounds lun_es%taf"
    expected: Any
    got: Any

    def __str__(self) -> str:
        return f"{self.what}: expected {_short(self.expected)}, the case has {_short(self.got)}"


@dataclass
class BuiltDomain:
    filters: dict[str, np.ndarray]  # member -> Fortran indices setFilters put in it
    topology: dict[str, np.ndarray]  # name -> the values the case holds
    bounds: dict[str, list[tuple[int, int]]]  # name -> (lb, ub) per dimension, as allocated
    expected_topology: dict[str, np.ndarray]
    expected_bounds: dict[str, list[tuple[int, int]]]
    expected_filters: dict[str, np.ndarray]
    unchecked_filters: list[str] = field(default_factory=list)  # built, but not predicted

    def check(self) -> list[Mismatch]:
        out = []
        for name, want in self.expected_topology.items():
            got = self.topology.get(name)
            if got is None:
                out.append(Mismatch(f"topology {name}", "allocated", "not allocated"))
            elif not _same(want, got):
                out.append(Mismatch(f"topology {name}", want, got))
        for name, want in self.expected_bounds.items():
            got = self.bounds.get(name)
            if got is not None and [tuple(b) for b in got] != [tuple(b) for b in want]:
                out.append(Mismatch(f"bounds {name}", want, got))
        for member, want in self.expected_filters.items():
            got = self.filters.get(member)
            if got is not None and not np.array_equal(np.asarray(want), np.asarray(got)):
                out.append(Mismatch(f"filter {member}", want, got))
        return out

    def assert_consistent(self) -> None:
        bad = self.check()
        if bad:
            raise AssertionError("the case's domain differs from the declaration:\n  "
                                 + "\n  ".join(str(m) for m in bad))


def built(session: "Session") -> BuiltDomain:
    """The domain as the case holds it after init (Session.built)."""
    inv, dom = session.inv, session.domain
    members = sorted(m for m, kind in inv.filters.items() if kind == 1 and f"num_{m}" in inv.filters)
    names = [f"filter%num_{m}" for m in members] + [f"filter%{m}" for m in members]
    allocated = [n for n in dom.extents if n in inv.vars and not inv.vars[n].driver]
    session.statics(sorted(set(names) | set(dom.topology) | set(allocated)))
    filters = {}
    for m in members:
        num, arr = session.static(f"filter%num_{m}"), session.static(f"filter%{m}")
        if num is None or arr is None:
            continue
        filters[m] = np.asarray(arr.data[: int(num.data)], dtype=np.int64)
    topology = {n: np.asarray(fa.data) for n in dom.topology if (fa := session.static(n)) is not None}
    bounds = {}
    for n in allocated:
        fa = session.static(n)
        if fa is not None and fa.data.ndim:
            bounds[n] = [(lb, lb + s - 1) for lb, s in zip(fa.lb, fa.data.shape)]
    expected = expected_filters(dom.layout)
    return BuiltDomain(
        filters=filters, topology=topology, bounds=bounds,
        expected_topology=dict(dom.topology),
        expected_bounds={n: dom.extents[n] for n in allocated if dom.extents.get(n)},
        expected_filters={m: v for m, v in expected.items() if m in filters},
        unchecked_filters=sorted(m for m in filters if m not in expected),
    )


@dataclass
class Verdict:
    """A check's outcome on one example: per judged element whether it holds."""

    check: str
    level: str
    elements: np.ndarray  # Fortran indices judged
    mask: np.ndarray  # passes `assume`
    ok: Optional[np.ndarray]  # None when the routine aborted
    values: dict[str, np.ndarray] = field(default_factory=dict)  # what the expression used, per element
    aborted: str = ""

    @property
    def passed(self) -> bool:
        return not self.aborted and self.ok is not None and bool(np.all(self.ok | ~self.mask))

    @property
    def failing(self) -> np.ndarray:
        if self.ok is None:
            return self.elements
        return self.elements[self.mask & ~self.ok]

    def frame(self) -> "pd.DataFrame":
        import pandas as pd

        rows: dict[str, Any] = {"assumed": self.mask}
        if self.ok is not None:
            rows["ok"] = self.ok
        for name, v in self.values.items():
            v = np.asarray(v)
            if v.shape[:1] == self.elements.shape:
                rows[name] = list(v) if v.ndim > 1 else v
        return pd.DataFrame(rows, index=pd.Index(self.elements, name=LEVEL_NAMES[self.level]))


@dataclass
class Result:
    """One run of an example: the inputs it set and the state it fetched afterwards."""

    session: "Session"
    scenario: "Scenario"
    example: "Example"
    arrays: dict[str, FArray]  # inputs (before) and fetched state (after) by name

    def frame(self, level: str = "p", names: Optional[list[str]] = None,
              aggregate: Optional[dict[str, str]] = None) -> "pd.DataFrame":
        """One row per element of `level`: the layout's description (hierarchy.Layout.table),
        then every variable of that level or a coarser one (an ancestor's value on each row),
        layers as `name[k]` (Fortran index). `aggregate` adds finer variables reduced over each
        element's descendants: "sum", "mean", "min", "max", or "wsum" (weighted by the
        descendants' weights relative to this level, as ELM's p2c/c2l do)."""
        import pandas as pd

        lay = self.session.layout
        df = lay.table(level)
        inv = self.session.inv
        chosen = names if names is not None else sorted(self.arrays)
        cols: dict[str, Any] = {}
        for name in chosen:
            var = inv.vars.get(name)
            fa = self.arrays.get(name)
            if var is None or fa is None or var.level is None:
                continue
            if LEVELS.index(var.level) > LEVELS.index(level):
                continue  # finer: only through `aggregate`
            rows = lay.ancestor_of(level, var.level)
            data = np.moveaxis(np.asarray(fa.data), var.subgrid_axis, 0)[rows]
            cols.update(_columns(name, data, fa.lb, var.subgrid_axis))
        for name, how in (aggregate or {}).items():
            var = inv.vars[name]
            fa = self.arrays.get(name)
            if fa is None:
                raise KeyError(f"{name} is not in this result (pass it to execute's outputs)")
            if LEVELS.index(var.level) <= LEVELS.index(level) or np.asarray(fa.data).ndim != 1:
                raise ValueError(f"aggregate: {name} is not a one-dimensional variable of a finer level")
            cols[f"{how}({name})"] = _aggregate(lay, np.asarray(fa.data, dtype=float), var.level, level, how)
        return pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1)

    def verdict(self, check: "Check | str") -> Verdict:
        """The check on this result's example (prepared as a test run prepares it)."""
        s = self.session
        chk = check if not isinstance(check, str) else next(c for c in s.spec.checks if c.name == check)
        ex = chk.prepare(s, self.scenario, self.example)
        return s.evaluate(self.scenario, chk, ex)


def _columns(name: str, data: np.ndarray, lb: tuple[int, ...], axis: int) -> dict[str, Any]:
    if data.ndim == 1:
        return {name: data}
    rest = [l for k, l in enumerate(lb) if k != axis]
    flat = data.reshape(data.shape[0], -1)
    idx = np.indices(data.shape[1:]).reshape(len(data.shape) - 1, -1).T
    return {f"{name}[{', '.join(str(i + l) for i, l in zip(ix, rest))}]": flat[:, j] for j, ix in enumerate(idx)}


def _aggregate(lay, values: np.ndarray, fine: str, level: str, how: str) -> np.ndarray:
    owner = lay.ancestor_of(fine, level)
    n = lay.count(level)
    if how == "wsum":
        return np.bincount(owner, weights=values * lay.weight(fine, level), minlength=n)
    if how == "sum":
        return np.bincount(owner, weights=values, minlength=n)
    if how == "mean":
        return np.bincount(owner, weights=values, minlength=n) / np.bincount(owner, minlength=n)
    if how in ("min", "max"):
        out = np.full(n, np.inf if how == "min" else -np.inf)
        (np.minimum if how == "min" else np.maximum).at(out, owner, values)
        return out
    raise ValueError(f"aggregate: unknown reduction {how!r} (sum, wsum, mean, min, max)")


def _same(want: np.ndarray, got: np.ndarray) -> bool:
    want, got = np.asarray(want), np.asarray(got)
    if want.shape != got.shape:
        return False
    if want.dtype == bool:
        return bool(np.array_equal(want, got.astype(bool)))
    if np.issubdtype(want.dtype, np.floating):
        return bool(np.allclose(want, got, rtol=0, atol=0, equal_nan=True))
    return bool(np.array_equal(want, got))


def _short(v: Any) -> str:
    a = np.asarray(v) if not isinstance(v, (str, list)) else v
    if isinstance(a, np.ndarray) and a.size > 12:
        return f"{a.ravel()[:12].tolist()}... ({a.size} values)"
    return str(a.tolist() if isinstance(a, np.ndarray) else a)
