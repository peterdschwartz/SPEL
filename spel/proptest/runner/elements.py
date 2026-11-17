"""Where things are in a session's synthetic domain (`ElementSpace`): the elements of each level and
how they relate, the shapes and Fortran bounds of the case's variables, values placed into those
arrays, and the values a run does not change (topology, parameters, constants, filters), read once
from the backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np

from spel.proptest.domain import SynthDomain
from spel.proptest.errors import SpecError
from spel.proptest.fexpr import ExprError, FArray
from spel.proptest.hierarchy import Layout
from spel.proptest.inventory import LEVELS, Inventory
from spel.proptest.spec import ELEMENT_VARS

if TYPE_CHECKING:
    from spel.proptest.runner.drawing import Example


class ElementSpace:
    """The layout, the built domain and the inventory of one session, and its backend for the
    values that do not change."""

    def __init__(self, layout: Layout, domain: SynthDomain, inv: Inventory, backend: Any):
        self.layout, self.domain, self.inv, self.backend = layout, domain, inv, backend
        self._canon: dict[tuple[str, ...], Optional[str]] = {}
        self._static: dict[str, Optional[FArray]] = {}
        self._domain_fns: Optional[dict[str, Callable]] = None

    def canonical(self, path: tuple[str, ...]) -> Optional[str]:
        if path not in self._canon:
            self._canon[path] = self.inv.canonical(list(path))
        return self._canon[path]

    def static(self, name: str) -> Optional[FArray]:
        """A value that compute does not change (topology, parameters, constants, filters)."""
        if name not in self._static:
            self._static.update(self.backend.get([name]))
        return self._static[name]

    def statics(self, names: list[str]) -> None:
        todo = [n for n in names if n not in self._static]
        if todo:
            self._static.update(self.backend.get(todo))

    def domain_functions(self) -> dict[str, Callable]:
        """Topounit facts from the domain declaration: `elevation(t)` (declared elevation, NaN where
        none), `lowest(t)` (the lowest topounit of its gridcell, ties included), `uphill_count(t)`
        (other topounits whose top_pp%downhill_ti, as the domain gives it ([domain] downhill), is t), `downhill_inactive(t)`
        (t's downhill neighbour is an inactive, weight-0 topounit: set_topounit does not skip them).
        """
        if getattr(self, "_domain_fns", None) is None:
            lay = self.layout
            elev = np.array([np.nan if e is None else float(e) for e in lay.elevation])
            g = lay.ancestor_of("t", "g")
            lowest = np.zeros(elev.size, dtype=bool)
            for gi in np.unique(g):
                ts = np.flatnonzero(g == gi)
                if np.isfinite(elev[ts]).all():
                    lowest[ts] = elev[ts] == elev[ts].min()
            try:
                dn = np.asarray(lay.downhill())
            except Exception:  # elevations not declared: no neighbours
                dn = np.full(elev.size, -1)
            ups = np.zeros(elev.size, dtype=np.int64)
            for t, d in enumerate(dn):
                if 0 < d <= elev.size and d - 1 != t:
                    ups[d - 1] += 1

            def at(values):
                return lambda t: values[np.asarray(t, dtype=np.int64) - 1]

            act = lay.active("t")
            dead = np.array(
                [0 < d <= elev.size and not act[d - 1] for d in dn], dtype=bool
            )
            self._domain_fns = {
                "elevation": at(elev),
                "lowest": at(lowest),
                "uphill_count": at(ups),
                "downhill_inactive": at(dead),
            }
        return self._domain_fns

    def aggregate(
        self, how: str, x: FArray, level: str, member: Optional[str] = None
    ) -> np.ndarray:
        """`child_<how>(x, level[, filter%member])`: x (a variable of a finer level) reduced over each
        element of level; with a filter, only over x's elements in it (the session's filter, as `over`).
        """
        from spel.proptest.results import _aggregate

        var = self.inv.vars.get(x.name) if x.name else None
        if var is None or var.level is None:
            raise ExprError(
                f"child_{how}: {x.name or 'the argument'} is not a variable with a subgrid dimension"
            )
        if LEVELS.index(var.level) <= LEVELS.index(level):
            raise ExprError(
                f"child_{how}: {x.name} is of level {var.level!r}, not finer than {level!r}"
            )
        data = np.asarray(x.data, dtype=float)
        if data.ndim != 1:
            raise ExprError(
                f"child_{how}: {x.name} has more than one value per element"
            )
        include = None
        if member is not None:
            if member not in self.inv.filters:
                raise ExprError(
                    f"child_{how}: filter%{member} is not a filter of the case"
                )
            idx = self.elements((var.level, member))
            if idx.size and (idx.min() < 1 or idx.max() > data.size):
                raise ExprError(
                    f"child_{how}: filter%{member} holds elements of another level than {x.name}"
                )
            include = np.zeros(data.size, dtype=bool)
            include[idx - 1] = True
        return _aggregate(self.layout, data, var.level, level, how, include)

    def elements(self, over: tuple[str, Optional[str]]) -> np.ndarray:
        """Fortran indices of the elements `over` selects."""
        level, member = over
        if member is None:
            return np.arange(1, self.layout.count(level) + 1)
        num = int(self.static(f"filter%num_{member}").data)
        return np.asarray(self.static(f"filter%{member}").data[:num], dtype=np.int64)

    def locals_for(self, elems: np.ndarray, level: str) -> dict[str, Any]:
        """The element variables for elements of `level`: the element itself, its ancestor at
        each coarser level, and finer levels that are the same element (domain.Layout.one_to_one).
        A finer level with several elements per element (the columns of an urban landunit) is
        not bound: naming it is an error."""
        elems = np.asarray(elems, dtype=np.int64)
        out: dict[str, Any] = {}
        for lev in ELEMENT_VARS:
            if lev == level:
                out[lev] = elems
            elif LEVELS.index(lev) < LEVELS.index(level):
                out[lev] = (
                    self.layout.parent(level, lev)[elems - 1] if elems.size else elems
                )
            elif self.layout.one_to_one(level, lev):
                out[lev] = elems
        return out

    def shape(self, name: str) -> tuple[int, ...]:
        return tuple(hi - lo + 1 for lo, hi in self.domain.extents[name])

    def lbounds(self, name: str) -> tuple[int, ...]:
        return tuple(lo for lo, _ in self.domain.extents[name])

    def pft_of(self, var) -> Optional[np.ndarray]:
        """The pft of each element of the variable's level (-1: urban, lake), for per-pft rules."""
        return self.layout.pft_of(var.level) if var.level else None

    def half_count(self, level: str) -> int:
        """Elements of a level in one half of the sets (the base half of a relation)."""
        return self.layout.count(level) // 2

    def farrays(self, inputs: dict[str, np.ndarray]) -> dict[str, FArray]:
        return {
            n: FArray(
                a, self.lbounds(n) if n in self.domain.extents else (1,) * a.ndim, n
            )
            for n, a in inputs.items()
        }

    def arrays(
        self, ex: Example, state: Optional[dict[str, FArray]] = None
    ) -> dict[str, FArray]:
        """The example's inputs and, after a run, the state it returned, as Fortran arrays by name."""
        out = self.farrays(ex.inputs)
        out.update(state or {})
        return out

    def assign(
        self,
        name: str,
        target: np.ndarray,
        value: Any,
        rows: Optional[np.ndarray],
        elemental: bool,
        where: str,
    ) -> None:
        """A result into a Fortran-shaped array, at `rows` (0-based) of its subgrid axis.
        elemental: the value has one entry per element (elements first, `Evaluator.elemental`)
        and applies to every other dimension; otherwise it is the same for every element.
        """
        axis = self.inv.vars[name].subgrid_axis
        value = np.asarray(value)
        try:
            if axis is None:
                target[...] = value
                return
            t = np.moveaxis(target, axis, 0)
            sel = slice(None) if rows is None else rows
            sub = t[sel]
            if elemental and 0 < value.ndim < sub.ndim:
                value = value.reshape(value.shape + (1,) * (sub.ndim - value.ndim))
            t[sel] = np.broadcast_to(value, sub.shape).astype(target.dtype, copy=False)
        except ValueError:
            raise SpecError(
                f"{where}: a value of shape {value.shape} does not fit {name} "
                f"{tuple(target.shape)}"
            ) from None

    def partner(
        self,
        elems: np.ndarray,
        level: str,
        permutation: Optional[list[int]] = None,
    ) -> np.ndarray:
        """The follow-up element (Fortran index) of each base element of a level."""
        half = self.half_count(level)
        if permutation is None:
            return elems + half
        inverse = np.argsort(
            self.layout.row_permutation(level, np.asarray(permutation))
        )
        return inverse[elems - 1] + 1 + half

    def gridcell_mean(self, level: str, a: np.ndarray, axis: int) -> np.ndarray:
        """The mean of a per-element array over each gridcell's elements of `level` (and every
        other dimension): one value per gridcell."""
        t = (
            np.moveaxis(np.asarray(a, dtype=float), axis, 0)
            .reshape(self.layout.count(level), -1)
            .mean(axis=1)
        )
        g = self.layout.gridcell_of(level)
        n = self.layout.count("g")
        return np.bincount(g, weights=t, minlength=n) / np.bincount(g, minlength=n)

    def rows_of(self, level: str, k: int, other: str) -> np.ndarray:
        """0-based rows of level `other` that belong to element k (Fortran) of `level`: its
        ancestor, itself, or all its descendants (the columns of an urban landunit)."""
        if LEVELS.index(other) <= LEVELS.index(level):
            return np.asarray([self.layout.parent(level, other)[k - 1] - 1])
        return self.layout.descendants(level, k, other)

    def elements_at(self, level: str, elems: np.ndarray, other: str) -> np.ndarray:
        """The elements (Fortran indices, sorted, unique) of level `other` that belong to the
        given elements of `level` (their ancestors, themselves or their descendants)."""
        if not len(elems):
            return np.asarray(elems, dtype=np.int64)
        return (
            np.unique(
                np.concatenate([self.rows_of(level, int(k), other) for k in elems])
            )
            + 1
        )

    def describe_element(self, k: int, level: str = "p") -> str:
        return f"element {k} ({self.layout.describe(level, k)})"
