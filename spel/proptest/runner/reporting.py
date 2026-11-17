"""How a failing check is reported: the `Failure` (what failed, where, the report), raised as a
`PropertyFailure` that carries it; the `FailureSignature` that tells bugs apart while shrinking a
drawn domain; and the report itself (the failing element, the values the expression used, the
drawn inputs)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator, Optional

import numpy as np

from spel.proptest.checks import INPUT, PREV, Check
from spel.proptest.fexpr import (
    CHILD_AGGREGATES,
    Evaluator,
    ExprError,
    FArray,
    FuncExpression,
    children,
    designator,
    keyword_of,
    uses,
)
from spel.proptest.runner.elements import ElementSpace
from spel.proptest.runner.evaluation import _use_node
from spel.proptest.spec import ELEMENT_VARS

if TYPE_CHECKING:
    from spel.proptest.runner.drawing import Example


@dataclass(frozen=True)
class FailureSignature:
    """How a check failed, to tell bugs apart while shrinking a drawn domain: the routine aborted,
    or the check was false with these terms non-finite at the first failing element (empty: a
    finite mismatch). Terms are the expression's per-element operands: its variables, and its
    child_* reductions as a whole (a NaN in a column outside the element's sum does not count).
    The step is left out: a candidate domain redraws the inputs, so the first failing step moves.
    """

    aborted: bool = False
    nonfinite: frozenset[str] = frozenset()

    def text(self) -> str:
        if self.aborted:
            return "the routine aborted"
        return (
            ("non-finite: " + ", ".join(sorted(self.nonfinite)))
            if self.nonfinite
            else "finite values"
        )

    def to_json(self) -> dict:
        return {"aborted": self.aborted, "nonfinite": sorted(self.nonfinite)}


@dataclass(kw_only=True)
class Failure:
    """A check that failed on an example: what `Session.judge` found, saved by `FailureStore.save`."""

    scenario: str
    check: str
    example: Example
    message: str  # the report
    element: Optional[int] = None  # Fortran index of the first failing element
    signature: Optional[FailureSignature] = None


class PropertyFailure(AssertionError):
    """A check failed on an example (`Session.judge`): `failure` says how, and the message is its
    report. It is an exception because Hypothesis needs one to see a failing example; it re-raises
    the shrunk example's own, so `orchestration.run` saves `exc.failure`."""

    def __init__(self, failure: Failure):
        super().__init__(failure.message)
        self.failure = failure

    def __reduce__(self):
        return PropertyFailure, (self.failure,)


def report(
    space: ElementSpace,
    check: Check,
    text: str,
    node,
    ev: Evaluator,
    local: dict,
    elems: np.ndarray,
    bad: np.ndarray,
    ex: Example,
    mask: np.ndarray,
    level: Optional[str] = None,
) -> str:
    level = level or "p"
    k = int(elems[bad[0]])
    one = {
        v: (np.asarray(x)[bad[:1]] if isinstance(x, np.ndarray) else x)
        for v, x in local.items()
    }
    lines = [
        f"{check.name}: {text}",
        f"  fails at {bad.size} of {int(mask.sum())} elements; first: "
        f"{space.describe_element(k, level)}",
    ]
    shown = set()
    for u in uses(node):
        label = "%".join(u.path)
        if (
            label in shown
            or u.path[0] in ELEMENT_VARS
            or (len(u.path) == 1 and u.path[0] in (*check.names, INPUT))
        ):
            continue
        shown.add(label)
        try:
            val = ev(_use_node(node, u), one, 1)
        except ExprError:
            continue
        lines.append(f"    {label} = {_fmt(val)}")
    lines.append("  drawn inputs at that element:")
    for name in sorted(ex.inputs):
        var = space.inv.vars[name]
        axis = var.subgrid_axis
        arr = ex.inputs[name]
        if axis is None:
            val = arr
        else:
            rows = space.rows_of(level, k, var.level)
            val = np.take(arr, rows, axis=axis)
            if rows.size == 1:
                val = np.take(val, 0, axis=axis)
        lines.append(f"    {name} = {_fmt(val)}")
    return "\n".join(lines)


def terms(space: ElementSpace, check: Check, node) -> Iterator[tuple[str, Any]]:
    """The per-element operands of a check expression (label, node): child_* calls as a whole,
    and the variables it names (x(p), a namespace's base%x(c), bare names); not functions,
    element variables or literals."""

    def variable(path: list[str]) -> Optional[str]:
        """The canonical label of a variable (with its namespace), or None."""
        if path[0] in (*check.names, INPUT, PREV):
            name = space.canonical(tuple(path[1:])) if len(path) > 1 else None
            return f"{path[0]}%{name}" if name else None
        return space.canonical(tuple(path))

    stack = [node]
    while stack:
        n = stack.pop()
        if (kw := keyword_of(n)) is not None:
            stack.append(kw[1])
            continue
        if isinstance(n, FuncExpression):
            path = designator(n.function)
            if path is not None and len(path) == 1 and path[0] in CHILD_AGGREGATES:
                yield str(n), n
            elif path is not None and (label := variable(path)):
                yield label, n
            else:
                stack.extend(reversed(n.args))
            continue
        path = designator(n)
        if path is not None:
            if label := variable(path):
                yield label, n
            continue
        stack.extend(reversed(children(n)))


def signature(space: ElementSpace, check: Check, node, ev: Evaluator, one: dict) -> FailureSignature:
    """The FailureSignature of a failure: the terms non-finite at the element `one` binds."""
    bad = set()
    for label, sub in terms(space, check, node):
        try:
            v = np.asarray(ev(sub, one, 1), dtype=float)
        except (ExprError, TypeError, ValueError):
            continue
        if v.size and not np.all(np.isfinite(v)):
            bad.add(label)
    return FailureSignature(nonfinite=frozenset(bad))


def _fmt(v: Any) -> str:
    if isinstance(v, FArray):
        v = v.data
    a = np.asarray(v)
    if a.ndim and a.size == 1:
        a = a.reshape(())
    if a.ndim == 0:
        return repr(a.item())
    with np.printoptions(precision=6, threshold=12, edgeitems=4, linewidth=110):
        return np.array2string(a.squeeze())


def _indent(text: str, pre: str = "    ") -> str:
    return "\n".join(pre + line for line in text.splitlines())
