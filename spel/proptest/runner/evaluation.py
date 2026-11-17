"""Spec expressions in a session (`Evaluation`): how names resolve (state arrays, values that do not
change, parameters, constants, time scalars, filters), the evaluator with its functions, and the
namespaces checks use (`input%`, `prev%`, `base%`/`follow%`)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np

from spel.proptest.checks import INPUT, Expr
from spel.proptest.domain import SynthDomain
from spel.proptest.errors import SpecError
from spel.proptest.fexpr import Evaluator, ExprError, FArray, FuncExpression, Namespace, children, designator, uses
from spel.proptest.inventory import Inventory
from spel.proptest.runner.elements import ElementSpace
from spel.proptest.spec import Scenario, Spec

if TYPE_CHECKING:
    from spel.proptest.runner.drawing import Example

STATE_KINDS = ("input", "inout", "output")


class Evaluation:
    """Expressions over one session's state. `time_now`: the time scalars of the step evaluated
    now (multi-step runs; None otherwise)."""

    def __init__(self, space: ElementSpace, spec: Spec, inv: Inventory, domain: SynthDomain, case_dir):
        self.space, self.spec, self.inv, self.domain, self.case_dir = space, spec, inv, domain, case_dir
        self.layout = space.layout
        self.time_now: Optional[dict[str, Any]] = None
        self._subspaces: dict[tuple[str, str], dict] = {}

    def parse(self, text: str, where: str, namespaces: tuple[str, ...] = ()) -> Expr:
        from spel.proptest.spec import Names

        return Names(self.inv, set(self.spec.functions)).expr(text, where, namespaces)

    def resolver(self, arrays: dict[str, Any]) -> Callable[[list[str]], Any]:
        constants = self.domain.constants
        time_names = self.inv.time_names() or ()

        def resolve(path: list[str]) -> Any:
            name = self.space.canonical(tuple(path))
            if name is not None:
                if name in arrays:
                    return arrays[name]
                if self.inv.vars[name].kind in STATE_KINDS:
                    raise ExprError(f"{name} is not available here")
                return self.as_value(self.space.static(name), name)
            if len(path) == 1:
                head = path[0]
                if self.inv.parameters.get(head) is not None:
                    return self.inv.parameters[head]
                if head in constants:
                    v = np.asarray(constants[head])
                    return FArray(v, (1,) * v.ndim, head) if v.ndim else v[()]
                if head in time_names:
                    if self.time_now is not None and head in self.time_now:
                        return self.time_now[head]
                    return self.as_value(self.space.static(head), head)
            if len(path) == 2 and path[0] == "filter":
                return self.as_value(
                    self.space.static(f"filter%{path[1]}"), f"filter%{path[1]}"
                )
            raise KeyError("%".join(path))

        return resolve

    @staticmethod
    def as_value(fa: Optional[FArray], name: str) -> Any:
        if fa is None:
            raise ExprError(f"{name} is not allocated in the case")
        return fa.data[()] if fa.data.ndim == 0 else fa

    def evaluator(
        self,
        arrays: dict[str, Any],
        functions: Optional[dict[str, Callable]] = None,
        prev: Any = None,
        history: Optional[list] = None,
    ) -> Evaluator:
        """prev: the state before the step (`tend(x)`): its arrays, or (arrays, namespaces);
        default: none. history: the states after steps 1..k as (arrays, namespaces), for `acc(x)`.
        """

        def env(e):
            if not isinstance(e, tuple):
                return self.resolver(e), {}
            if (
                len(e) == 3
            ):  # (arrays, namespaces, prev): a step of the history with its own prev
                return self.resolver(e[0]), e[1], env(e[2])
            return self.resolver(e[0]), e[1]

        return Evaluator(
            self.resolver(arrays),
            {**self.spec.functions, **self.space.domain_functions(), **(functions or {})},
            prev=env(prev) if prev is not None else None,
            aggregate=self.space.aggregate,
            history=[env(h) for h in history] if history else None,
        )

    def needed(
        self, exprs: list[Expr], namespaces: tuple[str, ...] = ("base", "follow")
    ) -> list[str]:
        """State variables the expressions read after compute (fetched from the backend)."""
        out = set()
        for e in exprs:
            for u in uses(e.node):
                path = u.path[1:] if u.path[0] in namespaces else u.path
                if not path:
                    continue
                name = self.space.canonical(tuple(path))
                if name is not None and self.inv.vars[name].kind in STATE_KINDS:
                    out.add(name)
        return sorted(out)

    def state_namespace(self, label: str, arrays: dict[str, FArray]) -> Namespace:
        def get(path: list[str]) -> Any:
            name = self.space.canonical(tuple(path))
            if name is None or name not in arrays:
                raise ExprError(f"{label}%{'%'.join(path)}: not available")
            return arrays[name]

        return Namespace(label, get)

    def half(
        self,
        arrays: dict[str, Any],
        which: int,
        permutation: Optional[list[int]] = None,
    ) -> Namespace:
        """`base%` (which = 0) or `follow%` (1): the state restricted to that half of the sets.
        With a permutation of the follow-up rows, `follow%x(p)` is the follow-up of base element p.
        """
        label = "base" if which == 0 else "follow"

        def get(path: list[str]) -> Any:
            name = self.space.canonical(tuple(path))
            if name is None or name not in arrays:
                raise ExprError(f"{label}%{'%'.join(path)}: not a state variable")
            fa = arrays[name]
            var = self.inv.vars[name]
            axis = var.subgrid_axis
            if axis is None:
                return fa
            half = self.space.half_count(var.level)
            inverse = None
            if permutation is not None and which == 1:
                inverse = np.argsort(
                    self.layout.row_permutation(var.level, np.asarray(permutation))
                )
            sl = [slice(None)] * fa.data.ndim
            sl[axis] = slice(which * half, (which + 1) * half)
            data = fa.data[tuple(sl)]
            if inverse is not None:
                data = np.take(data, inverse, axis=axis)
            lb = list(fa.lb)
            lb[axis] = 1
            return FArray(data, tuple(lb), name)

        return Namespace(label, get)

    def inputs_namespace(self, ex: Example) -> Namespace:
        before = self.space.farrays(ex.inputs)

        def get(path: list[str]) -> Any:
            name = self.space.canonical(tuple(path))
            if name is None or name not in before:
                raise ExprError(
                    f"{INPUT}%{'%'.join(path)}: not an input the scenario sets"
                )
            return before[name]

        return Namespace(INPUT, get)

    def context_functions(
        self, scenario: Scenario, ex: Example, level: Optional[str] = None
    ) -> dict[str, Callable]:
        """Functions that need the example: `active_dir(p, "<output>", k)` (p: elements of
        `level`, default the scenario's)."""
        level = level or scenario.level

        def active_dir(p, output, k=1):
            from spel.proptest import subspace

            key = (scenario.name, str(output))
            if key not in self._subspaces:
                self._subspaces[key] = subspace.load(self.case_dir, *key)
            res = self._subspaces[key]
            k = int(k)
            if not 1 <= k <= len(res["inputs"]):
                raise ExprError(f"active_dir: direction {k} of {len(res['inputs'])}")
            coord = np.zeros(
                self.layout.count("g")
            )  # per gridcell: subspace steps move a gridcell as a whole
            for name, w in zip(
                res["inputs"], np.asarray(res["eigenvectors"])[:, k - 1]
            ):
                if name not in ex.u:
                    raise ExprError(
                        f"active_dir: {name} is not a ruled input of this scenario any more"
                    )
                var = self.inv.vars[name]
                u = np.asarray(ex.u[name], dtype=float)
                # a layered input moved as a whole (its mean u over the gridcell's elements); one
                # without a subgrid dimension is the same for every element
                u = (
                    u.mean()
                    if var.subgrid_axis is None
                    else self.space.gridcell_mean(var.level, u, var.subgrid_axis)
                )
                coord += w * (2.0 * u - 1.0)
            return coord[self.layout.gridcell_of(level)[np.asarray(p) - 1]]

        return {"active_dir": active_dir}


def _per_element(value: Any, shape: tuple[int, ...], where: str) -> np.ndarray:
    """A logical per element (or one for all of them)."""
    v = np.asarray(value, dtype=bool)
    if v.ndim > 1 or (v.ndim == 1 and v.shape != shape):
        raise SpecError(
            f"{where}: one logical per element expected, got shape {v.shape} "
            "(reduce the other dimensions, e.g. with all(...))"
        )
    return np.broadcast_to(v, shape)


def _use_node(root, use):
    """The node of `root` that is the use (first match): the call x(...) or the bare designator."""
    stack = [root]
    while stack:
        n = stack.pop()
        if isinstance(n, FuncExpression):
            if designator(n.function) == list(use.path) and use.applied:
                return n
        elif not use.applied and designator(n) == list(use.path):
            return n
        stack.extend(reversed(children(n)))
    raise ExprError("use not found")
