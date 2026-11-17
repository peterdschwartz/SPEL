"""Examples (`Example`) and how a session draws them (`Drawing`): every ruled input from its rule,
`derive`, multi-step forcing, `assume`, and the follow-up inputs of relations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from spel.proptest import strategies as strat
from spel.proptest.domain import SynthDomain
from spel.proptest.errors import SpecError
from spel.proptest.fexpr import ExprError, evaluate_bound, parse_section
from spel.proptest.inventory import Inventory
from spel.proptest.rules import KINDS as RULE_KINDS
from spel.proptest.rules import Rule
from spel.proptest.runner.elements import ElementSpace
from spel.proptest.runner.evaluation import Evaluation, _per_element
from spel.proptest.spec import INPUT_KINDS, Relation, Scenario, Spec


@dataclass
class Example:
    inputs: dict[str, np.ndarray]  # inst%field -> Fortran-shaped array
    u: dict[str, np.ndarray] = field(default_factory=dict)
    modes: dict[str, int] = field(default_factory=dict)
    seed: int = 0
    extra: dict[str, Any] = field(
        default_factory=dict
    )  # what a check kind adds (JSON values)
    follow: Optional["Example"] = (
        None  # the follow-up of a relation that runs it under other namelist options
    )
    forcing: dict[str, np.ndarray] = field(
        default_factory=dict
    )  # multi-step: inst%field -> (step, *shape)


class Drawing:
    """Draws one session's examples (its configuration decides which inputs are live)."""

    def __init__(self, space: ElementSpace, evaluation: Evaluation, spec: Spec, inv: Inventory, domain: SynthDomain):
        self.space, self.evaluation, self.spec, self.inv, self.domain = space, evaluation, spec, inv, domain
        self.layout = space.layout
        self._levels: dict[tuple[str, str], tuple] = {}

    def rule_shape(
        self, name: str, rule: Rule
    ) -> tuple[tuple[int, ...], Optional[np.ndarray]]:
        """The shape of the u a rule maps, and the variable's own value for a relative rule (`scale`)."""
        if RULE_KINDS[rule.kind].relative and self.inv.vars[name].kind in INPUT_KINDS:
            base = np.asarray(self.space.static(name).data, dtype=float)
            return base.shape, base
        return (
            self.space.shape(name),
            None,
        )  # (a relative rule of a variable that is dormant here: only its u)

    def level_rules(self, name: str, rule: Rule) -> list[tuple[tuple, Rule]]:
        """A rule's `levels` overrides as (index into the variable's array, rule): each key is a
        section of the first dimension that is not the subgrid one."""
        out = []
        for text, sub in rule.by_level:
            key = (name, text)
            if key not in self._levels:
                var = self.inv.vars[name]
                axis = next(k for k, d in enumerate(var.dims) if not d.level)
                lb, n = self.space.lbounds(name)[axis], self.space.shape(name)[axis]
                env = self.inv.bound_env(self.layout.sizes(), self.domain.constants)
                env.update(self.spec.domain.dims)
                try:
                    start, end = parse_section(text)
                    lo = 0 if start is None else evaluate_bound(start, env) - lb
                    hi = n if end is None else evaluate_bound(end, env) - lb + 1
                except ExprError as exc:
                    raise SpecError(f"{name}: levels {text!r}: {exc}") from None
                if not 0 <= lo < hi <= n:
                    raise SpecError(
                        f"{name}: levels {text!r} is outside {lb}:{lb + n - 1}"
                    )
                sel = [slice(None)] * var.rank
                sel[axis] = slice(lo, hi)
                self._levels[key] = tuple(sel)
            out.append((self._levels[key], sub))
        return out

    def inputs_at(self, scenario: Scenario, u: Any = 0.5) -> Example:
        """The example with every ruled input at quantile `u` of its rule, then `derive`.
        u: one number for everything, a callable `shape -> array` (e.g. `rng.random`), or a
        dict input -> array (inputs it lacks: 0.5)."""
        ex = Example({})
        for name in sorted(scenario.inputs):
            rule = scenario.inputs[name]
            var = self.inv.vars[name]
            if name not in self.domain.extents:
                continue  # dormant here and of unknown shape: a session that reads it draws it (`fill_missing`)
            shape, base = self.rule_shape(name, rule)
            if isinstance(u, dict):
                uu = np.broadcast_to(np.asarray(u.get(name, 0.5), dtype=float), shape)
            elif callable(u):
                uu = np.asarray(u(shape), dtype=float)
            else:
                uu = np.full(shape, float(u))
            ex.u[name] = uu
            if base is None and RULE_KINDS[rule.kind].relative:
                continue  # dormant here; a session that reads it fills it from the same u (`fill_missing`)
            ex.inputs[name] = strat.values_of(
                rule,
                uu,
                self.space.pft_of(var),
                var.subgrid_axis,
                var.type,
                base,
                self.level_rules(name, rule),
            )
        self.derive(scenario, ex, skip=set())
        self.forcing_at(scenario, ex, u)
        return ex

    def forcing_at(self, scenario: Scenario, ex: Example, u: Any) -> None:
        """The values of every forcing rule for each step (step axis first), from u as inputs_at."""
        if scenario.steps <= 1:
            return
        for name in sorted(scenario.forcing):
            rule = scenario.forcing[name]
            var = self.inv.vars[name]
            if name not in self.domain.extents:
                continue
            shape, base = self.rule_shape(name, rule)
            ext = (scenario.steps, *shape)
            key = f"forcing:{name}"
            if isinstance(u, dict):
                uu = np.broadcast_to(np.asarray(u.get(key, 0.5), dtype=float), ext)
            elif callable(u):
                uu = np.asarray(u(ext), dtype=float)
            else:
                uu = np.full(ext, float(u))
            ex.u[key] = uu
            if base is None and RULE_KINDS[rule.kind].relative:
                continue
            ex.forcing[name] = np.stack(
                [
                    strat.values_of(
                        rule,
                        uu[k],
                        self.space.pft_of(var),
                        var.subgrid_axis,
                        var.type,
                        base,
                        self.level_rules(name, rule),
                    )
                    for k in range(scenario.steps)
                ]
            )

    def fill_missing(self, scenario: Scenario, ex: Example) -> None:
        """Ruled inputs this session reads that the example does not hold yet (a relative rule
        needs the variable's own value, which only a session where it is live has)."""
        for name in sorted(scenario.inputs):
            if name in ex.inputs or self.inv.vars[name].kind not in INPUT_KINDS:
                continue
            rule = scenario.inputs[name]
            var = self.inv.vars[name]
            shape, base = self.rule_shape(name, rule)
            if name in ex.u:
                uu = np.broadcast_to(np.asarray(ex.u[name], dtype=float), shape)
            else:  # the drawing session did not know its shape: from the example's seed
                uu = np.random.default_rng([ex.seed, sum(name.encode())]).random(shape)
            ex.inputs[name] = strat.values_of(
                rule,
                uu,
                self.space.pft_of(var),
                var.subgrid_axis,
                var.type,
                base,
                self.level_rules(name, rule),
            )

    def draw_example(self, scenario: Scenario, choose: "strat.Chooser") -> Example:
        """An example drawn as a test run draws it: a seed, then per input a mode (and a constant u)."""
        seed = strat.draw_seed(choose)
        rng = np.random.default_rng(seed)
        u, modes = {}, {}
        for name in sorted(scenario.inputs):
            rule = scenario.inputs[name]
            if name not in self.domain.extents:
                continue
            shape, _ = self.rule_shape(name, rule)
            u[name], modes[name] = strat.draw_u(
                choose, name, rule, shape, rng, self.inv.vars[name].subgrid_axis
            )
        for name in sorted(scenario.forcing) if scenario.steps > 1 else ():
            rule = scenario.forcing[name]
            if name not in self.domain.extents:
                continue
            shape, _ = self.rule_shape(name, rule)
            axis = self.inv.vars[name].subgrid_axis
            key = f"forcing:{name}"  # a mode over (step, *shape): constant, i.i.d., edges, per element
            u[key], modes[key] = strat.draw_u(
                choose,
                key,
                rule,
                (scenario.steps, *shape),
                rng,
                None if axis is None else axis + 1,
            )
        ex = self.inputs_at(scenario, u)
        ex.modes, ex.seed = modes, seed
        return ex

    def sample(self, scenario: Scenario | str, seed: int = 0) -> Example:
        """An example drawn like a test run's (the same modes), from `seed` instead of Hypothesis."""
        scen = self.spec.scenarios[scenario] if isinstance(scenario, str) else scenario
        return self.draw_example(scen, strat.RngChooser(np.random.default_rng(seed)))

    def derive(self, scenario: Scenario, ex: Example, skip: set[str]) -> None:
        for name, expr in scenario.derive:
            if name in skip:
                continue
            var = self.inv.vars[name]
            if (
                var.kind not in INPUT_KINDS
            ):  # dormant under this configuration: nothing reads it
                continue
            if name not in ex.inputs:
                ex.inputs[name] = np.zeros(
                    self.space.shape(name), dtype=float if var.type == "real" else np.int32
                )
            ev = self.evaluation.evaluator(self.space.farrays(ex.inputs))
            n = self.layout.count(var.level) if var.level else None
            local = self.space.locals_for(np.arange(1, n + 1), var.level) if var.level else {}
            where = f"[scenario.{scenario.name}].derive {name}"
            try:
                value = ev(expr.node, local, n)
            except ExprError as exc:
                raise SpecError(f"{where}: {exc}") from None
            self.space.assign(
                name, ex.inputs[name], value, None, ev.elemental(expr.node), where
            )

    def follow_up(self, scenario: Scenario, rel: Relation, ex: Example) -> Example:
        """Sets nsets/2.. become a copy of the base sets, then the relation's transforms."""
        out = Example(
            {k: v.copy() for k, v in ex.inputs.items()},
            ex.u,
            ex.modes,
            ex.seed,
            dict(ex.extra),
        )
        out.forcing = {k: v.copy() for k, v in ex.forcing.items()}
        for name, arr in out.inputs.items():
            var = self.inv.vars[name]
            if var.subgrid_axis is not None:
                half = self.space.half_count(var.level)
                t = np.moveaxis(arr, var.subgrid_axis, 0)
                t[half : 2 * half] = t[:half]
        for (
            name,
            arr,
        ) in out.forcing.items():  # (step, *shape): the subgrid axis is one further
            var = self.inv.vars[name]
            if var.subgrid_axis is not None:
                half = self.space.half_count(var.level)
                t = np.moveaxis(arr, var.subgrid_axis + 1, 0)
                t[half : 2 * half] = t[:half]
        values = []
        for name, expr in rel.follow:
            ev = self.evaluation.evaluator(self.space.farrays(out.inputs))
            level = self.inv.vars[name].level
            half = self.space.half_count(level) if level else None
            local = self.space.locals_for(np.arange(1, half + 1), level) if level else {}
            try:
                values.append(
                    (name, ev(expr.node, local, half), ev.elemental(expr.node))
                )
            except ExprError as exc:
                raise SpecError(
                    f"[[relation]] {rel.name!r} follow {name}: {exc}"
                ) from None
        for name, value, elemental in values:  # all transforms see the base values
            if name not in out.inputs:
                raise SpecError(
                    f"[[relation]] {rel.name!r}: {name} is neither ruled nor derived"
                )
            level = self.inv.vars[name].level
            half = self.space.half_count(level) if level else 0
            self.space.assign(
                name,
                out.inputs[name],
                value,
                np.arange(half, 2 * half) if level else None,
                elemental,
                f"[[relation]] {rel.name!r} follow {name}",
            )
        self.derive(scenario, out, skip={n for n, _ in rel.follow})
        return out

    def follow_example(
        self, scenario: Scenario, rel: Relation, ex: Example, target: "Drawing"
    ) -> Example:
        """The follow-up of a relation that runs it in another session (`target`: the drawing of
        this case under other options): a copy of the example, the `follow` transforms over every element (each
        an expression of the base inputs), then the target's `derive`."""
        out = Example(
            {k: v.copy() for k, v in ex.inputs.items()}, ex.u, ex.modes, ex.seed
        )
        target.fill_missing(scenario, out)
        ev = self.evaluation.evaluator(self.space.farrays(ex.inputs))
        for name, expr in rel.follow:
            where = f"[[relation]] {rel.name!r} follow {name}"
            if name not in out.inputs:
                raise SpecError(
                    f"[[relation]] {rel.name!r}: {name} is neither ruled nor derived"
                )
            level = self.inv.vars[name].level
            n = self.layout.count(level) if level else None
            local = self.space.locals_for(np.arange(1, n + 1), level) if level else {}
            try:
                value = ev(expr.node, local, n)
            except ExprError as exc:
                raise SpecError(f"{where}: {exc}") from None
            target.space.assign(
                name, out.inputs[name], value, None, ev.elemental(expr.node), where
            )
        target.derive(scenario, out, skip={n_ for n_, _ in rel.follow})
        return out

    def permute_follow(self, ex: Example) -> None:
        """Reorder the follow-up gridcells among those of the same declaration (from the
        example's seed), each with everything under it: follow-up gridcell k then holds what
        gridcell `permutation[k]` held. Recorded in `ex.extra`."""
        half = self.space.half_count("g")
        labels = [g.label for g in self.layout.gridcells[:half]]
        rng = np.random.default_rng(ex.seed + 1)
        perm = np.arange(half)
        for label in sorted(set(labels)):
            rows = np.flatnonzero(np.asarray(labels) == label)
            perm[rows] = rng.permutation(rows)
        for name, arr in ex.inputs.items():
            var = self.inv.vars[name]
            if var.subgrid_axis is not None:
                h = self.space.half_count(var.level)
                t = np.moveaxis(arr, var.subgrid_axis, 0)
                t[h : 2 * h] = t[h : 2 * h][
                    self.layout.row_permutation(var.level, perm)
                ]
        ex.extra["permutation"] = perm.tolist()

    def assume_mask(
        self, scenario: Scenario, ex: Example, elems: np.ndarray
    ) -> np.ndarray:
        mask = np.ones(len(elems), dtype=bool)
        if not scenario.assume:
            return mask
        ev = self.evaluation.evaluator(self.space.farrays(ex.inputs))
        local = self.space.locals_for(elems, scenario.level)
        for e in scenario.assume:
            where = f"[scenario.{scenario.name}].assume {e.text!r}"
            try:
                r = ev(e.node, local, len(elems))
            except ExprError as exc:
                raise SpecError(f"{where}: {exc}") from None
            mask &= _per_element(r, mask.shape, where)
        return mask
