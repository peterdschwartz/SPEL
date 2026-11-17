"""Running the routines (`Stepping`): one call, or a multi-step run with its time scalars, forcing
and `between` assignments, and the boundary every multi-step scenario must declare."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from spel.proptest.checks import PREV, Expr
from spel.proptest.domain import DomainSpec, SynthDomain
from spel.proptest.errors import SpecError
from spel.proptest.fexpr import ExprError, FArray
from spel.proptest.inventory import Inventory
from spel.proptest.runner.drawing import Example
from spel.proptest.runner.elements import ElementSpace
from spel.proptest.runner.evaluation import Evaluation
from spel.proptest.spec import INPUT_KINDS, STEP_NAME, Scenario, Spec
from spel.proptest.worker import FortranAbort


@dataclass
class StepRun:
    """A multi-step run of an example: after each step, every value it set or fetched (the state the
    checks see), and the time scalars of the step."""

    example: Example
    scenario: str
    states: list[dict[str, FArray]]
    times: list[dict[str, Any]]


@dataclass
class JudgedStep:
    """The step of a multi-step run a check is judged after now (None: the last). One object is
    shared by a session, its siblings and its `other`, so relations and differentials compare
    the same step."""

    k: Optional[int] = None


class Stepping:
    """Runs one session's examples on its backend."""

    def __init__(self, backend: Any, space: ElementSpace, evaluation: Evaluation, spec: Spec, inv: Inventory,
                 base_inv: Inventory, domain: SynthDomain, domain_spec: DomainSpec, variant: Any,
                 judged: JudgedStep):
        self.backend, self.space, self.evaluation, self.spec = backend, space, evaluation, spec
        self.inv, self.base_inv, self.domain, self.domain_spec = inv, base_inv, domain, domain_spec
        self.variant, self.judged = variant, judged
        self.layout = space.layout
        self._run: Optional[StepRun] = None
        self._boundary_ok: set[str] = set()

    def live_outputs(self) -> list[str]:
        """What the routines compute per element under this session's configuration."""
        consts = self.domain.constants
        return sorted(
            v.name
            for v in self.inv.of_kind("output", "inout")
            if v.level and not v.driver and self.inv.guard_active(v, consts)
        )

    def live_inputs(self, ex: Example) -> dict[str, np.ndarray]:
        """The example's inputs this session's configuration reads (a rule for a variable that
        is dead here is dormant: its value is drawn but not set)."""
        return {
            n: a for n, a in ex.inputs.items() if self.inv.vars[n].kind in INPUT_KINDS
        }

    def compute(
        self, scenario: Scenario, ex: Example, outputs: list[str]
    ) -> dict[str, FArray]:
        if (
            scenario.steps > 1
        ):  # the state after the step judged now (default: the last)
            run = self.run_steps(scenario, ex)
            k = self.judged.k or scenario.steps
            state = run.states[k - 1]
            missing = [n for n in outputs if n not in state]
            if missing:
                raise SpecError(f"not fetched in the multi-step run: {missing}")
            return dict(state)
        res = self.backend.run(self.live_inputs(ex), outputs, trap=scenario.trap_fpe)
        missing = [n for n, v in res.items() if v is None]
        if missing:
            raise SpecError(f"not allocated in the case: {missing}")
        return res

    def start_time(self, scenario: Scenario) -> dict[str, Any]:
        """The time scalars of step 1: the domain's ([domain.time], defaults), then the scenario's."""
        from spel.proptest.domain import time_values

        values, _ = time_values(self.inv, self.domain_spec, self.domain.constants)
        values.update(scenario.time)
        return values

    def check_boundary(self, scenario: Scenario) -> None:
        """Every variable other elm_drv routines write within a step (coupling.boundary) that the
        routines read must be declared in the scenario: forcing, between or keep."""
        if scenario.name in self._boundary_ok or scenario.steps <= 1:
            return
        if (self.variant.acc or "OFF").upper() != "OFF":
            raise SpecError(
                f"[scenario.{scenario.name}]: multi-step runs need an ACC=OFF build for now "
                "(the time scalars are not updated on the device between steps)"
            )
        if (
            getattr(self.base_inv, "unit_test", None) is not None
        ):  # a synthetic inventory (tests) has none
            from spel.proptest import coupling
            from spel.proptest import spec as specmod

            values = specmod.constants(self.spec, self.base_inv, scenario)
            found = coupling.boundary(self.base_inv, values)
            undeclared = {
                n: w
                for n, w in found.items()
                if self.inv.vars[n].kind in ("input", "inout")
                and not scenario.boundary_declared(n)
            }
            if undeclared:
                lines = [
                    f"  {n}: {coupling.describe(w)}"
                    for n, w in sorted(undeclared.items())
                ]
                raise SpecError(
                    f"[scenario.{scenario.name}]: {len(undeclared)} variables the routines read are also written "
                    "by other elm_drv routines within a step; say what stands in for them between steps in "
                    "`forcing`, `between` or `keep` (SPEC.md, Multi-step runs):\n"
                    + "\n".join(lines)
                )
        self._boundary_ok.add(scenario.name)

    def step_names(self) -> list[str]:
        """What a multi-step run fetches after every step: every output and inout live here."""
        return list(self.live_outputs())

    def run_steps(self, scenario: Scenario, ex: Example) -> StepRun:
        """Run the example's steps (once per example: the run is kept for the checks of every step).
        Before step 1 the state after init with the example's inputs; before every step its time
        scalars and forcing values, after it the `between` assignments."""
        from spel.proptest.timestep import advance

        if (
            self._run is not None
            and self._run.example is ex
            and self._run.scenario == scenario.name
        ):
            return self._run
        self.check_boundary(scenario)
        names = self.step_names()
        # the outputs under this configuration, NaN before every step (not the manifest's, which
        # counts every branch: an input here can be an output there and must keep its value);
        # carried outputs keep what the previous step left where a step does not write them
        poison = sorted(
            n
            for n, v in self.inv.vars.items()
            if v.kind == "output" and v.type == "real" and n not in scenario.carry
        )
        tnames = {e.lower() for e in (self.inv.time_names() or ())}
        time = self.start_time(scenario)
        before = self.space.farrays(ex.inputs)  # the state before step 1
        start = self.live_inputs(ex)
        start.update(
            {
                n: ex.inputs[n]
                for n in scenario.carry  # dormant where nothing uses it
                if n in ex.inputs
                and self.inv.vars[n].kind != "unused"
                and n in self.domain.extents
            }
        )
        self.backend.begin(start)
        states: list[dict[str, FArray]] = []
        times: list[dict[str, Any]] = []
        pending: dict[str, np.ndarray] = {}  # `between` values for the next step
        for k in range(1, scenario.steps + 1):
            if k > 1:
                time = advance(time)
            sets: dict[str, Any] = dict(pending)
            sets.update({n: np.asarray(v) for n, v in time.items() if n in tnames})
            self.evaluation.time_now = time
            try:
                for name, values in ex.forcing.items():
                    if self.inv.vars[name].kind in INPUT_KINDS:
                        sets[name] = np.asarray(values[k - 1])
                for (
                    name,
                    expr,
                ) in scenario.forcing_exprs:  # dormant where nothing reads the target
                    if self.inv.vars[name].kind in INPUT_KINDS:
                        sets[name] = self.step_assign(
                            scenario, name, expr, before, before, k, "forcing"
                        )
                try:
                    state = self.backend.step(
                        sets, names, trap=scenario.trap_fpe, poison=poison
                    )
                except FortranAbort as exc:
                    raise FortranAbort(f"step {k} of {scenario.steps}: {exc}") from None
                after = dict(before)
                after.update(
                    {
                        n: FArray(np.asarray(v), self.space.lbounds(n), n)
                        for n, v in sets.items()
                        if n not in tnames and n in self.domain.extents
                    }
                )
                after.update({n: v for n, v in state.items() if v is not None})
                pending = {}
                for name, expr in scenario.between:
                    if self.inv.vars[name].kind not in INPUT_KINDS:
                        continue  # dormant under these options (ELM's own code is gated the same way)
                    value = self.step_assign(
                        scenario, name, expr, after, before, k, "between"
                    )
                    after[name] = FArray(value, self.space.lbounds(name), name)
                    pending[name] = value
            finally:
                self.evaluation.time_now = None
            states.append(after)
            times.append(dict(time))
            before = after
        self._run = StepRun(ex, scenario.name, states, times)
        return self._run

    def step_assign(
        self,
        scenario: Scenario,
        name: str,
        expr: Expr,
        arrays: dict[str, FArray],
        prev: dict[str, FArray],
        k: int,
        table: str,
    ) -> np.ndarray:
        """A `forcing` or `between` expression for every element of the target's level: bare names
        are `arrays`, `prev%x` is `prev`, `step` is k."""
        var = self.inv.vars[name]
        target = (
            np.array(arrays[name].data, copy=True)
            if name in arrays
            else np.zeros(
                self.space.shape(name), dtype=float if var.type == "real" else np.int32
            )
        )
        ev = self.evaluation.evaluator(arrays, prev=prev)
        n = self.layout.count(var.level) if var.level else None
        local = self.space.locals_for(np.arange(1, n + 1), var.level) if var.level else {}
        local[STEP_NAME] = np.int64(k)
        local[PREV] = self.evaluation.state_namespace(PREV, prev)
        where = f"[scenario.{scenario.name}].{table}.{name} (step {k})"
        try:
            value = ev(expr.node, local, n)
        except ExprError as exc:
            raise SpecError(f"{where}: {exc}") from None
        self.space.assign(name, target, value, None, ev.elemental(expr.node), where)
        return target
