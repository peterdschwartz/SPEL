"""A proptest session (`Session`): one spec on one case, build and configuration, its synthetic
domain and a backend running the routines, and how a check is judged on an example.

One Hypothesis example = one call of the routine on the whole synthetic domain (`steps` > 1: a
run of that many calls, `Stepping.run_steps`):

1. draw a value for every element of every ruled input (`Drawing.draw_example`, strategies.py),
   parameters fixed unless ruled;
2. `derive` rules, evaluated element-wise over every element;
3. the check prepares the example (checks.py). A relation fills the follow-up half of the sets
   (sets nsets/2..nsets-1) with a copy of the base half, then the `follow` transforms, then
   `derive` again (except for the transformed variables);
4. `assume` masks the elements whose inputs violate it (an example with none left is rejected);
5. reset the case, set the inputs, compute (in the worker subprocess; `Stepping.compute`);
6. evaluate the check's expression per element (`Session.judge`); a failing element raises
   `PropertyFailure` with its set, pft, used values and drawn inputs (reporting.py), which
   Hypothesis then shrinks. The shrunk example is saved (`Session.save_failure`, failures.py)
   and can be run again without Hypothesis (`orchestration.replay`).

The domain (hierarchy.Layout) is ELM's hierarchy of declared gridcells: every level has its own
element count. The element variable of the level a check runs over is the element; those of
coarser levels are its ancestors (the layout's, which the subgrid index fields `veg_pp%column`,
... hold too). A finer level's variable is bound only when it is the same element (counts are
equal: one child each). The halves of a relation are the first and second half of the sets at
every level; `permute` exchanges whole gridcells of the same declaration.

The work is done by collaborators, each with explicit inputs: `ElementSpace` (elements.py),
`Evaluation` (evaluation.py), `Drawing` (drawing.py), `Stepping` (stepping.py) and `FailureStore`
(failures.py). Judging stays here: the check kinds' hooks take the session.

`Session` talks to the routine through a backend: `get(names)`, `run(inputs, outputs, trap)`,
`begin`/`step` (multi-step runs), `stop()`. The default is a `worker.Worker` on the case's
library; tests pass their own.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np

from spel.proptest import strategies as strat
from spel.proptest.checks import INPUT, PREV, Check, Expr
from spel.proptest.domain import SynthDomain, build_domain
from spel.proptest.errors import SpecError
from spel.proptest.fexpr import Evaluator, ExprError, FArray, Namespace, uses
from spel.proptest.inventory import Inventory
from spel.proptest.paths import data_dir, slug, work_dir
from spel.proptest.rules import Rule
from spel.proptest.runner.drawing import Drawing, Example
from spel.proptest.runner.elements import ElementSpace
from spel.proptest.runner.evaluation import Evaluation, _per_element, _use_node
from spel.proptest.runner.failures import FailureStore
from spel.proptest.runner.reporting import Failure, FailureSignature, PropertyFailure, _indent, report, signature
from spel.proptest.runner.stepping import JudgedStep, StepRun, Stepping
from spel.proptest.spec import INPUT_KINDS, ELEMENT_VARS, STEP_NAME, Relation, Scenario, Spec, missing_rules
from spel.proptest.worker import FortranAbort, Worker

if TYPE_CHECKING:
    from spel.proptest.results import BuiltDomain, Result, Verdict


@dataclass
class ProbeResult:
    """What a few examples of a scenario do (`Session.probe`)."""

    scenario: str
    examples: int
    elements: int  # selected by `over`
    assume_pass: float  # fraction of them passing `assume`, over the examples
    outputs: int
    never_written: list[
        str
    ]  # real outputs still NaN at every checked element of every example
    partly_written: dict[
        str, float
    ]  # output -> fraction of (element, example) left NaN
    unchanged: list[str]  # inout variables that never differ from their input
    constant: dict[str, float]  # outputs with one value everywhere
    aborted: list[str] = field(default_factory=list)

    def text(self) -> str:
        lines = [
            f"scenario {self.scenario}: {self.examples} examples (the first at every rule's midpoint), "
            f"{self.elements} elements, {self.assume_pass:.0%} pass `assume`"
        ]
        for msg in self.aborted:
            lines.append("  the routine aborted:\n" + _indent(msg, "    "))
        lines.append(
            f"  {self.outputs} outputs; never written on the checked elements: {len(self.never_written)}"
        )
        lines += [f"    {n}" for n in self.never_written]
        if self.partly_written:
            lines.append("  written on some elements only (fraction left NaN):")
            lines += [
                f"    {n}: {f:.0%}" for n, f in sorted(self.partly_written.items())
            ]
        if self.unchanged:
            lines.append("  read and written, but never changed:")
            lines += [f"    {n}" for n in self.unchanged]
        if self.constant:
            lines.append("  the same value at every element of every example:")
            lines += [f"    {n} = {v!r}" for n, v in sorted(self.constant.items())]
        return "\n".join(lines)


class Session:
    """A spec, its case's library, the synthetic domain and a backend running the case."""

    def __init__(
        self,
        spec: Spec,
        inv: Inventory,
        fc: Optional[str] = None,
        rebuild: bool = False,
        quiet: bool = True,
        no_build: bool = False,
        acc: Optional[str] = None,
        dbg: Optional[bool] = None,
        backend: Any = None,
        other: Optional["Session"] = None,
        namelist: Optional[dict[str, Any]] = None,
    ):
        """The library is rebuilt when the case's sources, its analysis or the generated shim
        changed since the last build (an incremental make), or when `rebuild`; `no_build` uses
        the library as it is (it must exist). fc/acc/dbg choose the build (first the arguments,
        then the spec's `[case] build`).
        namelist: options over the spec's `[namelist]` (a scenario's, a relation's follow-up).
        The session's inventory is the case's under that configuration (`Inventory.configured`),
        and its domain's constants file carries the options.
        backend: an object with get/run/stop instead of a worker on the library (nothing is
        built), or a function `options -> backend`, called with the options the spec and
        `namelist` set (siblings under other options then get their own).
        other: the session of `[compare]`."""
        from spel.proptest import build
        from spel.proptest import spec as specmod

        self.spec, self.base_inv = spec, inv
        self.namelist = dict(namelist or {})
        self.options = specmod.options(
            spec, extra=self.namelist
        )  # what the spec sets, for the constants file
        self.inv = inv.configured(specmod.constants(spec, inv, extra=self.namelist))
        self.case_dir = spec.case_dir
        self.work = work_dir(self.case_dir)
        self.data = data_dir(self.case_dir)
        self._build_args = dict(rebuild=rebuild, quiet=quiet, no_build=no_build)
        self.variant = build.Variant.of(spec.build, fc=fc, acc=acc, dbg=dbg)
        if backend is None:
            self.lib = build.library_path(self.case_dir, self.variant)
            why = (
                "--rebuild"
                if rebuild
                else build.stale(self.case_dir, inv, self.variant)
            )
            if why and no_build and self.lib.is_file():
                print(
                    f"spel proptest: using {self.lib} as it is ({why})", file=sys.stderr
                )
            elif why:
                print(
                    f"spel proptest: building {self.lib} ({why})",
                    file=sys.stderr,
                    flush=True,
                )
                build.build(self.case_dir, inv, quiet=quiet, variant=self.variant)
        from spel.proptest.shim import direct_init

        init = spec.domain.init or ("direct" if direct_init(inv) else "files")
        self.domain_spec = dataclasses.replace(
            spec.domain, constants={**spec.domain.constants, **self.options}, init=init
        )
        self.domain: SynthDomain = build_domain(
            self.inv, self.domain_spec, self.work / f"domain-{self._domain_key()}"
        )
        self.layout = self.domain.layout
        self._backend_arg = backend
        if backend is None:
            backend = Worker(
                self.lib,
                self.case_dir / "spel_py_manifest.json",
                self.domain.path_arg,
                log_dir=self.work / "logs",
                env=build.runtime_environment(self.case_dir, self.variant),
                direct=self.domain.direct,
            )
        elif callable(backend) and not hasattr(backend, "run"):
            backend = backend(dict(self.options))
        self.backend = backend
        self._other = other
        self._siblings: dict[tuple, "Session"] = {}
        # the step judged now (shared with siblings and `other`)
        self.judged = JudgedStep()
        # expressions a check builds for this configuration
        self.exprs: dict[tuple, Expr] = {}
        self._built = None  # results.BuiltDomain, on first use
        self.failure_suffix = ""  # drawn domains: failures/<scenario>__<check><suffix>
        # drawn domains: which one this session runs (saved with its failures)
        self.drawn = ""

        # the collaborators, bottom-up: each uses only those before it
        self.space = ElementSpace(self.layout, self.domain, self.inv, self.backend)
        self.evaluation = Evaluation(self.space, spec, self.inv, self.domain, self.case_dir)
        self.drawing = Drawing(self.space, self.evaluation, spec, self.inv, self.domain)
        self.stepping = Stepping(self.backend, self.space, self.evaluation, spec, self.inv, inv, self.domain,
                                 self.domain_spec, self.variant, self.judged)
        self.failures = FailureStore(self.space, spec, self.inv, self.domain, self.domain_spec, self.case_dir,
                                     self.data, self.options)

    @property
    def worker(self):  # the former name
        return self.backend

    def _domain_key(self) -> str:
        text = json.dumps(self.domain_spec.to_table(), sort_keys=True, default=str)
        return hashlib.sha1(text.encode()).hexdigest()[:10]

    @property
    def other(self) -> "Session":
        """The session of the spec's `[compare]` case or build (differential checks), on the same
        domain specification. Made on first use."""
        if self._other is None:
            cmp = self.spec.compare
            if cmp is None:
                raise SpecError("the spec has no [compare] table")
            inv = (
                self.base_inv
                if cmp.case_dir == self.case_dir
                else Inventory.load(cmp.case_dir, units=False)
            )
            spec = dataclasses.replace(
                self.spec,
                case_dir=cmp.case_dir,
                build=cmp.build,
                compare=None,
                checks=[],
            )
            self._other = Session(spec, inv, namelist=self.namelist, **self._build_args)
            self._other._share_judged(self.judged)
        return self._other

    def sibling(self, namelist: dict[str, Any]) -> "Session":
        """A session of the same case and build under further namelist options (its own worker
        process: the options are module scalars). Made on first use, closed with this one.
        """
        if not namelist:
            return self
        key = tuple(sorted((k, repr(v)) for k, v in namelist.items()))
        if key not in self._siblings:
            args = dict(self._build_args, rebuild=False)
            self._siblings[key] = Session(
                self.spec,
                self.base_inv,
                fc=self.variant.fc or None,
                acc=self.variant.acc,
                dbg=self.variant.dbg,
                backend=self._backend_arg,
                namelist={**self.namelist, **namelist},
                **args,
            )
            self._siblings[key]._share_judged(self.judged)
        return self._siblings[key]

    def for_scenario(self, scenario: Scenario) -> "Session":
        """The session a scenario runs in: this one, or its sibling under the scenario's options."""
        return self.sibling(scenario.namelist)

    def close(self) -> None:
        self.backend.stop()
        for s in self._siblings.values():
            s.close()
        if self._other is not None:
            self._other.close()

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def checks(self, scenario: Scenario) -> list[Check]:
        return [
            c
            for c in self.spec.checks
            if scenario in self.spec.scenarios_of(c.scenarios)
        ]

    def _spaces_at(
        self, scenario: Scenario, check: Check, ex: Example, j: int
    ) -> dict[str, Namespace]:
        """The check's namespaces (base%, follow%, ...) after step j."""
        k = self.judged.k
        self.judged.k = j
        try:
            return check.namespaces(
                self, scenario, ex, self.space.arrays(ex, self.stepping.compute(scenario, ex, []))
            )
        finally:
            self.judged.k = k

    def _history_at(
        self, scenario: Scenario, check: Check, ex: Example, j: int
    ) -> tuple:
        """(arrays, namespaces, (arrays, namespaces) before it) of step j, for acc(x)."""
        run = self.stepping.run_steps(scenario, ex)
        before = run.states[j - 2] if j > 1 else self.space.farrays(ex.inputs)
        spaces = self._spaces_at(scenario, check, ex, j)
        spaces[INPUT] = self.evaluation.inputs_namespace(ex)
        spaces[PREV] = self.evaluation.state_namespace(PREV, before)
        prev_spaces = self._spaces_at(scenario, check, ex, j - 1) if j > 1 else {}
        prev_spaces[INPUT] = spaces[INPUT]
        prev_spaces[PREV] = spaces[PREV]
        return run.states[j - 1], spaces, (before, prev_spaces)

    def check_example(self, scenario: Scenario, check: Check, draw: Callable) -> None:
        ex = check.prepare(
            self, scenario, self.drawing.draw_example(scenario, strat.HypothesisChooser(draw))
        )
        self.judge(scenario, check, ex)

    def judge(
        self, scenario: Scenario, check: Check, ex: Example, hypothesis: bool = True
    ) -> None:
        """Run a prepared example and evaluate the check per element: PropertyFailure carrying the
        `Failure` (with its report) when an element fails or the routine aborts. hypothesis=False: outside a
        Hypothesis test (replay), an example without an element to judge is an error."""
        over = check.over or (scenario.level, scenario.filter)
        elems = check.elements(self, ex, self.space.elements(over), over[0])
        mask = check.mask(self, scenario, ex, elems, over[0])
        if hypothesis:
            from hypothesis import assume, note

            assume(bool(mask.any()))
            note(f"input modes: {strat.describe_modes(ex.modes)}")
        elif not mask.any():
            raise SpecError(
                f"{check.name!r}: no element of the example passes `over` and `assume`"
            )
        verdict, ctx = self._evaluate(scenario, check, ex, elems, mask, over[0])
        if verdict.aborted:
            raise PropertyFailure(
                Failure(
                    scenario=scenario.name,
                    check=check.name,
                    example=ex,
                    message=f"{check.name}: the routine aborted\n{verdict.aborted}",
                    signature=FailureSignature(aborted=True),
                )
            )
        bad = np.flatnonzero(mask & ~verdict.ok)
        if bad.size:
            text, node, ev, local = ctx
            msg = report(
                self.space, check, text, node, ev, local, elems, bad, ex, mask, over[0]
            )
            if scenario.steps > 1:
                head, _, rest = msg.partition("\n")
                msg = f"{head}\n  after step {verdict.step} of {scenario.steps}\n{rest}"
            one = {
                v: (np.asarray(x)[bad[:1]] if isinstance(x, np.ndarray) else x)
                for v, x in local.items()
            }
            raise PropertyFailure(
                Failure(
                    scenario=scenario.name,
                    check=check.name,
                    example=ex,
                    message=msg,
                    element=int(elems[bad[0]]),
                    signature=signature(self.space, check, node, ev, one),
                )
            )

    def evaluate(self, scenario: Scenario, check: Check, ex: Example) -> "Verdict":
        """The check on a prepared example, per element, without raising (results.Verdict)."""
        over = check.over or (scenario.level, scenario.filter)
        elems = check.elements(self, ex, self.space.elements(over), over[0])
        mask = check.mask(self, scenario, ex, elems, over[0])
        verdict, ctx = self._evaluate(scenario, check, ex, elems, mask, over[0])
        if ctx is not None:
            text, node, ev, local = ctx
            for u in uses(node):
                if u.path[0] in ELEMENT_VARS or (
                    u.applied
                    and len(u.path) == 1
                    and self.space.canonical(tuple(u.path)) is None
                ):
                    continue
                head = u.path[0] if u.path[0] in (*check.names, INPUT) else None
                name = self.space.canonical(tuple(u.path[1:] if head else u.path))
                label = (f"{head}%" if head else "") + (
                    name or "%".join(u.path[1:] if head else u.path)
                )
                if label in verdict.values:
                    continue
                try:
                    verdict.values[label] = np.asarray(
                        ev(_use_node(node, u), local, len(elems))
                    )
                except ExprError:
                    continue
        return verdict

    def _evaluate(
        self,
        scenario: Scenario,
        check: Check,
        ex: Example,
        elems: np.ndarray,
        mask: np.ndarray,
        level: str,
    ) -> tuple["Verdict", Optional[tuple]]:
        """The check after every step it judges (one step: after the call); the first failing step's verdict."""
        if scenario.steps <= 1:
            return self._evaluate_at(scenario, check, ex, elems, mask, level, 1)
        out = None
        for k in check.judged_steps(scenario.steps):
            self.judged.k = k
            try:
                out = self._evaluate_at(scenario, check, ex, elems, mask, level, k)
            finally:
                self.judged.k = None
                self.evaluation.time_now = None
            if not out[0].passed:
                return out
        if out is None:
            raise SpecError(
                f"{check.name!r}: steps = {check.steps!r} selects no step of {scenario.steps}"
            )
        return out

    def _evaluate_at(
        self,
        scenario: Scenario,
        check: Check,
        ex: Example,
        elems: np.ndarray,
        mask: np.ndarray,
        level: str,
        k: int,
    ) -> tuple["Verdict", Optional[tuple]]:
        from spel.proptest.results import Verdict

        judged = check.expr_for(self, scenario)
        text, node = judged.text, judged.node
        outputs = self.evaluation.needed([judged, *scenario.assume], (*check.names, INPUT, PREV))
        try:
            arrays = self.space.arrays(ex, self.stepping.compute(scenario, ex, outputs))
            spaces = check.namespaces(self, scenario, ex, arrays)
            spaces[INPUT] = self.evaluation.inputs_namespace(ex)
            prev_spaces: dict[str, Namespace] = {INPUT: spaces[INPUT]}
            history = None
            if scenario.steps > 1:
                run = self.stepping.run_steps(scenario, ex)
                prev = run.states[k - 2] if k > 1 else self.space.farrays(ex.inputs)
                if k > 1 and any(u.path[0] in check.names for u in uses(node)):
                    prev_spaces.update(self._spaces_at(scenario, check, ex, k - 1))
                if any(u.applied and u.path == ("acc",) for u in uses(node)):
                    history = [
                        self._history_at(scenario, check, ex, j)
                        for j in range(1, k + 1)
                    ]
                self.judged.k = k
                self.evaluation.time_now = run.times[k - 1]
            else:
                prev = self.space.farrays(ex.inputs)
            spaces[PREV] = self.evaluation.state_namespace(PREV, prev)
            prev_spaces[PREV] = self.evaluation.state_namespace(PREV, prev)
        except FortranAbort as exc:
            return (
                Verdict(
                    check=check.name, level=level, elements=elems, mask=mask, ok=None, aborted=str(exc), step=k
                ),
                None,
            )
        ev = self.evaluation.evaluator(
            arrays,
            self.evaluation.context_functions(scenario, ex, level),
            prev=(prev, prev_spaces),
            history=history,
        )
        local = self.space.locals_for(elems, level)
        local.update(spaces)
        local[STEP_NAME] = np.int64(k)
        try:
            ok = _per_element(
                ev(node, local, len(elems)), elems.shape, f"{check.name!r}: {text!r}"
            )
        except ExprError as exc:
            raise SpecError(f"{check.name!r}: {text!r}: {exc}") from None
        return Verdict(check=check.name, level=level, elements=elems, mask=mask, ok=ok, step=k), (
            text,
            node,
            ev,
            local,
        )

    def execute(
        self, scenario: Scenario | str, ex: Example, outputs: str | list[str] = "all"
    ) -> "Result":
        """Run an example and keep its inputs and the state after it (results.Result). outputs:
        "all" (every output and inout live here) or names."""
        from spel.proptest.results import Result

        scen = self.spec.scenarios[scenario] if isinstance(scenario, str) else scenario
        names = (
            self.stepping.live_outputs()
            if outputs == "all"
            else [self.inv.canonical(tuple(n.split("%"))) or n for n in outputs]
        )
        if scen.steps > 1:
            run = self.stepping.run_steps(scen, ex)
            return Result(
                session=self,
                scenario=scen,
                example=ex,
                arrays=self.space.arrays(ex, run.states[-1]),
                steps=run.states,
                times=run.times,
            )
        state = self.stepping.compute(scen, ex, list(names))
        return Result(session=self, scenario=scen, example=ex, arrays=self.space.arrays(ex, state))

    def built(self) -> "BuiltDomain":
        """The domain as the case built it at init, with `check()` against the declaration."""
        from spel.proptest.results import built

        if self._built is None:
            self._built = built(self)
        return self._built

    def probe(self, scenario: Scenario, n: int = 5, seed: int = 0) -> ProbeResult:
        """Run `n` examples (the first with every input at its rule's midpoint, the others drawn
        uniformly) and say what the routine did on the scenario's elements: which outputs it
        never wrote, which inout variables it never changed, which outputs never vary.
        """
        self.preflight(scenario)
        rng = np.random.default_rng(seed)
        elems = self.space.elements((scenario.level, scenario.filter))
        names = self.stepping.live_outputs()
        unwritten: dict[str, list[int]] = {}
        seen: dict[str, int] = {}  # elements judged per output (its own level's)
        changed: dict[str, bool] = {}
        lo: dict[str, float] = {}
        hi: dict[str, float] = {}
        aborted: list[str] = []
        passing = total = judged = 0
        for k in range(max(1, n)):
            ex = self.drawing.inputs_at(scenario, 0.5 if k == 0 else rng.random)
            mask = self.drawing.assume_mask(scenario, ex, elems)
            total += elems.size
            passing += int(mask.sum())
            rows = elems[mask]
            if not rows.size:
                continue
            try:
                state = self.backend.run(
                    self.stepping.live_inputs(ex), names, trap=scenario.trap_fpe
                )
            except FortranAbort as exc:
                aborted.append(str(exc))
                continue
            judged += rows.size
            for name, fa in state.items():
                if fa is None:
                    continue
                var = self.inv.vars[name]
                axis = var.subgrid_axis
                own = self.space.elements_at(
                    scenario.level, rows, var.level
                )  # the output's elements there
                data = np.moveaxis(np.asarray(fa.data), axis, 0)[
                    own - fa.lb[axis]
                ].reshape(own.size, -1)
                if var.kind == "output" and var.type == "real":
                    missing = np.isnan(data).all(axis=1)
                    unwritten.setdefault(name, []).append(int(missing.sum()))
                    seen[name] = seen.get(name, 0) + own.size
                    data = data[~missing]
                elif name in ex.inputs:
                    before = np.moveaxis(np.asarray(ex.inputs[name]), axis, 0)[
                        own - 1
                    ].reshape(own.size, -1)
                    changed[name] = changed.get(name, False) or not np.array_equal(
                        before, data, equal_nan=True
                    )
                if var.kind == "output" and data.size:
                    vals = data[np.isfinite(data.astype(float))].astype(float)
                    if vals.size:
                        lo[name] = min(lo.get(name, np.inf), float(vals.min()))
                        hi[name] = max(hi.get(name, -np.inf), float(vals.max()))
        never = sorted(
            n_ for n_, c in unwritten.items() if seen.get(n_) and sum(c) == seen[n_]
        )
        partly = {
            n_: sum(c) / seen[n_]
            for n_, c in unwritten.items()
            if seen.get(n_) and 0 < sum(c) < seen[n_]
        }
        return ProbeResult(
            scenario=scenario.name,
            examples=max(1, n),
            elements=int(elems.size),
            assume_pass=passing / total if total else 0.0,
            outputs=len(names),
            never_written=never,
            partly_written=partly,
            unchanged=sorted(n_ for n_, c in changed.items() if not c),
            constant={
                n_: lo[n_] for n_ in sorted(lo) if lo[n_] == hi[n_] and n_ not in never
            },
            aborted=aborted,
        )

    def hypothesis_test(
        self, scenario: Scenario, check: Check, settings: Optional[dict] = None
    ) -> Callable:
        from hypothesis import HealthCheck, given
        from hypothesis import settings as hsettings
        from hypothesis import strategies as st
        from hypothesis.database import DirectoryBasedExampleDatabase

        cfg = dict(self.spec.settings)
        cfg.update(settings or {})
        seed = cfg.pop("seed", None)
        label = f"{scenario.name}__{check.name}"
        # one example database per check of a spec: its failing examples are replayed first
        key = hashlib.sha1(f"{self.spec.path}:{label}".encode()).hexdigest()[:8]
        opts = dict(
            max_examples=int(cfg.pop("max_examples", 100)),
            deadline=None,
            suppress_health_check=list(HealthCheck),
            database=DirectoryBasedExampleDatabase(
                str(self.data / "hypothesis" / f"{slug(label)[:60]}-{key}")
            ),
            print_blob=bool(cfg.pop("print_blob", False)),
        )
        cfg.pop("deadline", None)
        opts.update(cfg)

        def test(data):
            self.check_example(scenario, check, data.draw)

        test.__name__ = "test_" + slug(label)
        test.__qualname__ = test.__name__
        wrapped = hsettings(**opts)(given(st.data())(test))
        if seed is not None:
            from hypothesis import seed as hseed

            wrapped = hseed(int(seed))(wrapped)
        return wrapped

    def preflight(
        self, scenario: Scenario, checks: Optional[list[Check]] = None
    ) -> None:
        """Refuse a scenario that would run on a silently defaulted input. checks: the ones that
        will run (default: all of the scenario's)."""
        pinned = self.domain.pinned
        ruled = sorted(
            n
            for n in [*scenario.inputs, *(v for v, _ in scenario.derive)]
            if n in pinned
        )
        if ruled:
            raise SpecError(
                f"[scenario.{scenario.name}]: {', '.join(ruled)} are fixed by the domain (the declared "
                "topounit elevations decide downhill_ti); remove their rules"
            )
        missing = [
            n
            for n in missing_rules(scenario, self.inv, self.domain.constants)
            if n not in pinned
        ]
        if missing:
            raise SpecError(
                f"[scenario.{scenario.name}]: inputs without a rule (add them, or `extends` the "
                f"stats rule file): {', '.join(missing)}"
            )
        filled = {
            n: why for n, why in self.domain.filled.items() if n not in scenario.inputs
        }
        if filled:
            raise SpecError(
                f"[scenario.{scenario.name}]: parameters the routines read have no value (give each a rule "
                "under `inputs`, or use a reference capture that has them): "
                + "; ".join(f"{n} ({why})" for n, why in sorted(filled.items()))
            )
        for check in self.checks(scenario) if checks is None else checks:
            check.preflight(self, scenario)

    def accepts(self, scenario: Scenario, source: "Session", where: str) -> None:
        """Whether this session can run the examples `source` makes for the scenario (the other
        side of a differential check, the follow-up of a relation under other options): the
        inputs it reads exist with the same shapes, and none of its own is left without a value.
        where: how messages name this session."""
        names = sorted(
            n
            for n in set(scenario.inputs) | {v for v, _ in scenario.derive}
            if n not in self.inv.vars or self.inv.vars[n].kind in INPUT_KINDS
        )
        absent = [
            n for n in names if n not in self.inv.vars or n not in self.domain.extents
        ]
        if absent:
            raise SpecError(
                f"{where}: it has no input {', '.join(absent)} (the scenario sets them)"
            )
        differ = [
            f"{n} {self.space.shape(n)} vs {source.shape(n)}"
            for n in names
            if n in source.domain.extents and self.space.shape(n) != source.shape(n)
        ]
        if differ:
            raise SpecError(
                f"{where}: other shapes than {source.case_dir.name}: {'; '.join(differ)}"
            )
        try:
            self.preflight(scenario, [])
        except SpecError as exc:
            raise SpecError(f"{where}: {exc}") from None

    def _share_judged(self, judged: JudgedStep) -> None:
        """Judge the same step as another session (a sibling or `other` of it)."""
        self.judged = self.stepping.judged = judged

    # -- saved failures
    def save_failure(self, failure: Failure) -> Path:
        """Write a failing example (`PropertyFailure.failure`; after a Hypothesis run, the shrunk
        one); FailureStore.save. `replay` runs it again."""
        return self.failures.save(failure, self.drawn, self.failure_suffix, lambda nl: self.sibling(nl).failures)

    def load_failure(self, directory: Path) -> tuple[dict, Example]:
        """A saved failure: its `example.json` and the example with the saved inputs; FailureStore.load."""
        return self.failures.load(directory, lambda nl: self.sibling(nl).failures)

    def follow_example(self, scenario: Scenario, rel: Relation, ex: Example, target: "Session") -> Example:
        """Drawing.follow_example, into `target`'s configuration."""
        return self.drawing.follow_example(scenario, rel, ex, target.drawing)

    # -- what other modules call on a session: the collaborators' methods

    def canonical(self, path: tuple[str, ...]) -> Optional[str]:
        """ElementSpace.canonical."""
        return self.space.canonical(path)

    def static(self, name: str) -> Optional[FArray]:
        """ElementSpace.static."""
        return self.space.static(name)

    def statics(self, names: list[str]) -> None:
        """ElementSpace.statics."""
        return self.space.statics(names)

    def shape(self, name: str) -> tuple[int, ...]:
        """ElementSpace.shape."""
        return self.space.shape(name)

    def lbounds(self, name: str) -> tuple[int, ...]:
        """ElementSpace.lbounds."""
        return self.space.lbounds(name)

    def elements(self, over: tuple[str, Optional[str]]) -> np.ndarray:
        """ElementSpace.elements."""
        return self.space.elements(over)

    def locals_for(self, elems: np.ndarray, level: str) -> dict[str, Any]:
        """ElementSpace.locals_for."""
        return self.space.locals_for(elems, level)

    def rows_of(self, level: str, k: int, other: str) -> np.ndarray:
        """ElementSpace.rows_of."""
        return self.space.rows_of(level, k, other)

    def elements_at(self, level: str, elems: np.ndarray, other: str) -> np.ndarray:
        """ElementSpace.elements_at."""
        return self.space.elements_at(level, elems, other)

    def half_count(self, level: str) -> int:
        """ElementSpace.half_count."""
        return self.space.half_count(level)

    def partner(
        self,
        elems: np.ndarray,
        level: str,
        permutation: Optional[list[int]] = None,
    ) -> np.ndarray:
        """ElementSpace.partner."""
        return self.space.partner(elems, level, permutation)

    def arrays(
        self, ex: Example, state: Optional[dict[str, FArray]] = None
    ) -> dict[str, FArray]:
        """ElementSpace.arrays."""
        return self.space.arrays(ex, state)

    def evaluator(
        self,
        arrays: dict[str, Any],
        functions: Optional[dict[str, Callable]] = None,
        prev: Any = None,
        history: Optional[list] = None,
    ) -> Evaluator:
        """Evaluation.evaluator."""
        return self.evaluation.evaluator(arrays, functions, prev, history)

    def parse(self, text: str, where: str, namespaces: tuple[str, ...] = ()) -> Expr:
        """Evaluation.parse."""
        return self.evaluation.parse(text, where, namespaces)

    def needed(
        self, exprs: list[Expr], namespaces: tuple[str, ...] = ("base", "follow")
    ) -> list[str]:
        """Evaluation.needed."""
        return self.evaluation.needed(exprs, namespaces)

    def half(
        self,
        arrays: dict[str, Any],
        which: int,
        permutation: Optional[list[int]] = None,
    ) -> Namespace:
        """Evaluation.half."""
        return self.evaluation.half(arrays, which, permutation)

    def context_functions(
        self, scenario: Scenario, ex: Example, level: Optional[str] = None
    ) -> dict[str, Callable]:
        """Evaluation.context_functions."""
        return self.evaluation.context_functions(scenario, ex, level)

    def inputs_at(self, scenario: Scenario, u: Any = 0.5) -> Example:
        """Drawing.inputs_at."""
        return self.drawing.inputs_at(scenario, u)

    def sample(self, scenario: Scenario | str, seed: int = 0) -> Example:
        """Drawing.sample."""
        return self.drawing.sample(scenario, seed)

    def draw_example(self, scenario: Scenario, choose: "strat.Chooser") -> Example:
        """Drawing.draw_example."""
        return self.drawing.draw_example(scenario, choose)

    def rule_shape(
        self, name: str, rule: Rule
    ) -> tuple[tuple[int, ...], Optional[np.ndarray]]:
        """Drawing.rule_shape."""
        return self.drawing.rule_shape(name, rule)

    def level_rules(self, name: str, rule: Rule) -> list[tuple[tuple, Rule]]:
        """Drawing.level_rules."""
        return self.drawing.level_rules(name, rule)

    def assume_mask(
        self, scenario: Scenario, ex: Example, elems: np.ndarray
    ) -> np.ndarray:
        """Drawing.assume_mask."""
        return self.drawing.assume_mask(scenario, ex, elems)

    def follow_up(self, scenario: Scenario, rel: Relation, ex: Example) -> Example:
        """Drawing.follow_up."""
        return self.drawing.follow_up(scenario, rel, ex)

    def permute_follow(self, ex: Example) -> None:
        """Drawing.permute_follow."""
        return self.drawing.permute_follow(ex)

    def compute(
        self, scenario: Scenario, ex: Example, outputs: list[str]
    ) -> dict[str, FArray]:
        """Stepping.compute."""
        return self.stepping.compute(scenario, ex, outputs)

    def live_inputs(self, ex: Example) -> dict[str, np.ndarray]:
        """Stepping.live_inputs."""
        return self.stepping.live_inputs(ex)

    def live_outputs(self) -> list[str]:
        """Stepping.live_outputs."""
        return self.stepping.live_outputs()

    def run_steps(self, scenario: Scenario, ex: Example) -> StepRun:
        """Stepping.run_steps."""
        return self.stepping.run_steps(scenario, ex)

    def capture_dir(self) -> Path:
        """FailureStore.capture_dir."""
        return self.failures.capture_dir()
