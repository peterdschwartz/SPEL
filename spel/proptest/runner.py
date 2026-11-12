"""Run the checks of a `<name>.proptest.toml` with Hypothesis.

One Hypothesis example = one call of the routine on the whole synthetic domain:

1. draw a value for every element of every ruled input (strategies.py), parameters fixed unless
   ruled;
2. `derive` rules, evaluated element-wise over every element;
3. the check prepares the example (checks.py). A relation fills the follow-up half of the sets
   (sets nsets/2..nsets-1) with a copy of the base half, then the `follow` transforms, then
   `derive` again (except for the transformed variables);
4. `assume` masks the elements whose inputs violate it (an example with none left is rejected);
5. reset the case, set the inputs, compute (in the worker subprocess);
6. evaluate the check's expression per element (`Session.judge`); a failing element raises
   `PropertyFailure` with its set, pft, used values and drawn inputs, which Hypothesis then
   shrinks. The shrunk example is saved (`Session.save_failure`) and can be run again without
   Hypothesis (`replay`).

The domain (hierarchy.Layout) is ELM's hierarchy of declared gridcells: every level has its own
element count. The element variable of the level a check runs over is the element; those of
coarser levels are its ancestors (the layout's, which the subgrid index fields `veg_pp%column`,
... hold too). A finer level's variable is bound only when it is the same element (counts are
equal: one child each). The halves of a relation are the first and second half of the sets at
every level; `permute` exchanges whole gridcells of the same declaration.

`Session` talks to the routine through a backend: `get(names)`, `run(inputs, outputs, trap)`,
`stop()`. The default is a `worker.Worker` on the case's library; tests pass their own.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np

from spel.proptest import strategies as strat
from spel.proptest.checks import INPUT, PREV, Check, Expr
from spel.proptest.domain import CONSTANTS_FILE, INPUTS_FILE, Domain, build_domain
from spel.proptest.errors import SpecError
from spel.proptest.fexpr import (ExprError, Evaluator, FArray, FuncExpression, Namespace, children, designator,
                                 evaluate_bound, parse_section, uses)
from spel.proptest.inventory import LEVELS, Inventory
from spel.proptest.paths import data_dir, slug, work_dir
from spel.proptest.rules import KINDS as RULE_KINDS
from spel.proptest.rules import Rule
from spel.proptest.spec import (ELEMENT_VARS, INPUT_KINDS, STEP_NAME, Property, Relation, Scenario,  # noqa: F401
                                Spec, dormant_rules, missing_rules)
from spel.proptest.worker import FortranAbort, Worker

if TYPE_CHECKING:
    from spel.proptest.results import BuiltDomain, Result, Verdict

STATE_KINDS = ("input", "inout", "output")
FAILURES_DIR = "failures"
EXAMPLE_FILE = "example.json"
FORCING_FILE = "forcing.npz"  # a multi-step example's forcing values (step axis first)


class PropertyFailure(AssertionError):
    pass


@dataclass
class Example:
    inputs: dict[str, np.ndarray]  # inst%field -> Fortran-shaped array
    u: dict[str, np.ndarray] = field(default_factory=dict)
    modes: dict[str, int] = field(default_factory=dict)
    seed: int = 0
    extra: dict[str, Any] = field(default_factory=dict)  # what a check kind adds (JSON values)
    follow: Optional["Example"] = None  # the follow-up of a relation that runs it under other namelist options
    forcing: dict[str, np.ndarray] = field(default_factory=dict)  # multi-step: inst%field -> (step, *shape)


@dataclass
class StepRun:
    """A multi-step run of an example: after each step, every value it set or fetched (the state the
    checks see), and the time scalars of the step."""

    example: Example
    scenario: str
    states: list[dict[str, FArray]]
    times: list[dict[str, Any]]


@dataclass
class Failure:
    scenario: str
    check: str
    example: Example
    message: str
    element: Optional[int] = None  # Fortran index of the first failing element


@dataclass
class ProbeResult:
    """What a few examples of a scenario do (`Session.probe`)."""

    scenario: str
    examples: int
    elements: int  # selected by `over`
    assume_pass: float  # fraction of them passing `assume`, over the examples
    outputs: int
    never_written: list[str]  # real outputs still NaN at every checked element of every example
    partly_written: dict[str, float]  # output -> fraction of (element, example) left NaN
    unchanged: list[str]  # inout variables that never differ from their input
    constant: dict[str, float]  # outputs with one value everywhere
    aborted: list[str] = field(default_factory=list)

    def text(self) -> str:
        lines = [f"scenario {self.scenario}: {self.examples} examples (the first at every rule's midpoint), "
                 f"{self.elements} elements, {self.assume_pass:.0%} pass `assume`"]
        for msg in self.aborted:
            lines.append("  the routine aborted:\n" + _indent(msg, "    "))
        lines.append(f"  {self.outputs} outputs; never written on the checked elements: {len(self.never_written)}")
        lines += [f"    {n}" for n in self.never_written]
        if self.partly_written:
            lines.append("  written on some elements only (fraction left NaN):")
            lines += [f"    {n}: {f:.0%}" for n, f in sorted(self.partly_written.items())]
        if self.unchanged:
            lines.append("  read and written, but never changed:")
            lines += [f"    {n}" for n in self.unchanged]
        if self.constant:
            lines.append("  the same value at every element of every example:")
            lines += [f"    {n} = {v!r}" for n, v in sorted(self.constant.items())]
        return "\n".join(lines)


class Session:
    """A spec, its case's library, the synthetic domain and a backend running the case."""

    def __init__(self, spec: Spec, inv: Inventory, fc: Optional[str] = None, rebuild: bool = False,
                 quiet: bool = True, no_build: bool = False, acc: Optional[str] = None, dbg: Optional[bool] = None,
                 backend: Any = None, other: Optional["Session"] = None, namelist: Optional[dict[str, Any]] = None):
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
        self.options = specmod.options(spec, extra=self.namelist)  # what the spec sets, for the constants file
        self.inv = inv.configured(specmod.constants(spec, inv, extra=self.namelist))
        self.case_dir = spec.case_dir
        self.work = work_dir(self.case_dir)
        self.data = data_dir(self.case_dir)
        self._build_args = dict(rebuild=rebuild, quiet=quiet, no_build=no_build)
        self.variant = build.Variant.of(spec.build, fc=fc, acc=acc, dbg=dbg)
        if backend is None:
            self.lib = build.library_path(self.case_dir, self.variant)
            why = "--rebuild" if rebuild else build.stale(self.case_dir, inv, self.variant)
            if why and no_build and self.lib.is_file():
                print(f"spel proptest: using {self.lib} as it is ({why})", file=sys.stderr)
            elif why:
                print(f"spel proptest: building {self.lib} ({why})", file=sys.stderr, flush=True)
                build.build(self.case_dir, inv, quiet=quiet, variant=self.variant)
        from spel.proptest.shim import direct_init

        init = spec.domain.init or ("direct" if direct_init(inv) else "files")
        self.domain_spec = dataclasses.replace(spec.domain, constants={**spec.domain.constants, **self.options},
                                               init=init)
        self.domain: Domain = build_domain(self.inv, self.domain_spec, self.work / f"domain-{self._domain_key()}")
        self.layout = self.domain.layout
        self._backend_arg = backend
        if backend is None:
            backend = Worker(self.lib, self.case_dir / "spel_py_manifest.json", self.domain.path_arg,
                             log_dir=self.work / "logs", env=build.runtime_environment(self.case_dir, self.variant),
                             direct=self.domain.direct)
        elif callable(backend) and not hasattr(backend, "run"):
            backend = backend(dict(self.options))
        self.backend = backend
        self._other = other
        self._siblings: dict[tuple, "Session"] = {}
        self._step: dict[str, Any] = {"k": None}  # the step judged now (shared with siblings and `other`)
        self._run: Optional[StepRun] = None  # the last multi-step run (one example at a time)
        self._time_now: Optional[dict[str, Any]] = None  # time scalars of the step evaluated now
        self._boundary_ok: set[str] = set()
        self.exprs: dict[tuple, Expr] = {}  # expressions a check builds for this configuration
        self._static: dict[str, Optional[FArray]] = {}
        self._built = None  # results.BuiltDomain, on first use
        self._canon: dict[tuple[str, ...], Optional[str]] = {}
        self._levels: dict[tuple[str, str], tuple] = {}
        self._subspaces: dict[tuple[str, str], dict] = {}
        self.last_failure: Optional[Failure] = None
        self._capture_dir: Optional[Path] = None

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
            inv = self.base_inv if cmp.case_dir == self.case_dir else Inventory.load(cmp.case_dir, units=False)
            spec = dataclasses.replace(self.spec, case_dir=cmp.case_dir, build=cmp.build, compare=None, checks=[])
            self._other = Session(spec, inv, namelist=self.namelist, **self._build_args)
            self._other._step = self._step
        return self._other

    def sibling(self, namelist: dict[str, Any]) -> "Session":
        """A session of the same case and build under further namelist options (its own worker
        process: the options are module scalars). Made on first use, closed with this one."""
        if not namelist:
            return self
        key = tuple(sorted((k, repr(v)) for k, v in namelist.items()))
        if key not in self._siblings:
            args = dict(self._build_args, rebuild=False)
            self._siblings[key] = Session(self.spec, self.base_inv, fc=self.variant.fc or None, acc=self.variant.acc,
                                          dbg=self.variant.dbg, backend=self._backend_arg,
                                          namelist={**self.namelist, **namelist}, **args)
            self._siblings[key]._step = self._step
        return self._siblings[key]

    def for_scenario(self, scenario: Scenario) -> "Session":
        """The session a scenario runs in: this one, or its sibling under the scenario's options."""
        return self.sibling(scenario.namelist)

    def parse(self, text: str, where: str, namespaces: tuple[str, ...] = ()) -> Expr:
        from spel.proptest.spec import Names

        return Names(self.inv, set(self.spec.functions)).expr(text, where, namespaces)

    def live_outputs(self) -> list[str]:
        """What the routines compute per element under this session's configuration."""
        consts = self.domain.constants
        return sorted(v.name for v in self.inv.of_kind("output", "inout")
                      if v.level and not v.driver and self.inv.guard_active(v, consts))

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

    # -- names and values
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

    def resolver(self, arrays: dict[str, Any]) -> Callable[[list[str]], Any]:
        constants = self.domain.constants
        time_names = self.inv.time_names() or ()

        def resolve(path: list[str]) -> Any:
            name = self.canonical(tuple(path))
            if name is not None:
                if name in arrays:
                    return arrays[name]
                if self.inv.vars[name].kind in STATE_KINDS:
                    raise ExprError(f"{name} is not available here")
                return self._as_value(self.static(name), name)
            if len(path) == 1:
                head = path[0]
                if self.inv.parameters.get(head) is not None:
                    return self.inv.parameters[head]
                if head in constants:
                    v = np.asarray(constants[head])
                    return FArray(v, (1,) * v.ndim, head) if v.ndim else v[()]
                if head in time_names:
                    if self._time_now is not None and head in self._time_now:
                        return self._time_now[head]
                    return self._as_value(self.static(head), head)
            if len(path) == 2 and path[0] == "filter":
                return self._as_value(self.static(f"filter%{path[1]}"), f"filter%{path[1]}")
            raise KeyError("%".join(path))

        return resolve

    @staticmethod
    def _as_value(fa: Optional[FArray], name: str) -> Any:
        if fa is None:
            raise ExprError(f"{name} is not allocated in the case")
        return fa.data[()] if fa.data.ndim == 0 else fa

    def evaluator(self, arrays: dict[str, Any], functions: Optional[dict[str, Callable]] = None,
                  prev: Any = None, history: Optional[list] = None) -> Evaluator:
        """prev: the state before the step (`tend(x)`): its arrays, or (arrays, namespaces);
        default: none. history: the states after steps 1..k as (arrays, namespaces), for `acc(x)`."""
        def env(e):
            if not isinstance(e, tuple):
                return self.resolver(e), {}
            if len(e) == 3:  # (arrays, namespaces, prev): a step of the history with its own prev
                return self.resolver(e[0]), e[1], env(e[2])
            return self.resolver(e[0]), e[1]

        return Evaluator(self.resolver(arrays), {**self.spec.functions, **self.domain_functions(), **(functions or {})},
                         prev=env(prev) if prev is not None else None, aggregate=self._aggregate,
                         history=[env(h) for h in history] if history else None)

    def domain_functions(self) -> dict[str, Callable]:
        """Topounit facts from the domain declaration: `elevation(t)` (declared elevation, NaN where
        none), `lowest(t)` (the lowest topounit of its gridcell, ties included), `uphill_count(t)`
        (other topounits whose top_pp%downhill_ti, as ELM computes it, is t), `downhill_inactive(t)`
        (t's downhill neighbour is an inactive, weight-0 topounit: set_topounit does not skip them)."""
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
            dead = np.array([0 < d <= elev.size and not act[d - 1] for d in dn], dtype=bool)
            self._domain_fns = {"elevation": at(elev), "lowest": at(lowest), "uphill_count": at(ups),
                                "downhill_inactive": at(dead)}
        return self._domain_fns

    def _aggregate(self, how: str, x: FArray, level: str) -> np.ndarray:
        """`child_<how>(x, level)`: x (a variable of a finer level) reduced over each element of level."""
        from spel.proptest.results import _aggregate

        var = self.inv.vars.get(x.name) if x.name else None
        if var is None or var.level is None:
            raise ExprError(f"child_{how}: {x.name or 'the argument'} is not a variable with a subgrid dimension")
        if LEVELS.index(var.level) <= LEVELS.index(level):
            raise ExprError(f"child_{how}: {x.name} is of level {var.level!r}, not finer than {level!r}")
        data = np.asarray(x.data, dtype=float)
        if data.ndim != 1:
            raise ExprError(f"child_{how}: {x.name} has more than one value per element")
        return _aggregate(self.layout, data, var.level, level, how)

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
                out[lev] = self.layout.parent(level, lev)[elems - 1] if elems.size else elems
            elif self.layout.one_to_one(level, lev):
                out[lev] = elems
        return out

    def checks(self, scenario: Scenario) -> list[Check]:
        return [c for c in self.spec.checks if scenario in self.spec.scenarios_of(c.scenarios)]

    # -- inputs
    def shape(self, name: str) -> tuple[int, ...]:
        return tuple(hi - lo + 1 for lo, hi in self.domain.extents[name])

    def lbounds(self, name: str) -> tuple[int, ...]:
        return tuple(lo for lo, _ in self.domain.extents[name])

    def rule_shape(self, name: str, rule: Rule) -> tuple[tuple[int, ...], Optional[np.ndarray]]:
        """The shape of the u a rule maps, and the variable's own value for a relative rule (`scale`)."""
        if RULE_KINDS[rule.kind].relative and self.inv.vars[name].kind in INPUT_KINDS:
            base = np.asarray(self.static(name).data, dtype=float)
            return base.shape, base
        return self.shape(name), None  # (a relative rule of a variable that is dormant here: only its u)

    def level_rules(self, name: str, rule: Rule) -> list[tuple[tuple, Rule]]:
        """A rule's `levels` overrides as (index into the variable's array, rule): each key is a
        section of the first dimension that is not the subgrid one."""
        out = []
        for text, sub in rule.by_level:
            key = (name, text)
            if key not in self._levels:
                var = self.inv.vars[name]
                axis = next(k for k, d in enumerate(var.dims) if not d.level)
                lb, n = self.lbounds(name)[axis], self.shape(name)[axis]
                env = self.inv.bound_env(self.layout.sizes(), self.domain.constants)
                env.update(self.spec.domain.dims)
                try:
                    start, end = parse_section(text)
                    lo = 0 if start is None else evaluate_bound(start, env) - lb
                    hi = n if end is None else evaluate_bound(end, env) - lb + 1
                except ExprError as exc:
                    raise SpecError(f"{name}: levels {text!r}: {exc}") from None
                if not 0 <= lo < hi <= n:
                    raise SpecError(f"{name}: levels {text!r} is outside {lb}:{lb + n - 1}")
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
            ex.inputs[name] = strat.values_of(rule, uu, self._pft(var), var.subgrid_axis, var.type, base,
                                              self.level_rules(name, rule))
        self.derive(scenario, ex, skip=set())
        self._forcing_at(scenario, ex, u)
        return ex

    def _forcing_at(self, scenario: Scenario, ex: Example, u: Any) -> None:
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
            ex.forcing[name] = np.stack([
                strat.values_of(rule, uu[k], self._pft(var), var.subgrid_axis, var.type, base,
                                self.level_rules(name, rule)) for k in range(scenario.steps)])

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
            ex.inputs[name] = strat.values_of(rule, uu, self._pft(var), var.subgrid_axis, var.type, base,
                                              self.level_rules(name, rule))

    def _pft(self, var) -> Optional[np.ndarray]:
        """The pft of each element of the variable's level (-1: urban, lake), for per-pft rules."""
        return self.layout.pft_of(var.level) if var.level else None

    def half_count(self, level: str) -> int:
        """Elements of a level in one half of the sets (the base half of a relation)."""
        return self.layout.count(level) // 2

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
            u[name], modes[name] = strat.draw_u(choose, name, rule, shape, rng, self.inv.vars[name].subgrid_axis)
        for name in sorted(scenario.forcing) if scenario.steps > 1 else ():
            rule = scenario.forcing[name]
            if name not in self.domain.extents:
                continue
            shape, _ = self.rule_shape(name, rule)
            axis = self.inv.vars[name].subgrid_axis
            key = f"forcing:{name}"  # a mode over (step, *shape): constant, i.i.d., edges, per element
            u[key], modes[key] = strat.draw_u(choose, key, rule, (scenario.steps, *shape), rng,
                                              None if axis is None else axis + 1)
        ex = self.inputs_at(scenario, u)
        ex.modes, ex.seed = modes, seed
        return ex

    def sample(self, scenario: Scenario | str, seed: int = 0) -> Example:
        """An example drawn like a test run's (the same modes), from `seed` instead of Hypothesis."""
        scen = self.spec.scenarios[scenario] if isinstance(scenario, str) else scenario
        return self.draw_example(scen, strat.RngChooser(np.random.default_rng(seed)))

    def _farrays(self, inputs: dict[str, np.ndarray]) -> dict[str, FArray]:
        return {n: FArray(a, self.lbounds(n) if n in self.domain.extents else (1,) * a.ndim, n)
                for n, a in inputs.items()}

    def arrays(self, ex: Example, state: Optional[dict[str, FArray]] = None) -> dict[str, FArray]:
        """The example's inputs and, after a run, the state it returned, as Fortran arrays by name."""
        out = self._farrays(ex.inputs)
        out.update(state or {})
        return out

    def _assign(self, name: str, target: np.ndarray, value: Any, rows: Optional[np.ndarray],
                elemental: bool, where: str) -> None:
        """A result into a Fortran-shaped array, at `rows` (0-based) of its subgrid axis.
        elemental: the value has one entry per element (elements first, `Evaluator.elemental`)
        and applies to every other dimension; otherwise it is the same for every element."""
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
            raise SpecError(f"{where}: a value of shape {value.shape} does not fit {name} "
                            f"{tuple(target.shape)}") from None

    def derive(self, scenario: Scenario, ex: Example, skip: set[str]) -> None:
        for name, expr in scenario.derive:
            if name in skip:
                continue
            var = self.inv.vars[name]
            if var.kind not in INPUT_KINDS:  # dormant under this configuration: nothing reads it
                continue
            if name not in ex.inputs:
                ex.inputs[name] = np.zeros(self.shape(name), dtype=float if var.type == "real" else np.int32)
            ev = self.evaluator(self._farrays(ex.inputs))
            n = self.layout.count(var.level) if var.level else None
            local = self.locals_for(np.arange(1, n + 1), var.level) if var.level else {}
            where = f"[scenario.{scenario.name}].derive {name}"
            try:
                value = ev(expr.node, local, n)
            except ExprError as exc:
                raise SpecError(f"{where}: {exc}") from None
            self._assign(name, ex.inputs[name], value, None, ev.elemental(expr.node), where)

    def follow_up(self, scenario: Scenario, rel: Relation, ex: Example) -> Example:
        """Sets nsets/2.. become a copy of the base sets, then the relation's transforms."""
        out = Example({k: v.copy() for k, v in ex.inputs.items()}, ex.u, ex.modes, ex.seed, dict(ex.extra))
        out.forcing = {k: v.copy() for k, v in ex.forcing.items()}
        for name, arr in out.inputs.items():
            var = self.inv.vars[name]
            if var.subgrid_axis is not None:
                half = self.half_count(var.level)
                t = np.moveaxis(arr, var.subgrid_axis, 0)
                t[half:2 * half] = t[:half]
        for name, arr in out.forcing.items():  # (step, *shape): the subgrid axis is one further
            var = self.inv.vars[name]
            if var.subgrid_axis is not None:
                half = self.half_count(var.level)
                t = np.moveaxis(arr, var.subgrid_axis + 1, 0)
                t[half:2 * half] = t[:half]
        values = []
        for name, expr in rel.follow:
            ev = self.evaluator(self._farrays(out.inputs))
            level = self.inv.vars[name].level
            half = self.half_count(level) if level else None
            local = self.locals_for(np.arange(1, half + 1), level) if level else {}
            try:
                values.append((name, ev(expr.node, local, half), ev.elemental(expr.node)))
            except ExprError as exc:
                raise SpecError(f"[[relation]] {rel.name!r} follow {name}: {exc}") from None
        for name, value, elemental in values:  # all transforms see the base values
            if name not in out.inputs:
                raise SpecError(f"[[relation]] {rel.name!r}: {name} is neither ruled nor derived")
            level = self.inv.vars[name].level
            half = self.half_count(level) if level else 0
            self._assign(name, out.inputs[name], value, np.arange(half, 2 * half) if level else None, elemental,
                         f"[[relation]] {rel.name!r} follow {name}")
        self.derive(scenario, out, skip={n for n, _ in rel.follow})
        return out

    def follow_example(self, scenario: Scenario, rel: Relation, ex: Example, target: "Session") -> Example:
        """The follow-up of a relation that runs it in another session (`target`: this case under
        other options): a copy of the example, the `follow` transforms over every element (each
        an expression of the base inputs), then the target's `derive`."""
        out = Example({k: v.copy() for k, v in ex.inputs.items()}, ex.u, ex.modes, ex.seed)
        target.fill_missing(scenario, out)
        ev = self.evaluator(self._farrays(ex.inputs))
        for name, expr in rel.follow:
            where = f"[[relation]] {rel.name!r} follow {name}"
            if name not in out.inputs:
                raise SpecError(f"[[relation]] {rel.name!r}: {name} is neither ruled nor derived")
            level = self.inv.vars[name].level
            n = self.layout.count(level) if level else None
            local = self.locals_for(np.arange(1, n + 1), level) if level else {}
            try:
                value = ev(expr.node, local, n)
            except ExprError as exc:
                raise SpecError(f"{where}: {exc}") from None
            target._assign(name, out.inputs[name], value, None, ev.elemental(expr.node), where)
        target.derive(scenario, out, skip={n_ for n_, _ in rel.follow})
        return out

    def permute_follow(self, ex: Example) -> None:
        """Reorder the follow-up gridcells among those of the same declaration (from the
        example's seed), each with everything under it: follow-up gridcell k then holds what
        gridcell `permutation[k]` held. Recorded in `ex.extra`."""
        half = self.half_count("g")
        labels = [g.label for g in self.layout.levels["g"].decl[:half]]
        rng = np.random.default_rng(ex.seed + 1)
        perm = np.arange(half)
        for label in sorted(set(labels)):
            rows = np.flatnonzero(np.asarray(labels) == label)
            perm[rows] = rng.permutation(rows)
        for name, arr in ex.inputs.items():
            var = self.inv.vars[name]
            if var.subgrid_axis is not None:
                h = self.half_count(var.level)
                t = np.moveaxis(arr, var.subgrid_axis, 0)
                t[h:2 * h] = t[h:2 * h][self.layout.row_permutation(var.level, perm)]
        ex.extra["permutation"] = perm.tolist()

    def partner(self, elems: np.ndarray, level: str, permutation: Optional[list[int]] = None) -> np.ndarray:
        """The follow-up element (Fortran index) of each base element of a level."""
        half = self.half_count(level)
        if permutation is None:
            return elems + half
        inverse = np.argsort(self.layout.row_permutation(level, np.asarray(permutation)))
        return inverse[elems - 1] + 1 + half

    def assume_mask(self, scenario: Scenario, ex: Example, elems: np.ndarray) -> np.ndarray:
        mask = np.ones(len(elems), dtype=bool)
        if not scenario.assume:
            return mask
        ev = self.evaluator(self._farrays(ex.inputs))
        local = self.locals_for(elems, scenario.level)
        for e in scenario.assume:
            where = f"[scenario.{scenario.name}].assume {e.text!r}"
            try:
                r = ev(e.node, local, len(elems))
            except ExprError as exc:
                raise SpecError(f"{where}: {exc}") from None
            mask &= _per_element(r, mask.shape, where)
        return mask

    # -- running
    def needed(self, exprs: list[Expr], namespaces: tuple[str, ...] = ("base", "follow")) -> list[str]:
        """State variables the expressions read after compute (fetched from the backend)."""
        out = set()
        for e in exprs:
            for u in uses(e.node):
                path = u.path[1:] if u.path[0] in namespaces else u.path
                if not path:
                    continue
                name = self.canonical(tuple(path))
                if name is not None and self.inv.vars[name].kind in STATE_KINDS:
                    out.add(name)
        return sorted(out)

    def live_inputs(self, ex: Example) -> dict[str, np.ndarray]:
        """The example's inputs this session's configuration reads (a rule for a variable that
        is dead here is dormant: its value is drawn but not set)."""
        return {n: a for n, a in ex.inputs.items() if self.inv.vars[n].kind in INPUT_KINDS}

    def compute(self, scenario: Scenario, ex: Example, outputs: list[str]) -> dict[str, FArray]:
        if scenario.steps > 1:  # the state after the step judged now (default: the last)
            run = self.run_steps(scenario, ex)
            k = self._step["k"] or scenario.steps
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

    # -- multi-step runs
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
            raise SpecError(f"[scenario.{scenario.name}]: multi-step runs need an ACC=OFF build for now "
                            "(the time scalars are not updated on the device between steps)")
        if getattr(self.base_inv, "unit_test", None) is not None:  # a synthetic inventory (tests) has none
            from spel.proptest import coupling
            from spel.proptest import spec as specmod

            values = specmod.constants(self.spec, self.base_inv, scenario)
            found = coupling.boundary(self.base_inv, values)
            undeclared = {n: w for n, w in found.items()
                          if self.inv.vars[n].kind in ("input", "inout") and not scenario.boundary_declared(n)}
            if undeclared:
                lines = [f"  {n}: {coupling.describe(w)}" for n, w in sorted(undeclared.items())]
                raise SpecError(
                    f"[scenario.{scenario.name}]: {len(undeclared)} variables the routines read are also written "
                    "by other elm_drv routines within a step; say what stands in for them between steps in "
                    "`forcing`, `between` or `keep` (SPEC.md, Multi-step runs):\n" + "\n".join(lines))
        self._boundary_ok.add(scenario.name)

    def step_names(self) -> list[str]:
        """What a multi-step run fetches after every step: every output and inout live here."""
        return list(self.live_outputs())

    def run_steps(self, scenario: Scenario, ex: Example) -> StepRun:
        """Run the example's steps (once per example: the run is kept for the checks of every step).
        Before step 1 the state after init with the example's inputs; before every step its time
        scalars and forcing values, after it the `between` assignments."""
        from spel.proptest.timestep import advance

        if self._run is not None and self._run.example is ex and self._run.scenario == scenario.name:
            return self._run
        self.check_boundary(scenario)
        names = self.step_names()
        # the outputs under this configuration, NaN before every step (not the manifest's, which
        # counts every branch: an input here can be an output there and must keep its value);
        # carried outputs keep what the previous step left where a step does not write them
        poison = sorted(n for n, v in self.inv.vars.items()
                        if v.kind == "output" and v.type == "real" and n not in scenario.carry)
        tnames = {e.lower() for e in (self.inv.time_names() or ())}
        time = self.start_time(scenario)
        before = self._farrays(ex.inputs)  # the state before step 1
        start = self.live_inputs(ex)
        start.update({n: ex.inputs[n] for n in scenario.carry  # dormant where nothing uses it
                      if n in ex.inputs and self.inv.vars[n].kind != "unused" and n in self.domain.extents})
        self.backend.begin(start)
        states: list[dict[str, FArray]] = []
        times: list[dict[str, Any]] = []
        pending: dict[str, np.ndarray] = {}  # `between` values for the next step
        for k in range(1, scenario.steps + 1):
            if k > 1:
                time = advance(time)
            sets: dict[str, Any] = dict(pending)
            sets.update({n: np.asarray(v) for n, v in time.items() if n in tnames})
            self._time_now = time
            try:
                for name, values in ex.forcing.items():
                    if self.inv.vars[name].kind in INPUT_KINDS:
                        sets[name] = np.asarray(values[k - 1])
                for name, expr in scenario.forcing_exprs:  # dormant where nothing reads the target
                    if self.inv.vars[name].kind in INPUT_KINDS:
                        sets[name] = self._step_assign(scenario, name, expr, before, before, k, "forcing")
                try:
                    state = self.backend.step(sets, names, trap=scenario.trap_fpe, poison=poison)
                except FortranAbort as exc:
                    raise FortranAbort(f"step {k} of {scenario.steps}: {exc}") from None
                after = dict(before)
                after.update({n: FArray(np.asarray(v), self.lbounds(n), n) for n, v in sets.items()
                              if n not in tnames and n in self.domain.extents})
                after.update({n: v for n, v in state.items() if v is not None})
                pending = {}
                for name, expr in scenario.between:
                    if self.inv.vars[name].kind not in INPUT_KINDS:
                        continue  # dormant under these options (ELM's own code is gated the same way)
                    value = self._step_assign(scenario, name, expr, after, before, k, "between")
                    after[name] = FArray(value, self.lbounds(name), name)
                    pending[name] = value
            finally:
                self._time_now = None
            states.append(after)
            times.append(dict(time))
            before = after
        self._run = StepRun(ex, scenario.name, states, times)
        return self._run

    def _step_assign(self, scenario: Scenario, name: str, expr: Expr, arrays: dict[str, FArray],
                     prev: dict[str, FArray], k: int, table: str) -> np.ndarray:
        """A `forcing` or `between` expression for every element of the target's level: bare names
        are `arrays`, `prev%x` is `prev`, `step` is k."""
        var = self.inv.vars[name]
        target = (np.array(arrays[name].data, copy=True) if name in arrays
                  else np.zeros(self.shape(name), dtype=float if var.type == "real" else np.int32))
        ev = self.evaluator(arrays, prev=prev)
        n = self.layout.count(var.level) if var.level else None
        local = self.locals_for(np.arange(1, n + 1), var.level) if var.level else {}
        local[STEP_NAME] = np.int64(k)
        local[PREV] = self._state_namespace(PREV, prev)
        where = f"[scenario.{scenario.name}].{table}.{name} (step {k})"
        try:
            value = ev(expr.node, local, n)
        except ExprError as exc:
            raise SpecError(f"{where}: {exc}") from None
        self._assign(name, target, value, None, ev.elemental(expr.node), where)
        return target

    def _spaces_at(self, scenario: Scenario, check: Check, ex: Example, j: int) -> dict[str, Namespace]:
        """The check's namespaces (base%, follow%, ...) after step j."""
        k = self._step["k"]
        self._step["k"] = j
        try:
            return check.namespaces(self, scenario, ex, self.arrays(ex, self.compute(scenario, ex, [])))
        finally:
            self._step["k"] = k

    def _history_at(self, scenario: Scenario, check: Check, ex: Example, j: int) -> tuple:
        """(arrays, namespaces, (arrays, namespaces) before it) of step j, for acc(x)."""
        run = self.run_steps(scenario, ex)
        before = run.states[j - 2] if j > 1 else self._farrays(ex.inputs)
        spaces = self._spaces_at(scenario, check, ex, j)
        spaces[INPUT] = self._inputs_namespace(ex)
        spaces[PREV] = self._state_namespace(PREV, before)
        prev_spaces = self._spaces_at(scenario, check, ex, j - 1) if j > 1 else {}
        prev_spaces[INPUT] = spaces[INPUT]
        prev_spaces[PREV] = spaces[PREV]
        return run.states[j - 1], spaces, (before, prev_spaces)

    def _state_namespace(self, label: str, arrays: dict[str, FArray]) -> Namespace:
        def get(path: list[str]) -> Any:
            name = self.canonical(tuple(path))
            if name is None or name not in arrays:
                raise ExprError(f"{label}%{'%'.join(path)}: not available")
            return arrays[name]

        return Namespace(label, get)

    def check_example(self, scenario: Scenario, check: Check, draw: Callable) -> None:
        ex = check.prepare(self, scenario, self.draw_example(scenario, strat.HypothesisChooser(draw)))
        self.judge(scenario, check, ex)

    def judge(self, scenario: Scenario, check: Check, ex: Example, hypothesis: bool = True) -> None:
        """Run a prepared example and evaluate the check per element: PropertyFailure with a
        report when an element fails or the routine aborts. hypothesis=False: outside a
        Hypothesis test (replay), an example without an element to judge is an error."""
        over = check.over or (scenario.level, scenario.filter)
        elems = check.elements(self, ex, self.elements(over), over[0])
        mask = check.mask(self, scenario, ex, elems, over[0])
        if hypothesis:
            from hypothesis import assume, note

            assume(bool(mask.any()))
            note(f"input modes: {strat.describe_modes(ex.modes)}")
        elif not mask.any():
            raise SpecError(f"{check.name!r}: no element of the example passes `over` and `assume`")
        verdict, ctx = self._evaluate(scenario, check, ex, elems, mask, over[0])
        if verdict.aborted:
            msg = f"{check.name}: the routine aborted\n{verdict.aborted}"
            self.last_failure = Failure(scenario.name, check.name, ex, msg)
            raise PropertyFailure(msg)
        bad = np.flatnonzero(mask & ~verdict.ok)
        if bad.size:
            text, node, ev, local = ctx
            msg = self.report(check, text, node, ev, local, elems, bad, ex, mask, over[0])
            if scenario.steps > 1:
                head, _, rest = msg.partition("\n")
                msg = f"{head}\n  after step {verdict.step} of {scenario.steps}\n{rest}"
            self.last_failure = Failure(scenario.name, check.name, ex, msg, int(elems[bad[0]]))
            raise PropertyFailure(msg)

    def evaluate(self, scenario: Scenario, check: Check, ex: Example) -> "Verdict":
        """The check on a prepared example, per element, without raising (results.Verdict)."""
        over = check.over or (scenario.level, scenario.filter)
        elems = check.elements(self, ex, self.elements(over), over[0])
        mask = check.mask(self, scenario, ex, elems, over[0])
        verdict, ctx = self._evaluate(scenario, check, ex, elems, mask, over[0])
        if ctx is not None:
            text, node, ev, local = ctx
            for u in uses(node):
                if u.path[0] in ELEMENT_VARS or (u.applied and len(u.path) == 1 and self.canonical(tuple(u.path)) is None):
                    continue
                head = u.path[0] if u.path[0] in (*check.names, INPUT) else None
                name = self.canonical(tuple(u.path[1:] if head else u.path))
                label = (f"{head}%" if head else "") + (name or "%".join(u.path[1:] if head else u.path))
                if label in verdict.values:
                    continue
                try:
                    verdict.values[label] = np.asarray(ev(_use_node(node, u), local, len(elems)))
                except ExprError:
                    continue
        return verdict

    def _evaluate(self, scenario: Scenario, check: Check, ex: Example, elems: np.ndarray, mask: np.ndarray,
                  level: str) -> tuple["Verdict", Optional[tuple]]:
        """The check after every step it judges (one step: after the call); the first failing step's verdict."""
        if scenario.steps <= 1:
            return self._evaluate_at(scenario, check, ex, elems, mask, level, 1)
        out = None
        for k in check.judged_steps(scenario.steps):
            self._step["k"] = k
            try:
                out = self._evaluate_at(scenario, check, ex, elems, mask, level, k)
            finally:
                self._step["k"] = None
                self._time_now = None
            if not out[0].passed:
                return out
        if out is None:
            raise SpecError(f"{check.name!r}: steps = {check.steps!r} selects no step of {scenario.steps}")
        return out

    def _evaluate_at(self, scenario: Scenario, check: Check, ex: Example, elems: np.ndarray, mask: np.ndarray,
                     level: str, k: int) -> tuple["Verdict", Optional[tuple]]:
        from spel.proptest.results import Verdict

        judged = check.expr_for(self, scenario)
        text, node = judged.text, judged.node
        outputs = self.needed([judged, *scenario.assume], (*check.names, INPUT, PREV))
        try:
            arrays = self.arrays(ex, self.compute(scenario, ex, outputs))
            spaces = check.namespaces(self, scenario, ex, arrays)
            spaces[INPUT] = self._inputs_namespace(ex)
            prev_spaces: dict[str, Namespace] = {INPUT: spaces[INPUT]}
            history = None
            if scenario.steps > 1:
                run = self.run_steps(scenario, ex)
                prev = run.states[k - 2] if k > 1 else self._farrays(ex.inputs)
                if k > 1 and any(u.path[0] in check.names for u in uses(node)):
                    prev_spaces.update(self._spaces_at(scenario, check, ex, k - 1))
                if any(u.applied and u.path == ("acc",) for u in uses(node)):
                    history = [self._history_at(scenario, check, ex, j) for j in range(1, k + 1)]
                self._step["k"] = k
                self._time_now = run.times[k - 1]
            else:
                prev = self._farrays(ex.inputs)
            spaces[PREV] = self._state_namespace(PREV, prev)
            prev_spaces[PREV] = self._state_namespace(PREV, prev)
        except FortranAbort as exc:
            return Verdict(check.name, level, elems, mask, None, aborted=str(exc), step=k), None
        ev = self.evaluator(arrays, self.context_functions(scenario, ex, level), prev=(prev, prev_spaces),
                            history=history)
        local = self.locals_for(elems, level)
        local.update(spaces)
        local[STEP_NAME] = np.int64(k)
        try:
            ok = _per_element(ev(node, local, len(elems)), elems.shape, f"{check.name!r}: {text!r}")
        except ExprError as exc:
            raise SpecError(f"{check.name!r}: {text!r}: {exc}") from None
        return Verdict(check.name, level, elems, mask, ok, step=k), (text, node, ev, local)

    def execute(self, scenario: Scenario | str, ex: Example, outputs: str | list[str] = "all") -> "Result":
        """Run an example and keep its inputs and the state after it (results.Result). outputs:
        "all" (every output and inout live here) or names."""
        from spel.proptest.results import Result

        scen = self.spec.scenarios[scenario] if isinstance(scenario, str) else scenario
        names = self.live_outputs() if outputs == "all" else [self.inv.canonical(tuple(n.split("%"))) or n
                                                              for n in outputs]
        if scen.steps > 1:
            run = self.run_steps(scen, ex)
            return Result(self, scen, ex, self.arrays(ex, run.states[-1]), steps=run.states, times=run.times)
        state = self.compute(scen, ex, list(names))
        return Result(self, scen, ex, self.arrays(ex, state))

    def built(self) -> "BuiltDomain":
        """The domain as the case built it at init, with `check()` against the declaration."""
        from spel.proptest.results import built

        if self._built is None:
            self._built = built(self)
        return self._built

    def half(self, arrays: dict[str, Any], which: int, permutation: Optional[list[int]] = None) -> Namespace:
        """`base%` (which = 0) or `follow%` (1): the state restricted to that half of the sets.
        With a permutation of the follow-up rows, `follow%x(p)` is the follow-up of base element p."""
        label = "base" if which == 0 else "follow"

        def get(path: list[str]) -> Any:
            name = self.canonical(tuple(path))
            if name is None or name not in arrays:
                raise ExprError(f"{label}%{'%'.join(path)}: not a state variable")
            fa = arrays[name]
            var = self.inv.vars[name]
            axis = var.subgrid_axis
            if axis is None:
                return fa
            half = self.half_count(var.level)
            inverse = None
            if permutation is not None and which == 1:
                inverse = np.argsort(self.layout.row_permutation(var.level, np.asarray(permutation)))
            sl = [slice(None)] * fa.data.ndim
            sl[axis] = slice(which * half, (which + 1) * half)
            data = fa.data[tuple(sl)]
            if inverse is not None:
                data = np.take(data, inverse, axis=axis)
            lb = list(fa.lb)
            lb[axis] = 1
            return FArray(data, tuple(lb), name)

        return Namespace(label, get)

    _half = half  # the former name

    def _inputs_namespace(self, ex: Example) -> Namespace:
        before = self._farrays(ex.inputs)

        def get(path: list[str]) -> Any:
            name = self.canonical(tuple(path))
            if name is None or name not in before:
                raise ExprError(f"{INPUT}%{'%'.join(path)}: not an input the scenario sets")
            return before[name]

        return Namespace(INPUT, get)

    def context_functions(self, scenario: Scenario, ex: Example, level: Optional[str] = None) -> dict[str, Callable]:
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
            coord = np.zeros(self.layout.count("g"))  # per gridcell: subspace steps move a gridcell as a whole
            for name, w in zip(res["inputs"], np.asarray(res["eigenvectors"])[:, k - 1]):
                if name not in ex.u:
                    raise ExprError(f"active_dir: {name} is not a ruled input of this scenario any more")
                var = self.inv.vars[name]
                u = np.asarray(ex.u[name], dtype=float)
                # a layered input moved as a whole (its mean u over the gridcell's elements); one
                # without a subgrid dimension is the same for every element
                u = u.mean() if var.subgrid_axis is None else self.gridcell_mean(var.level, u, var.subgrid_axis)
                coord += w * (2.0 * u - 1.0)
            return coord[self.layout.gridcell_of(level)[np.asarray(p) - 1]]

        return {"active_dir": active_dir}

    # -- reports
    def gridcell_mean(self, level: str, a: np.ndarray, axis: int) -> np.ndarray:
        """The mean of a per-element array over each gridcell's elements of `level` (and every
        other dimension): one value per gridcell."""
        t = np.moveaxis(np.asarray(a, dtype=float), axis, 0).reshape(self.layout.count(level), -1).mean(axis=1)
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
        return np.unique(np.concatenate([self.rows_of(level, int(k), other) for k in elems])) + 1

    def report(self, check: Check, text: str, node, ev: Evaluator, local: dict, elems: np.ndarray,
               bad: np.ndarray, ex: Example, mask: np.ndarray, level: Optional[str] = None) -> str:
        level = level or "p"
        k = int(elems[bad[0]])
        one = {v: (np.asarray(x)[bad[:1]] if isinstance(x, np.ndarray) else x) for v, x in local.items()}
        lines = [f"{check.name}: {text}", f"  fails at {bad.size} of {int(mask.sum())} elements; first: "
                 f"{self.describe_element(k, level)}"]
        shown = set()
        for u in uses(node):
            label = "%".join(u.path)
            if label in shown or u.path[0] in ELEMENT_VARS or (len(u.path) == 1 and u.path[0] in (*check.names, INPUT)):
                continue
            shown.add(label)
            try:
                val = ev(_use_node(node, u), one, 1)
            except ExprError:
                continue
            lines.append(f"    {label} = {_fmt(val)}")
        lines.append("  drawn inputs at that element:")
        for name in sorted(ex.inputs):
            var = self.inv.vars[name]
            axis = var.subgrid_axis
            arr = ex.inputs[name]
            if axis is None:
                val = arr
            else:
                rows = self.rows_of(level, k, var.level)
                val = np.take(arr, rows, axis=axis)
                if rows.size == 1:
                    val = np.take(val, 0, axis=axis)
            lines.append(f"    {name} = {_fmt(val)}")
        return "\n".join(lines)

    def describe_element(self, k: int, level: str = "p") -> str:
        return f"element {k} ({self.layout.describe(level, k)})"

    # -- saved failures
    def save_failure(self, failure: Optional[Failure] = None) -> Optional[Path]:
        """Write a failing example (default: the last one, which after a Hypothesis run is the
        shrunk one) to `<data>/failures/<scenario>__<check>/`: the domain's two files with the
        example's inputs written in (the format of a capture, readable by the case), and
        `example.json`. The follow-up of a relation under other namelist options goes to
        `follow/` the same way. `replay` runs it again."""
        f = failure or self.last_failure
        if f is None:
            return None
        out = self.data / FAILURES_DIR / slug(f"{f.scenario}__{f.check}")
        self._write_example(out, f.example)
        meta = {
            "spec": str(self.spec.path), "case": self.case_dir.name, "scenario": f.scenario, "check": f.check,
            "element": f.element, "message": f.message, "seed": f.example.seed,
            "modes": {k: strat.MODES[v] for k, v in sorted(f.example.modes.items())},
            "extra": f.example.extra, "inputs": sorted(f.example.inputs), "namelist": self.options,
        }
        if f.example.forcing:
            np.savez(out / FORCING_FILE, **{k.replace("%", "__"): v for k, v in f.example.forcing.items()})
            meta["forcing"] = sorted(f.example.forcing)
        if f.example.follow is not None:
            self.sibling(f.example.extra.get("namelist", {}))._write_example(out / "follow", f.example.follow)
            meta["follow_inputs"] = sorted(f.example.follow.inputs)
        (out / EXAMPLE_FILE).write_text(json.dumps(meta, indent=1, default=str))
        return out

    def capture_dir(self) -> Path:
        """A directory with the domain in capture format (both files): the domain's own, or for
        `init = "direct"`, which writes no inputs file, the same domain written once as files
        (saved failures must stay readable by the case and by `replay`)."""
        if (self.domain.directory / INPUTS_FILE).is_file():
            return self.domain.directory
        if self._capture_dir is None:
            spec = dataclasses.replace(self.domain_spec, init="files")
            self._capture_dir = build_domain(self.inv, spec, self.domain.directory.with_name(
                self.domain.directory.name + "-files")).directory
        return self._capture_dir

    def _write_example(self, out: Path, example: Example) -> None:
        import netCDF4

        out.mkdir(parents=True, exist_ok=True)
        for name in (INPUTS_FILE, CONSTANTS_FILE):
            shutil.copyfile(self.capture_dir() / name, out / name)
        with netCDF4.Dataset(out / INPUTS_FILE, "a") as ds:
            for name, arr in example.inputs.items():
                nc = self.inv.vars[name].nc_name
                if nc not in ds.variables:  # a dormant input the domain of this configuration does not hold
                    continue
                a = np.asarray(arr)
                ds.variables[nc][...] = (a.astype(np.int32) if a.dtype == bool else a).T

    def _read_example(self, directory: Path, names: list[str]) -> dict[str, np.ndarray]:
        import netCDF4

        out: dict[str, np.ndarray] = {}
        with netCDF4.Dataset(directory / INPUTS_FILE) as ds:
            ds.set_auto_mask(False)
            for name in names:
                var = self.inv.vars.get(name)
                if var is None or name not in self.domain.extents:
                    raise SpecError(f"{directory}: {name} is not a variable of {self.case_dir.name} any more")
                if var.nc_name not in ds.variables:
                    continue
                a = np.asarray(ds.variables[var.nc_name][...]).T
                if a.size != int(np.prod(self.shape(name), dtype=int)):
                    raise SpecError(f"{directory}: {name} has shape {a.shape}, the spec's domain gives {self.shape(name)}")
                a = a.reshape(self.shape(name))
                out[name] = a != 0 if var.type == "logical" else a
        return out

    def load_failure(self, directory: Path) -> tuple[dict, Example]:
        """A saved failure: its `example.json` and the example with the saved inputs."""
        directory = Path(directory)
        try:
            meta = json.loads((directory / EXAMPLE_FILE).read_text())
        except OSError as exc:
            raise SpecError(f"{directory / EXAMPLE_FILE}: {exc.strerror}") from None
        ex = Example(self._read_example(directory, meta["inputs"]), seed=int(meta.get("seed", 0)),
                     extra=dict(meta.get("extra", {})))
        if meta.get("forcing"):
            with np.load(directory / FORCING_FILE) as z:
                ex.forcing = {k: np.asarray(z[k.replace("%", "__")]) for k in meta["forcing"]}
        if meta.get("follow_inputs"):
            sibling = self.sibling(ex.extra.get("namelist", {}))
            ex.follow = Example(sibling._read_example(directory / "follow", meta["follow_inputs"]), seed=ex.seed)
        return meta, ex

    # -- looking at a scenario
    def probe(self, scenario: Scenario, n: int = 5, seed: int = 0) -> ProbeResult:
        """Run `n` examples (the first with every input at its rule's midpoint, the others drawn
        uniformly) and say what the routine did on the scenario's elements: which outputs it
        never wrote, which inout variables it never changed, which outputs never vary."""
        self.preflight(scenario)
        rng = np.random.default_rng(seed)
        elems = self.elements((scenario.level, scenario.filter))
        names = self.live_outputs()
        unwritten: dict[str, list[int]] = {}
        seen: dict[str, int] = {}  # elements judged per output (its own level's)
        changed: dict[str, bool] = {}
        lo: dict[str, float] = {}
        hi: dict[str, float] = {}
        aborted: list[str] = []
        passing = total = judged = 0
        for k in range(max(1, n)):
            ex = self.inputs_at(scenario, 0.5 if k == 0 else rng.random)
            mask = self.assume_mask(scenario, ex, elems)
            total += elems.size
            passing += int(mask.sum())
            rows = elems[mask]
            if not rows.size:
                continue
            try:
                state = self.backend.run(self.live_inputs(ex), names, trap=scenario.trap_fpe)
            except FortranAbort as exc:
                aborted.append(str(exc))
                continue
            judged += rows.size
            for name, fa in state.items():
                if fa is None:
                    continue
                var = self.inv.vars[name]
                axis = var.subgrid_axis
                own = self.elements_at(scenario.level, rows, var.level)  # the output's elements there
                data = np.moveaxis(np.asarray(fa.data), axis, 0)[own - fa.lb[axis]].reshape(own.size, -1)
                if var.kind == "output" and var.type == "real":
                    missing = np.isnan(data).all(axis=1)
                    unwritten.setdefault(name, []).append(int(missing.sum()))
                    seen[name] = seen.get(name, 0) + own.size
                    data = data[~missing]
                elif name in ex.inputs:
                    before = np.moveaxis(np.asarray(ex.inputs[name]), axis, 0)[own - 1].reshape(own.size, -1)
                    changed[name] = changed.get(name, False) or not np.array_equal(before, data, equal_nan=True)
                if var.kind == "output" and data.size:
                    vals = data[np.isfinite(data.astype(float))].astype(float)
                    if vals.size:
                        lo[name] = min(lo.get(name, np.inf), float(vals.min()))
                        hi[name] = max(hi.get(name, -np.inf), float(vals.max()))
        never = sorted(n_ for n_, c in unwritten.items() if seen.get(n_) and sum(c) == seen[n_])
        partly = {n_: sum(c) / seen[n_] for n_, c in unwritten.items() if seen.get(n_) and 0 < sum(c) < seen[n_]}
        return ProbeResult(
            scenario=scenario.name, examples=max(1, n), elements=int(elems.size),
            assume_pass=passing / total if total else 0.0, outputs=len(names), never_written=never,
            partly_written=partly, unchanged=sorted(n_ for n_, c in changed.items() if not c),
            constant={n_: lo[n_] for n_ in sorted(lo) if lo[n_] == hi[n_] and n_ not in never}, aborted=aborted,
        )

    # -- hypothesis
    def hypothesis_test(self, scenario: Scenario, check: Check, settings: Optional[dict] = None) -> Callable:
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
            database=DirectoryBasedExampleDatabase(str(self.data / "hypothesis" / f"{slug(label)[:60]}-{key}")),
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

    def preflight(self, scenario: Scenario, checks: Optional[list[Check]] = None) -> None:
        """Refuse a scenario that would run on a silently defaulted input. checks: the ones that
        will run (default: all of the scenario's)."""
        pinned = self.domain.pinned
        ruled = sorted(n for n in [*scenario.inputs, *(v for v, _ in scenario.derive)] if n in pinned)
        if ruled:
            raise SpecError(f"[scenario.{scenario.name}]: {', '.join(ruled)} are fixed by the domain (the declared "
                            "topounit elevations decide downhill_ti); remove their rules")
        missing = [n for n in missing_rules(scenario, self.inv, self.domain.constants) if n not in pinned]
        if missing:
            raise SpecError(f"[scenario.{scenario.name}]: inputs without a rule (add them, or `extends` the "
                            f"stats rule file): {', '.join(missing)}")
        filled = {n: why for n, why in self.domain.filled.items() if n not in scenario.inputs}
        if filled:
            raise SpecError(
                f"[scenario.{scenario.name}]: parameters the routines read have no value (give each a rule "
                "under `inputs`, or use a reference capture that has them): "
                + "; ".join(f"{n} ({why})" for n, why in sorted(filled.items())))
        for check in self.checks(scenario) if checks is None else checks:
            check.preflight(self, scenario)

    def accepts(self, scenario: Scenario, source: "Session", where: str) -> None:
        """Whether this session can run the examples `source` makes for the scenario (the other
        side of a differential check, the follow-up of a relation under other options): the
        inputs it reads exist with the same shapes, and none of its own is left without a value.
        where: how messages name this session."""
        names = sorted(n for n in set(scenario.inputs) | {v for v, _ in scenario.derive}
                       if n not in self.inv.vars or self.inv.vars[n].kind in INPUT_KINDS)
        absent = [n for n in names if n not in self.inv.vars or n not in self.domain.extents]
        if absent:
            raise SpecError(f"{where}: it has no input {', '.join(absent)} (the scenario sets them)")
        differ = [f"{n} {self.shape(n)} vs {source.shape(n)}" for n in names
                  if n in source.domain.extents and self.shape(n) != source.shape(n)]
        if differ:
            raise SpecError(f"{where}: other shapes than {source.case_dir.name}: {'; '.join(differ)}")
        try:
            self.preflight(scenario, [])
        except SpecError as exc:
            raise SpecError(f"{where}: {exc}") from None


def _per_element(value: Any, shape: tuple[int, ...], where: str) -> np.ndarray:
    """A logical per element (or one for all of them)."""
    v = np.asarray(value, dtype=bool)
    if v.ndim > 1 or (v.ndim == 1 and v.shape != shape):
        raise SpecError(f"{where}: one logical per element expected, got shape {v.shape} "
                        "(reduce the other dimensions, e.g. with all(...))")
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


# -- entry points


@dataclass
class Outcome:
    scenario: str
    check: str
    passed: bool
    message: str = ""
    kind: str = ""  # property | relation | differential
    seconds: float = 0.0
    failure: str = ""  # directory of the saved failing example


def run(spec: Spec, inv: Inventory, scenarios: Optional[list[str]] = None, checks: Optional[list[str]] = None,
        settings: Optional[dict] = None, fc: Optional[str] = None, rebuild: bool = False,
        out=sys.stdout, no_build: bool = False, acc: Optional[str] = None, session: Optional[Session] = None,
        ) -> list[Outcome]:
    """Every selected check of every selected scenario as a Hypothesis test; one Outcome each.
    session: use this one (it stays open) instead of making one from the spec."""
    outcomes = []
    root = session or Session(spec, inv, fc=fc, rebuild=rebuild, no_build=no_build, acc=acc)
    try:
        for name, scen in spec.scenarios.items():
            if scenarios and not _selected(name, scenarios):
                continue
            selected = [c for c in root.checks(scen) if not checks or _selected(c.name, checks)]
            if not selected:
                continue
            s = root.for_scenario(scen)  # the scenario's namelist options: its own session
            s.preflight(scen, selected)
            for c in selected:
                print(f"[{name}] {c.table} {c.name!r} ...", file=out, flush=True)
                s.last_failure = None
                start = time.monotonic()
                try:
                    s.hypothesis_test(scen, c, settings)()
                except PropertyFailure as exc:
                    saved = s.save_failure()
                    outcomes.append(Outcome(name, c.name, False, str(exc), c.table, time.monotonic() - start,
                                            str(saved or "")))
                    print(f"  FAIL\n{_indent(str(exc))}", file=out, flush=True)
                    if saved:
                        print(f"  saved: {saved}  (spel proptest replay {saved})", file=out, flush=True)
                    continue
                except SpecError as exc:
                    outcomes.append(Outcome(name, c.name, False, str(exc), c.table, time.monotonic() - start))
                    print(f"  FAIL\n{_indent(str(exc))}", file=out, flush=True)
                    continue
                except Exception as exc:  # Hypothesis' own errors (e.g. Unsatisfiable)
                    outcomes.append(Outcome(name, c.name, False, f"{type(exc).__name__}: {exc}", c.table,
                                            time.monotonic() - start))
                    print(f"  ERROR {type(exc).__name__}: {exc}", file=out, flush=True)
                    continue
                outcomes.append(Outcome(name, c.name, True, "", c.table, time.monotonic() - start))
                print("  ok", file=out, flush=True)
    finally:
        if session is None:
            root.close()
    return outcomes


def replay(directory: Path, fc: Optional[str] = None, acc: Optional[str] = None, no_build: bool = False,
           out=sys.stdout) -> Outcome:
    """Run a saved failing example again (no Hypothesis): the same report when it still fails."""
    from spel.proptest import spec as specmod

    directory = Path(directory)
    try:
        meta = json.loads((directory / EXAMPLE_FILE).read_text())
    except OSError as exc:
        raise SpecError(f"{directory / EXAMPLE_FILE}: {exc.strerror}") from None
    sp, inv = specmod.load(Path(meta["spec"]))
    scen = sp.scenarios.get(meta["scenario"])
    check = next((c for c in sp.checks if c.name == meta["check"]), None)
    if scen is None or check is None:
        raise SpecError(f"{directory}: {meta['spec']} has no scenario {meta['scenario']!r} with check {meta['check']!r} any more")
    start = time.monotonic()
    with Session(sp, inv, fc=fc, acc=acc, no_build=no_build) as root:
        s = root.for_scenario(scen)
        _, ex = s.load_failure(directory)
        try:
            s.judge(scen, check, ex, hypothesis=False)
        except PropertyFailure as exc:
            print(f"[{scen.name}] {check.table} {check.name!r}: FAIL\n{_indent(str(exc))}", file=out)
            return Outcome(scen.name, check.name, False, str(exc), check.table, time.monotonic() - start, str(directory))
    print(f"[{scen.name}] {check.table} {check.name!r}: ok (the saved example passes now)", file=out)
    return Outcome(scen.name, check.name, True, "", check.table, time.monotonic() - start, str(directory))


def write_json(outcomes: list[Outcome], spec: Spec, settings: dict, path: Path) -> None:
    data = {"spec": str(spec.path), "case": spec.case_dir.name, "settings": {**spec.settings, **settings},
            "passed": sum(o.passed for o in outcomes), "failed": sum(not o.passed for o in outcomes),
            "outcomes": [dataclasses.asdict(o) for o in outcomes]}
    Path(path).write_text(json.dumps(data, indent=1))


def write_junit(outcomes: list[Outcome], spec: Spec, path: Path) -> None:
    """A JUnit XML report (one testcase per check, classname = the scenario), for CI."""
    from xml.etree import ElementTree as ET

    suite = ET.Element("testsuite", name=f"spel proptest {spec.path.name}", tests=str(len(outcomes)),
                       failures=str(sum(not o.passed for o in outcomes)),
                       time=f"{sum(o.seconds for o in outcomes):.3f}")
    for o in outcomes:
        case = ET.SubElement(suite, "testcase", classname=f"{spec.case_dir.name}.{o.scenario}", name=o.check,
                             time=f"{o.seconds:.3f}")
        if not o.passed:
            ET.SubElement(case, "failure", message=o.message.splitlines()[0] if o.message else "failed").text = o.message
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def _selected(name: str, patterns: list[str]) -> bool:
    """An exact name or a glob (names may contain `[`, e.g. "btran in [0, 1]")."""
    import fnmatch

    return name in patterns or any(fnmatch.fnmatchcase(name, p) for p in patterns)


def _indent(text: str, pre: str = "    ") -> str:
    return "\n".join(pre + line for line in text.splitlines())
