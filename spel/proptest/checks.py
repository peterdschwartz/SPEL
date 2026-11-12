"""The kinds of checks a spec can hold, one TOML array of tables each (`[[property]]`, ...).

A check is judged per element after the routine ran: `judged` is a logical Fortran expression,
evaluated over the elements the check selects. A kind decides

- `from_table`: how its table is read (names resolved through `spec.Names`);
- `prepare`: what it does to the drawn example before the run (a relation fills the follow-up
  half of the sets);
- `elements` / `mask`: which elements are judged;
- `namespaces`: the extra names its expression may use (`base%x`, `follow%x`, `ref%x`, `dev%x`).

The runner (`runner.Session.judge`) knows only this interface, so a new kind is a class here
with `@register`; nothing else changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Optional

import numpy as np

from spel.proptest.errors import SpecError
from spel.proptest.fexpr import ExprError, FArray, Namespace, parse_expr, uses

Over = Optional[tuple[str, Optional[str]]]  # (element variable, filter member or None)
# In every check `input%x` is the value x had before the run (a bare `x` is what the routine left:
# for a variable it both reads and writes, the new value).
INPUT = "input"
PREV = "prev"  # in a multi-step run: x before the step being judged (spec.PREV)


@dataclass
class Expr:
    text: str
    node: Any


def live_names(expr: Expr, inv, spaces: dict[str, Any], where: str, scenario: str, before: bool = False) -> None:
    """Every variable the expression names has a value in the scenario's configuration.
    inv: the configured inventory for bare names; spaces: namespace -> its inventory (None: not
    checked). before: evaluated before the routine runs, so outputs are not available either."""
    for u in uses(expr.node):
        head = u.path[0]
        target, path = (spaces[head], u.path[1:]) if head in spaces else (inv, u.path)
        if target is None or not path:
            continue
        name = target.canonical(list(path))
        if name is None:
            continue
        kind = target.vars[name].kind
        label = "%".join(u.path) + (f" ({name})" if "%".join(path) != name else "")
        if kind == "unused":
            raise SpecError(f"{where}: {label} has no value in scenario {scenario}: {target.why_unused(name)}")
        if before and kind == "output":
            raise SpecError(f"{where}: {label} is an output: the routine computes it; only inputs, parameters "
                            "and constants are available here")


class Check:
    table: ClassVar[str] = ""  # the TOML table name, also what reports call the check
    names: ClassVar[tuple[str, ...]] = ()  # namespaces its expression may use
    keys: ClassVar[set[str]] = set()  # keys of its table besides name, scenarios and over

    name: str
    scenarios: list[str]
    over: Over
    steps: Any = "all"  # in a multi-step run: "all", "last", a step, or [first, last] (1-based)

    @classmethod
    def from_table(cls, t: dict, names) -> "Check":
        raise NotImplementedError

    def judged_steps(self, n: int) -> list[int]:
        """The steps (1-based) of an n-step run the check is judged after."""
        s = self.steps
        if s == "all":
            return list(range(1, n + 1))
        if s == "last":
            return [n]
        if isinstance(s, int):
            return [s] if 1 <= s <= n else []
        return list(range(max(1, s[0]), min(n, s[1]) + 1))

    @classmethod
    def _common(cls, t: dict, names) -> tuple[str, dict]:
        """Checks of the table's keys; (where, the name/scenarios/over arguments)."""
        where = f"[[{cls.table}]] {t.get('name', '?')!r}"
        extra = set(t) - cls.keys - {"name", "scenarios", "over", "steps"}
        if extra:
            raise SpecError(f"{where}: unknown keys {sorted(extra)}")
        if "name" not in t:
            raise SpecError(f"{where}: `name` is required")
        return where, {"name": t["name"], "scenarios": list(t.get("scenarios", ["*"])),
                       "over": names.over(t.get("over"), where)}

    @property
    def judged(self) -> Expr:
        raise NotImplementedError

    def exprs(self) -> list[Expr]:
        """The expressions evaluated after the run (their state variables are fetched)."""
        return [self.judged]

    def check_spec(self, spec) -> None:
        """What the check needs of the rest of the spec (raise SpecError)."""

    def check_scenario(self, scenario, inv, configured) -> None:
        """The names the check uses have a value in a scenario it runs in. inv: the inventory
        under the scenario's configuration; configured(extra): the one under further options."""
        where = f"[[{self.table}]] {self.name!r}"
        live_names(self.judged, inv, {INPUT: inv, PREV: inv, **{n: inv for n in self.names}}, where, scenario.name)

    def expr_for(self, session, scenario) -> Expr:
        """The expression to judge in a session (default: `judged`; a kind may build it from
        the session's configuration)."""
        return self.judged

    def preflight(self, session, scenario) -> None:
        """What must hold before the scenario's first example runs (raise SpecError)."""

    def prepare(self, session, scenario, ex):
        return ex

    def elements(self, session, ex, elems: np.ndarray, level: str = "p") -> np.ndarray:
        """The elements (Fortran indices of `level`) to judge."""
        return elems

    def mask(self, session, scenario, ex, elems: np.ndarray, level: str = "p") -> np.ndarray:
        return session.assume_mask(scenario, ex, elems)

    def namespaces(self, session, scenario, ex, arrays: dict[str, Any]) -> dict[str, Namespace]:
        return {}


KINDS: dict[str, type[Check]] = {}


def parse_steps(value: Any, where: str) -> Any:
    """A check's `steps`: "all", "last", a step number, or [first, last] (1-based, inclusive)."""
    if value in ("all", "last"):
        return value
    if isinstance(value, int) and not isinstance(value, bool) and value >= 1:
        return value
    if (isinstance(value, list) and len(value) == 2 and all(isinstance(v, int) and not isinstance(v, bool)
                                                             for v in value) and 1 <= value[0] <= value[1]):
        return (value[0], value[1])
    raise SpecError(f"{where}: steps = {value!r}: \"all\", \"last\", a step number or [first, last]")


def register(cls: type[Check]) -> type[Check]:
    if not cls.table:
        raise ValueError(f"{cls.__name__}: a check kind needs a table name")
    read = cls.from_table.__func__

    def from_table(klass, t: dict, names) -> Check:  # every kind takes `steps` (multi-step runs)
        obj = read(klass, t, names)
        if "steps" in t:
            obj.steps = parse_steps(t["steps"], f"[[{klass.table}]] {t.get('name', '?')!r}")
        return obj

    cls.from_table = classmethod(from_table)
    KINDS[cls.table] = cls
    return cls


@register
@dataclass
class Property(Check):
    """An invariant: `expr` holds at every element."""

    name: str
    scenarios: list[str]
    expr: Expr
    over: Over = None

    table: ClassVar[str] = "property"
    keys: ClassVar[set[str]] = {"expr"}

    @classmethod
    def from_table(cls, t, names):
        where, common = cls._common(t, names)
        if "expr" not in t:
            raise SpecError(f"{where}: `expr` is required")
        return cls(expr=names.expr(t["expr"], where, (INPUT, PREV)), **common)

    @property
    def judged(self) -> Expr:
        return self.expr


@register
@dataclass
class Relation(Check):
    """A metamorphic relation: a base run and a follow-up, compared by `expect` through `base%x`
    and `follow%x`.

    Without `namelist`, both run in one call: the follow-up half of the sets gets a copy of the
    base half, changed by `follow` (input -> an expression of the base inputs). Without `follow`
    the two halves are copies. `permute`: the follow-up gridcells get the base gridcells of the
    same declaration in another order, each with everything under it;
    `follow%x(p)` is still the follow-up of base element p.

    With `namelist = { option = value }` the follow-up is the same example (changed by `follow`,
    if given) under other namelist options. Options are module scalars, so the two cannot share a
    call: the follow-up runs in a second session of the same library (`Session.sibling`), and
    `base%`/`follow%` are the two whole runs."""

    name: str
    scenarios: list[str]
    follow: list[tuple[str, Expr]]
    expect: Expr
    over: Over = None
    permute: bool = False
    namelist: dict[str, Any] = field(default_factory=dict)

    table: ClassVar[str] = "relation"
    names: ClassVar[tuple[str, ...]] = ("base", "follow")
    keys: ClassVar[set[str]] = {"follow", "expect", "permute", "namelist"}

    @classmethod
    def from_table(cls, t, names):
        where, common = cls._common(t, names)
        follow = []
        for key, text in t.get("follow", {}).items():
            v = names.var(key, f"{where}.follow")
            if names.inv.vars[v].level is None:
                raise SpecError(f"{where}.follow: {v} has no subgrid dimension (base and follow-up share it)")
            follow.append((v, names.expr(str(text), f"{where}.follow")))
        if "expect" not in t:
            raise SpecError(f"{where}: `expect` is required")
        namelist = names.namelist(t.get("namelist", {}), f"{where}.namelist")
        if namelist and t.get("permute"):
            raise SpecError(f"{where}: `permute` reorders the halves of one run; it cannot be combined with `namelist`")
        return cls(follow=follow, expect=names.expr(t["expect"], where, (*cls.names, INPUT, PREV)),
                   permute=bool(t.get("permute", False)), namelist=namelist, **common)

    @property
    def judged(self) -> Expr:
        return self.expect

    def check_spec(self, spec) -> None:
        if not self.namelist and spec.domain.nsets % 2:
            raise SpecError("[domain]: relations split the sets into base and follow-up halves: nsets must be even")

    def check_scenario(self, scenario, inv, configured) -> None:
        where = f"[[{self.table}]] {self.name!r}"
        after = configured(self.namelist) if self.namelist else inv
        for v, e in self.follow:
            if after.vars[v].kind not in ("input", "inout", "parameter"):
                raise SpecError(f"{where}.follow: {v} is not an input of the follow-up in scenario {scenario.name}"
                                + (f": {after.why_unused(v)}" if after.vars[v].kind == "unused" else ""))
            live_names(e, inv, {}, f"{where}.follow", scenario.name, before=True)
        live_names(self.expect, inv, {INPUT: inv, PREV: inv, "base": inv, "follow": after}, where, scenario.name)

    def preflight(self, session, scenario) -> None:
        if self.namelist:
            session.sibling(self.namelist).accepts(scenario, session, f"[[relation]] {self.name!r} namelist {self.namelist}")

    def prepare(self, session, scenario, ex):
        if self.namelist:
            ex.follow = session.follow_example(scenario, self, ex, session.sibling(self.namelist))
            ex.extra["namelist"] = dict(self.namelist)
            return ex
        out = session.follow_up(scenario, self, ex)
        if self.permute:
            session.permute_follow(out)
        return out

    def _partner(self, session, ex, elems: np.ndarray, level: str) -> np.ndarray:
        """The follow-up element (Fortran index) of each base element."""
        return session.partner(elems, level, ex.extra.get("permutation"))

    def elements(self, session, ex, elems, level="p"):
        if self.namelist:
            return elems
        half = session.half_count(level)
        selected = set(elems.tolist())
        base = np.asarray([k for k in elems if k <= half], dtype=np.int64)
        partner = self._partner(session, ex, base, level)
        return base[[int(k) in selected for k in partner]] if base.size else base

    def mask(self, session, scenario, ex, elems, level="p"):
        if self.namelist:
            return (session.assume_mask(scenario, ex, elems)
                    & session.sibling(self.namelist).assume_mask(scenario, ex.follow, elems))
        return (session.assume_mask(scenario, ex, elems)
                & session.assume_mask(scenario, ex, self._partner(session, ex, elems, level)))

    def namespaces(self, session, scenario, ex, arrays):
        if self.namelist:
            sibling = session.sibling(self.namelist)
            names = sibling.needed([self.expect], (*self.names, INPUT))
            theirs = sibling.arrays(ex.follow, sibling.compute(scenario, ex.follow, names))
            return {"base": _whole("base", session, arrays), "follow": _whole("follow", sibling, theirs)}
        return {"base": session.half(arrays, 0), "follow": session.half(arrays, 1, ex.extra.get("permutation"))}


@register
@dataclass
class Differential(Check):
    """The same example on two builds: this case (`ref%x`) and the one `[compare]` names
    (`dev%x`). `outputs` lists what must agree ("all" = every output both cases have), within
    `rtol`/`atol` (default: identical); `expect` replaces that with any expression of `ref%`
    and `dev%`."""

    name: str
    scenarios: list[str]
    expect: Expr
    outputs: list[str] = field(default_factory=list)
    over: Over = None
    all_outputs: bool = False  # outputs = "all": the list is made per configuration (`expr_for`)
    tolerance: str = ""  # ", rtol=..., atol=..." of the generated comparison

    table: ClassVar[str] = "differential"
    names: ClassVar[tuple[str, ...]] = ("ref", "dev")
    keys: ClassVar[set[str]] = {"outputs", "rtol", "atol", "expect"}

    @classmethod
    def from_table(cls, t, names):
        where, common = cls._common(t, names)
        outputs: list[str] = []
        tol = ""
        if "expect" in t:
            if {"outputs", "rtol", "atol"} & set(t):
                raise SpecError(f"{where}: give `expect`, or `outputs` (with rtol/atol), not both")
            text = str(t["expect"])
        else:
            listed = t.get("outputs", "all")
            tol = "".join(f", {k}={float(t[k])!r}" for k in ("rtol", "atol") if k in t)
            if listed == "all":
                # which outputs exist depends on the namelist options of the run: `expr_for` makes the list
                return cls(expect=Expr(f"agree(dev%<every output>, ref%<every output>{tol})", parse_expr(".true.")),
                           all_outputs=True, tolerance=tol, **common)
            if not (isinstance(listed, list) and listed):
                raise SpecError(f'{where}: outputs is "all" or a list of variables')
            outputs = [names.var(k, f"{where}.outputs") for k in listed]
            text = " .and. ".join(_agree(names.inv.vars[v], tol) for v in outputs)
        return cls(expect=names.expr(text, where, (*cls.names, INPUT, PREV)), outputs=outputs, tolerance=tol, **common)

    @property
    def judged(self) -> Expr:
        return self.expect

    def check_scenario(self, scenario, inv, configured) -> None:
        if not self.all_outputs:  # the other side is another case: its names are checked when it runs
            live_names(self.expect, inv, {INPUT: inv, PREV: inv, "ref": inv, "dev": None},
                       f"[[{self.table}]] {self.name!r}", scenario.name)

    def expr_for(self, session, scenario) -> Expr:
        """`outputs = "all"`: every output that is live under the session's configuration, in
        both cases."""
        if not self.all_outputs:
            return self.expect
        key = (self.table, self.name)
        if key not in session.exprs:
            theirs = session.other.inv.vars
            names = [n for n in session.live_outputs()
                     if n in theirs and theirs[n].kind in ("output", "inout")]
            if not names:
                raise SpecError(f"[[differential]] {self.name!r}: no output to compare in scenario {scenario.name}")
            text = " .and. ".join(_agree(session.inv.vars[n], self.tolerance) for n in names)
            session.exprs[key] = session.parse(text, f"[[differential]] {self.name!r}", (*self.names, INPUT))
        return session.exprs[key]

    def check_spec(self, spec) -> None:
        if spec.compare is None:
            raise SpecError(f"[[differential]] {self.name!r}: the spec has no [compare] table (the other case or build)")

    def preflight(self, session, scenario) -> None:
        other = session.other
        other.accepts(scenario, session, f"[compare] {other.case_dir.name} ({other.variant.describe()})")

    def namespaces(self, session, scenario, ex, arrays):
        other = session.other
        names = [n for n in session.needed([self.expr_for(session, scenario)], (*self.names, INPUT))
                 if n in other.inv.vars]
        theirs = dict(other.arrays(ex, other.compute(scenario, ex, names)))
        return {"ref": _whole("ref", session, arrays), "dev": _whole("dev", other, theirs)}


def _agree(var, tol: str) -> str:
    """`agree(dev%x(p), ref%x(p))` for a variable, the other dimensions reduced with all()."""
    subs = ", ".join(d.level if d.level else ":" for d in var.dims)
    text = f"agree(dev%{var.name}({subs}), ref%{var.name}({subs}){tol})"
    return f"all({text})" if var.rank > 1 else text


def _whole(label: str, session, arrays: dict[str, Any]) -> Namespace:
    def get(path: list[str]) -> Any:
        name = session.canonical(tuple(path))
        if name is None or name not in arrays:
            raise ExprError(f"{label}%{'%'.join(path)}: not a state variable of {session.case_dir.name}")
        return arrays[name]

    return Namespace(label, get)
