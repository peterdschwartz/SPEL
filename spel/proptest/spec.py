"""The `<name>.proptest.toml` file: case, domain, scenarios (input rules), properties, relations.

Every name in the file (rule keys, `derive`/`assume`/`expr`/`follow`/`expect` expressions) is
parsed with SPEL's Fortran parser and resolved against the case's inventory at load time:
`inst%field`, an associate name of the routine (`forc_t`), or the flat netCDF name; module
parameters and constants by name; element variables (g, t, l, c, p) and, in relations, `base%`/
`follow%`. An unknown name, an input without a rule, or a rule for a non-input stops loading.

Rule kinds are in rules.py, check kinds in checks.py, functions in functions.py; each has a
registry, and this module only reads the tables.
"""

from __future__ import annotations

import fnmatch
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from spel.proptest import checks as chk
from spel.proptest import rules
from spel.proptest.checks import Check, Differential, Expr, Property, Relation  # noqa: F401 (the spec's names)
from spel.proptest.domain import CONSTANTS_FILE, DomainSpec, read_constants
from spel.proptest.errors import SpecError
from spel.proptest.fexpr import ELEMENTAL, REDUCTIONS, ExprError, designator, parse_expr, uses
from spel.proptest.inventory import LEVELS, Inventory
from spel.proptest.rules import DISTS, Rule  # noqa: F401

ELEMENT_VARS = tuple(LEVELS)  # g t l c p: the element variables of expressions (runner.Session.locals_for)
RULE_KINDS = tuple(rules.KINDS)
_rule = rules.parse  # the former name
BUILD_KEYS = {"fc", "acc", "dbg"}


@dataclass
class Scenario:
    name: str
    level: str  # element level of `over`
    filter: Optional[str]  # clumpfilter member, None = every element
    inputs: dict[str, Rule]
    derive: list[tuple[str, Expr]]
    assume: list[Expr]
    trap_fpe: bool = False
    namelist: dict[str, Any] = field(default_factory=dict)  # options of this scenario, over [namelist]


@dataclass
class Compare:
    """`[compare]`: the other case (or another build of this one) of differential checks."""

    case_dir: Path
    build: dict[str, Any] = field(default_factory=dict)


@dataclass
class Spec:
    path: Path
    case_dir: Path
    settings: dict[str, Any]
    domain: DomainSpec
    units: dict[str, str]
    scenarios: dict[str, Scenario]
    checks: list[Check]
    functions: dict[str, Callable] = field(default_factory=dict)  # built-in and the spec's own
    build: dict[str, Any] = field(default_factory=dict)  # [case] build = { fc, acc, dbg }
    compare: Optional[Compare] = None
    namelist: dict[str, Any] = field(default_factory=dict)  # [namelist]: the base configuration

    @property
    def properties(self) -> list[Property]:
        return [c for c in self.checks if isinstance(c, Property)]

    @property
    def relations(self) -> list[Relation]:
        return [c for c in self.checks if isinstance(c, Relation)]

    def scenarios_of(self, patterns: list[str]) -> list[Scenario]:
        return [s for n, s in self.scenarios.items() if any(fnmatch.fnmatchcase(n, p) for p in patterns)]


# -- loading


def load(path: Path, inv: Optional[Inventory] = None, units: bool = True) -> tuple[Spec, Inventory]:
    path = Path(path).resolve()
    try:
        data = tomllib.loads(path.read_text())
    except OSError as exc:
        raise SpecError(f"{path}: {exc.strerror}") from None
    except tomllib.TOMLDecodeError as exc:
        raise SpecError(f"{path}: {exc}") from None
    case = data.get("case", {})
    if "name" not in case and "dir" not in case:
        raise SpecError("[case]: `name` (unit-tests/<name>) or `dir` is required")
    case_dir = _case_dir(case, path)
    if inv is None:
        inv = Inventory.load(case_dir, units=units)
    return parse(data, path, case_dir, inv), inv


def _case_dir(case: dict, path: Path) -> Path:
    if "dir" in case:
        d = Path(case["dir"])
        return d if d.is_absolute() else (path.parent / d).resolve()
    from spel.scripts.config import unittests_dir

    return Path(unittests_dir) / case["name"]


def _build(table: Any, where: str) -> dict[str, Any]:
    if not isinstance(table, dict) or set(table) - BUILD_KEYS:
        raise SpecError(f"{where}: build = {{ fc = \"...\", acc = \"OFF|MULTICORE|GPU|OPENMP\", dbg = false }}")
    return dict(table)


def parse(data: dict, path: Path, case_dir: Path, inv: Inventory) -> Spec:
    from spel.proptest import functions as fn

    known = {"case", "settings", "domain", "units", "namelist", "scenario", "functions", "compare", *chk.KINDS}
    extra = set(data) - known
    if extra:
        raise SpecError(f"unknown tables {sorted(extra)}")
    case = data.get("case", {})
    extra = set(case) - {"name", "dir", "build"}
    if extra:
        raise SpecError(f"[case]: unknown keys {sorted(extra)}")
    input_data = case_dir.parent / "input-data"
    domain = DomainSpec.from_table(data.get("domain", {}), path.parent, input_data)
    domain.check(inv)
    functions = fn.load(data.get("functions", {}), path.parent)
    resolver = Names(inv, set(functions), domain)
    units = {resolver.var(k, "[units]"): str(v) for k, v in data.get("units", {}).items()}
    for name, u in units.items():  # the spec's units win over the registrations'
        inv.vars[name].units = u

    scenarios = {}
    for name, table in data.get("scenario", {}).items():
        scenarios[name] = _scenario(name, table, path.parent, inv, resolver)
    if not scenarios:
        raise SpecError("no [scenario.<name>] table")
    checks = [cls.from_table(t, resolver) for key, cls in chk.KINDS.items() for t in data.get(key, [])]
    spec = Spec(path=path, case_dir=case_dir, settings=dict(data.get("settings", {})), domain=domain,
                units=units, scenarios=scenarios, checks=checks, functions=functions,
                build=_build(case.get("build", {}), "[case]"), compare=_compare(data.get("compare"), path, case_dir),
                namelist=namelist_table(data.get("namelist", {}), inv, "[namelist]"))
    for item in checks:
        item.check_spec(spec)
        for pat in item.scenarios:
            if not any(fnmatch.fnmatchcase(n, pat) for n in scenarios):
                raise SpecError(f"{item.name!r}: no scenario matches {pat!r}")
    _check_configurations(spec, inv)
    return spec


def namelist_table(table: Any, inv: Inventory, where: str) -> dict[str, Any]:
    """A `namelist` table: option -> value. An option is a namelist variable the routines test,
    or another scalar switch they read (`soil_stress_method`, which ELM sets in code); both are
    constants of the case, so the constants file of a run carries the value."""
    if not isinstance(table, dict):
        raise SpecError(f"{where}: a table of options, e.g. {{ use_c13 = true }}")
    options = inv.options()
    out = {}
    for key, value in table.items():
        name = key.lower()
        c = inv.constants.get(name)
        if name not in options and not (c is not None and c.rank == 0 and c.status):
            known = ", ".join(sorted(options)) or "none"
            raise SpecError(f"{where}: {key} is not an option of the case (namelist options its routines test: "
                            f"{known}; other scalar constants they read can be set too)")
        if isinstance(value, (dict, list, str)):
            raise SpecError(f"{where}: {key} takes a logical or a number")
        out[name] = value
    return out


INPUT_KINDS = ("input", "inout", "parameter")


def _check_configurations(spec: Spec, inv: Inventory) -> None:
    """What depends on the namelist options: every name an item uses has a value in the
    configuration of each scenario the item runs in. A variable only used under options that are
    off there is dead: naming it is an error (nothing is checked against a value that is not
    there). A *rule* for it is fine (dormant) as long as it is an input somewhere."""
    for sname, scen in spec.scenarios.items():
        cfg = inv.configured(constants(spec, inv, scen))
        where = f"[scenario.{sname}]"
        for v in scen.inputs:
            if inv.vars[v].kind not in INPUT_KINDS and cfg.vars[v].kind not in INPUT_KINDS:
                raise SpecError(f"{where}.inputs: {v} is a {inv.vars[v].kind}, not an input or parameter")
        for v, e in scen.derive:
            if inv.vars[v].kind not in INPUT_KINDS and cfg.vars[v].kind not in INPUT_KINDS:
                raise SpecError(f"{where}.derive: {v} is not an input")
            if cfg.vars[v].kind in INPUT_KINDS:  # a dormant target is not derived here
                chk.live_names(e, cfg, {}, f"{where}.derive {v}", sname, before=True)
        for e in scen.assume:
            chk.live_names(e, cfg, {}, f"{where}.assume", sname, before=True)
    for check in spec.checks:
        for scen in spec.scenarios_of(check.scenarios):
            cfg = inv.configured(constants(spec, inv, scen))
            check.check_scenario(scen, cfg, lambda extra, scen=scen: inv.configured(constants(spec, inv, scen, extra)))


def _compare(table: Optional[dict], path: Path, case_dir: Path) -> Optional[Compare]:
    if table is None:
        return None
    extra = set(table) - {"case", "dir", "build"}
    if extra or not table:
        raise SpecError("[compare]: give `case` (unit-tests/<case>) or `dir`, and/or `build` (of this case)")
    other = case_dir
    if "dir" in table:
        other = _case_dir({"dir": table["dir"]}, path)
    elif "case" in table:
        other = _case_dir({"name": table["case"]}, path)
    return Compare(case_dir=other, build=_build(table.get("build", {}), "[compare]"))


def constants(spec: Spec, inv: Inventory, scenario: Optional[Scenario] = None,
              extra: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """The constants a run would see, without building the domain: the reference's, the spec's
    [domain.constants] and [namelist], the scenario's `namelist`, then `extra` (a relation's
    follow-up options); 0 for scalars nobody gave (as build_domain writes them). This is also
    the namelist configuration of the run (`Inventory.configured`)."""
    out = _constants(spec.domain, inv)
    out.update(spec.namelist)
    if scenario is not None:
        out.update(scenario.namelist)
    out.update(extra or {})
    return out


def options(spec: Spec, scenario: Optional[Scenario] = None, extra: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """The options the spec itself sets for a run: [namelist], the scenario's, then `extra`."""
    return {**spec.namelist, **(scenario.namelist if scenario is not None else {}), **(extra or {})}


def _constants(domain: DomainSpec, inv: Inventory) -> dict[str, Any]:
    ref = domain.reference
    consts = read_constants(ref / CONSTANTS_FILE) if ref is not None else {}
    consts.update({k.lower(): v for k, v in domain.constants.items()})
    for c in inv.constants.values():
        if c.rank == 0:
            consts.setdefault(c.name, 0)
    return consts


class Names:
    """Resolution of names in the spec against the inventory."""

    def __init__(self, inv: Inventory, functions: set[str], domain: Optional[DomainSpec] = None):
        self.inv = inv
        self.functions = functions
        self.domain = domain

    def var(self, key: str, where: str) -> str:
        try:
            path = designator(parse_expr(key))
        except ExprError as exc:
            raise SpecError(f"{where}: {key!r}: {exc}") from None
        name = self.inv.canonical(path) if path else None
        if name is None:
            raise SpecError(f"{where}: {key!r} is not a variable of the case")
        return name

    def expr(self, text: str, where: str, namespaces: tuple[str, ...] = ()) -> Expr:
        """A parsed expression whose names are all known: element variables, variables of the
        case, parameters, constants, filters, functions, and `<namespace>%<variable>`."""
        try:
            node = parse_expr(text)
        except ExprError as exc:
            raise SpecError(f"{where}: {text!r}: {exc}") from None
        for u in uses(node):
            head = u.path[0]
            if head in namespaces:
                if len(u.path) > 1 and self.inv.canonical(list(u.path[1:])) is None:
                    raise SpecError(f"{where}: {'%'.join(u.path)!r}: {'%'.join(u.path[1:])!r} is not a variable "
                                    f"of the case, in {text!r}")
                continue
            if len(u.path) == 1 and head in ELEMENT_VARS:
                continue
            if u.applied and len(u.path) == 1 and (head in self.functions or head in ELEMENTAL or head in REDUCTIONS):
                continue
            if self.known(list(u.path)):
                continue
            raise SpecError(f"{where}: unknown name {'%'.join(u.path)!r} in {text!r}")
        return Expr(text, node)

    def namelist(self, table: Any, where: str) -> dict[str, Any]:
        return namelist_table(table, self.inv, where)

    def known(self, path: list[str]) -> bool:
        if self.inv.canonical(path) is not None:
            return True
        if len(path) == 1:
            name = path[0]
            if self.inv.parameters.get(name) is not None or name in self.inv.constants:
                return True
            if name in (self.inv.time_names() or ()):
                return True
        return len(path) == 2 and path[0] == "filter" and (path[1] in self.inv.filters)

    def over(self, text: Optional[str], where: str) -> Optional[tuple[str, Optional[str]]]:
        return _over(text, self.inv, where)

    def comparable_outputs(self) -> list[str]:
        """Every variable the routines compute for the elements, with its namelist guard on
        under the spec's constants (`outputs = "all"` of a differential check)."""
        consts = _constants(self.domain, self.inv) if self.domain is not None else {}
        out = []
        for v in self.inv.of_kind("output", "inout"):
            if v.level is None or v.driver:
                continue
            try:
                if self.inv.guard_active(v, consts):
                    out.append(v.name)
            except ExprError:
                continue
        return sorted(out)


def _over(text: Optional[str], inv: Inventory, where: str) -> Optional[tuple[str, Optional[str]]]:
    """`p` (every patch) or `p in <filter member>`."""
    if text is None:
        return None
    words = str(text).split()
    if len(words) == 1 and words[0] in ELEMENT_VARS:
        return words[0], None
    if len(words) == 3 and words[1] == "in" and words[0] in ELEMENT_VARS:
        member = words[2].lower()
        if member not in inv.filters or inv.filters[member] != 1:
            raise SpecError(f"{where}: {member!r} is not a filter (clumpfilter member)")
        return words[0], member
    raise SpecError(f"{where}: over = {text!r}: expected `p` or `p in <filter>`")


def _scenario(name: str, table: dict, base: Path, inv: Inventory, names: Names) -> Scenario:
    where = f"[scenario.{name}]"
    known = {"over", "extends", "trap_fpe", "inputs", "derive", "assume", "namelist"}
    extra = set(table) - known
    if extra:
        raise SpecError(f"{where}: unknown keys {sorted(extra)}")
    raw: dict[str, Any] = {}
    ext = table.get("extends", [])
    for f in [ext] if isinstance(ext, str) else ext:
        p = Path(f) if Path(f).is_absolute() else base / f
        try:
            raw.update(tomllib.loads(p.read_text()).get("inputs", {}))
        except OSError as exc:
            raise SpecError(f"{where}.extends: {p}: {exc.strerror}") from None
        except tomllib.TOMLDecodeError as exc:
            raise SpecError(f"{where}.extends: {p}: {exc}") from None
    raw.update(table.get("inputs", {}))
    inputs: dict[str, Rule] = {}
    for key, value in raw.items():
        v = names.var(key, f"{where}.inputs")
        info = inv.vars[v]
        rule = rules.parse(v, value, f"{where}.inputs.{key}")
        if info.kind == "parameter" and not rules.KINDS[rule.kind].parameters:
            ok = [k for k, kind in rules.KINDS.items() if kind.parameters]
            raise SpecError(f"{where}.inputs: parameter {v} takes one of {', '.join(ok)}")
        if rule.by_pft and info.level != "p":
            raise SpecError(f"{where}.inputs.{key}: `pft` overrides apply to patch variables only (a column or "
                            f"landunit holds several pfts); {v} is not one")
        for sec, _ in rule.by_level:
            _section(sec, names, f"{where}.inputs.{key}.levels")
            if info.rank - (1 if info.level else 0) < 1:
                raise SpecError(f"{where}.inputs.{key}: `levels` needs a dimension besides the subgrid one")
        inputs[v] = rule

    over = _over(table.get("over", "p"), inv, f"{where}.over")
    derive = []
    for key, text in table.get("derive", {}).items():
        v = names.var(key, f"{where}.derive")
        inputs.pop(v, None)
        derive.append((v, names.expr(str(text), f"{where}.derive.{key}")))
    derive = _order_derive(derive, inv, f"{where}.derive")
    listed = table.get("assume", [])
    if isinstance(listed, dict):  # the first form: [scenario.x.assume] exprs = [...]
        if set(listed) - {"exprs"}:
            raise SpecError(f"{where}.assume: a list of expressions")
        listed = listed.get("exprs", [])
    if isinstance(listed, str):
        listed = [listed]
    assume = [names.expr(t, f"{where}.assume") for t in listed]
    return Scenario(name=name, level=over[0], filter=over[1], inputs=inputs, derive=derive, assume=assume,
                    trap_fpe=bool(table.get("trap_fpe", False)),
                    namelist=names.namelist(table.get("namelist", {}), f"{where}.namelist"))


def _section(text: str, names: Names, where: str) -> None:
    """A `levels` key is a Fortran subscript of the level dimension: `3`, `1:3`, `:nlevsoi`."""
    from spel.proptest.fexpr import parse_section

    try:
        bounds = parse_section(text)
    except ExprError as exc:
        raise SpecError(f"{where}: {text!r}: {exc}") from None
    for node in bounds:
        if node is None:
            continue
        for u in uses(node):
            if not names.known(list(u.path)):
                raise SpecError(f"{where}: unknown name {'%'.join(u.path)!r} in {text!r}")


def _order_derive(derive: list[tuple[str, Expr]], inv: Inventory, where: str) -> list[tuple[str, Expr]]:
    """Derive rules in dependency order (a rule after the rules deriving what it reads)."""
    targets = {v for v, _ in derive}
    deps = {v: {c for u in uses(e.node) if (c := inv.canonical(list(u.path))) in targets and c != v}
            for v, e in derive}
    exprs = dict(derive)
    out: list[tuple[str, Expr]] = []
    done: set[str] = set()
    while len(out) < len(derive):
        ready = [v for v, _ in derive if v not in done and deps[v] <= done]
        if not ready:
            cycle = sorted(v for v in targets if v not in done)
            raise SpecError(f"{where}: circular derive rules among {cycle}")
        for v in ready:
            out.append((v, exprs[v]))
            done.add(v)
    return out


def dormant_rules(scenario: Scenario, inv: Inventory) -> list[str]:
    """Rules and derive targets of the scenario for variables that are not inputs under `inv`'s
    configuration (they are inputs under other namelist options, and are not set here)."""
    names = [*scenario.inputs, *(v for v, _ in scenario.derive)]
    return sorted(n for n in names if inv.vars[n].kind not in INPUT_KINDS)


def missing_rules(scenario: Scenario, inv: Inventory, constants: dict[str, Any]) -> list[str]:
    """Inputs read by the routines (guards on) that the scenario neither rules nor derives.
    inv: the inventory under the scenario's configuration (`Inventory.configured`), so an input
    only read under options that are off needs no rule."""
    have = set(scenario.inputs) | {v for v, _ in scenario.derive}
    out = []
    for v in inv.of_kind("input", "inout"):
        if v.name in have or v.instance == "bounds" or not inv.guard_active(v, constants):
            continue
        out.append(v.name)
    return sorted(out)
