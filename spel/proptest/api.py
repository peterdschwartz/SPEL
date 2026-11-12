"""Build proptest specs in Python, declaratively, and write them as TOML.

    from spel.proptest import api as pt
    from spel.proptest.api import Gridcell, Lake, Patch, Soil, Topounit, Urban

    case = pt.Case("urbval")
    dom = pt.Domain([Gridcell.of(Urban("md"), Urban("hd"), copies=2, name="town")], reference="cf_val")
    spec = pt.Spec(case, dom, namelist={"implicit_stress": False})
    base = spec.scenario("base", over="p in urbanp", extends=case.stats_rules(landunits=["urban"]),
                         inputs={"lun_pp%canyon_hwr": pt.Range(0.5, 1.0)}, derive=pt.URBAN_GEOMETRY)
    spec.property("canopy air is taf", "veg_es%t_ref2m(p) == lun_es%taf(l)", scenarios=[base])
    spec.to_toml("urbval.proptest.toml")
    with spec.session() as s:
        res = s.execute("base", s.sample("base", seed=1))
        res.frame("c")

One source of truth: the builder produces the dict `spec.parse` reads (`to_dict`), every piece
is checked when it is added by the same parser code a TOML file goes through (`spec.Names`,
`rules.parse`, `checks.KINDS[...]`, `DomainSpec.from_table`), and `validate()`/`session()` use
`spec.parse` on the whole. So a spec built here, its TOML file and `spel proptest` agree.
Expressions are Fortran text (the spec language); variables may be named by anything the
inventory knows (an alias, `inst%field`) and are stored by their canonical name.
"""

from __future__ import annotations

import difflib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Iterable, Optional, Sequence

import numpy as np

from spel.proptest import checks as chk
from spel.proptest import rules as rk
from spel.proptest import spec as sp
from spel.proptest import tomlw
from spel.proptest.domain import DomainSpec, expected_filters
from spel.proptest.errors import SpecError
from spel.proptest.hierarchy import (Column, Crop, Gridcell, Lake, LandunitKind, Layout, Patch, Soil,  # noqa: F401
                                     Topounit, Urban, UrbanDensity)
from spel.proptest.inventory import Inventory, VarInfo
from spel.proptest.scaffold import URBAN_DERIVE

URBAN_GEOMETRY: dict[str, str] = dict(URBAN_DERIVE)  # the canyon geometry as ELM derives it


# -- the case


@dataclass(frozen=True)
class Var:
    """A variable of the case, as the inventory classifies it."""

    name: str
    kind: str  # topology | parameter | input | inout | output | unused
    type: str
    level: Optional[str]
    dims: tuple[str, ...]
    units: str
    long_name: str
    guard: Optional[str]

    @classmethod
    def of(cls, v: VarInfo) -> "Var":
        return cls(v.name, v.kind, v.type, v.level, tuple(d.text for d in v.dims), v.units, v.long_name, v.guard)

    def __str__(self) -> str:
        return self.name


class Case:
    """A unit-test case: its inventory (from `fut.pkl`) and its reference capture."""

    def __init__(self, case: str | Path, reference: Optional[str | Path] = None, units: bool = True):
        from spel.proptest import scaffold

        p = Path(case)
        if p.is_dir():
            self.dir = p.resolve()
        else:
            from spel.scripts.config import unittests_dir

            self.dir = Path(unittests_dir) / str(case)
        self.name = self.dir.name
        self.inv = Inventory.load(self.dir, units=units)
        self.reference: Optional[Path] = scaffold.reference_dir(self.dir, str(reference) if reference else None)

    @classmethod
    def of(cls, inv: Inventory, reference: Optional[Path] = None) -> "Case":
        """A Case around an inventory already loaded (tests, `Spec.from_toml`)."""
        case = cls.__new__(cls)
        case.dir, case.name, case.inv, case.reference = Path(inv.case_dir), Path(inv.case_dir).name, inv, reference
        return case

    @property
    def input_data(self) -> Path:
        return self.dir.parent / "input-data"

    def canonical(self, name: str) -> str:
        """The inventory's name of a variable (`ht_roof` -> `lun_pp%ht_roof`); SpecError if none."""
        return sp.Names(self.inv, set()).var(name, f"{self.name}")

    def var(self, name: str) -> Var:
        return Var.of(self.inv.vars[self.canonical(name)])

    def vars(self, kind: Optional[str] = None, level: Optional[str] = None) -> list[Var]:
        return [Var.of(v) for v in self.inv.vars.values()
                if (kind is None or v.kind == kind) and (level is None or v.level == level)]

    def find(self, text: str, n: int = 10) -> list[Var]:
        """Variables whose name looks like `text` (substring first, then close matches)."""
        text = text.lower()
        names = list(self.inv.vars)
        hits = [k for k in names if text in k]
        hits += [k for k in difflib.get_close_matches(text, names, n=n, cutoff=0.4) if k not in hits]
        return [Var.of(self.inv.vars[k]) for k in hits[:n]]

    def stats_rules(self, landunits: Optional[Iterable[str | LandunitKind]] = None, where: Optional[str] = None,
                    stride: int = 1, namelist: Optional[dict[str, Any]] = None) -> Path:
        """Run `spel proptest stats` on the reference (only patches on these landunit kinds and
        `where`); the rule file's path, for a scenario's `extends`."""
        from spel.proptest import stats
        from spel.proptest.domain import CONSTANTS_FILE, read_constants

        if self.reference is None:
            raise SpecError(f"{self.name}: no reference capture (Case(..., reference=...))")
        consts = read_constants(self.reference / CONSTANTS_FILE)
        consts.update({k.lower(): v for k, v in (namelist or {}).items()})
        live = self.inv.configured(consts)
        kinds = {LandunitKind(k) if isinstance(k, str) else k for k in landunits} if landunits else None
        res = stats.compute(live, self.reference, where=where, stride=stride, constants=consts, landunits=kinds)
        return Path(stats.write(live, res, self.dir)[0])


# -- input rules


@dataclass(frozen=True)
class Rule:
    """An input rule; the subclasses are the rule kinds (rules.KINDS). Options common to all:
    `clip`, per-pft overrides (patch variables), per-level overrides (sections of the first
    non-subgrid dimension), `units`, `note`."""

    clip: Optional[tuple[float, float]] = field(default=None, kw_only=True)
    pft: dict[int, "Rule"] = field(default_factory=dict, kw_only=True)
    levels: dict[str, "Rule"] = field(default_factory=dict, kw_only=True)
    units: str = field(default="", kw_only=True)
    note: str = field(default="", kw_only=True)

    kind: ClassVar[str] = ""

    def args(self) -> dict[str, Any]:
        raise NotImplementedError

    def to_table(self) -> Any:
        out: dict[str, Any] = self.args()
        if self.clip is not None:
            out["clip"] = list(self.clip)
        if self.pft:
            out["pft"] = {str(p): r.to_table() for p, r in self.pft.items()}
        if self.levels:
            out["levels"] = {k: r.to_table() for k, r in self.levels.items()}
        if self.units:
            out["units"] = self.units
        if self.note:
            out["note"] = self.note
        if isinstance(self, Value) and set(out) == {"value"}:
            return out["value"]  # a bare value, as the TOML writes a constant
        return out

    @staticmethod
    def from_table(value: Any) -> "Rule":
        r = rk.parse("x", value, "rule")
        return _rule_of(r)


def _rule_of(r: rk.Rule) -> Rule:
    cls = RULES[r.kind]
    common = dict(clip=tuple(r.clip) if r.clip else None, pft={p: _rule_of(x) for p, x in r.by_pft.items()},
                  levels={k: _rule_of(x) for k, x in r.by_level})
    return cls.from_args(r.args, **common)


@dataclass(frozen=True)
class Range(Rule):
    lo: float
    hi: float
    kind: ClassVar[str] = "range"

    def args(self):
        return {"range": [self.lo, self.hi]}

    @classmethod
    def from_args(cls, a, **kw):
        return cls(*a["range"], **kw)


@dataclass(frozen=True)
class LogRange(Rule):
    lo: float
    hi: float
    kind: ClassVar[str] = "log_range"

    def args(self):
        return {"log_range": [self.lo, self.hi]}

    @classmethod
    def from_args(cls, a, **kw):
        return cls(*a["log_range"], **kw)


@dataclass(frozen=True)
class Values(Rule):
    values: tuple[Any, ...]  # Values([1, 2, 4])
    kind: ClassVar[str] = "values"

    def __post_init__(self):
        object.__setattr__(self, "values", tuple(self.values))

    def args(self):
        return {"values": list(self.values)}

    @classmethod
    def from_args(cls, a, **kw):
        return cls(a["values"], **kw)


@dataclass(frozen=True)
class Value(Rule):
    value: Any
    kind: ClassVar[str] = "value"

    def args(self):
        return {"value": self.value}

    @classmethod
    def from_args(cls, a, **kw):
        return cls(a["value"], **kw)


@dataclass(frozen=True)
class Scale(Rule):
    lo: float
    hi: float
    kind: ClassVar[str] = "scale"

    def args(self):
        return {"scale": [self.lo, self.hi]}

    @classmethod
    def from_args(cls, a, **kw):
        return cls(*a["scale"], **kw)


@dataclass(frozen=True)
class Dist(Rule):
    dist: str  # uniform | loguniform | normal | lognormal
    mean: Optional[float] = None
    sigma: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    kind: ClassVar[str] = "dist"

    def args(self):
        return {"dist": self.dist, **{k: getattr(self, k) for k in ("mean", "sigma", "min", "max")
                                      if getattr(self, k) is not None}}

    @classmethod
    def from_args(cls, a, **kw):
        return cls(a["dist"], **{k: a.get(k) for k in ("mean", "sigma", "min", "max")}, **kw)


RULES: dict[str, type[Rule]] = {c.kind: c for c in (Range, LogRange, Values, Value, Scale, Dist)}


def _rule_table(rule: Rule | Any) -> Any:
    return rule.to_table() if isinstance(rule, Rule) else rule


# -- the domain


class Domain:
    """The `[domain]` table: declared gridcells (hierarchy.py), sets, reference and overrides."""

    def __init__(self, gridcells: Sequence[Gridcell], nsets: int = 2, reference: Optional[str | Path] = None,
                 constants: Optional[dict[str, Any]] = None, time: Optional[dict[str, Any]] = None,
                 dims: Optional[dict[str, int]] = None, snow_layers: int = 0, init: Optional[str] = None):
        self.gridcells = list(gridcells)
        self.nsets = nsets
        self.reference = reference
        self.constants = dict(constants or {})
        self.time = dict(time or {})
        self.dims = dict(dims or {})
        self.snow_layers = snow_layers
        self.init = init
        Layout(self.gridcells, nsets)  # the structure and weights, checked now

    def to_table(self) -> dict:
        out: dict[str, Any] = {"nsets": self.nsets}
        if self.reference is not None:
            out["reference"] = str(self.reference)
        for key in ("constants", "time", "dims"):
            if getattr(self, key):
                out[key] = dict(getattr(self, key))
        if self.snow_layers:
            out["snow_layers"] = self.snow_layers
        if self.init is not None:
            out["init"] = self.init
        out["gridcell"] = [g.to_table() for g in self.gridcells]
        return out

    def spec(self, case: Case, base: Optional[Path] = None) -> DomainSpec:
        d = DomainSpec.from_table(self.to_table(), base or case.dir.parent, case.input_data)
        d.check(case.inv)
        return d

    def view(self, case: Optional[Case] = None) -> "DomainView":
        lay = Layout(self.gridcells, self.nsets)
        if case is not None:
            DomainSpec(self.gridcells, self.nsets).layout(case.inv.parameters)  # type codes
        return DomainView(lay)

    @classmethod
    def from_spec(cls, d: DomainSpec, input_data: Optional[Path] = None) -> "Domain":
        t = d.to_table(input_data)
        return cls(d.gridcells, d.nsets, t.get("reference"), d.constants, d.time, d.dims, d.snow_layers, d.init)


# -- querying a declared domain


@dataclass(frozen=True)
class PatchView:
    index: int
    pft: int
    weight: float
    wtgcell: float
    active: bool


@dataclass(frozen=True)
class ColumnView:
    index: int
    itype: int
    role: str
    weight: float
    active: bool
    patches: tuple[PatchView, ...]


@dataclass(frozen=True)
class LandunitView:
    index: int
    kind: LandunitKind
    density: Optional[UrbanDensity]
    itype: int
    weight: float
    active: bool
    columns: tuple[ColumnView, ...]


@dataclass(frozen=True)
class TopounitView:
    index: int
    number: int  # within its gridcell (topo_grc_ind)
    weight: float
    active: bool
    elevation: Optional[float]
    landunits: tuple[LandunitView, ...]


@dataclass(frozen=True)
class GridcellView:
    index: int
    set: int
    name: str
    topounits: tuple[TopounitView, ...]

    @property
    def landunits(self) -> list[LandunitView]:
        return [lu for t in self.topounits for lu in t.landunits]

    @property
    def columns(self) -> list[ColumnView]:
        return [c for lu in self.landunits for c in lu.columns]

    @property
    def patches(self) -> list[PatchView]:
        return [p for c in self.columns for p in c.patches]


class DomainView:
    """A declared domain as ELM will hold it: element indices (Fortran), hierarchy, weights."""

    def __init__(self, layout: Layout):
        self.layout = layout
        self.sizes = layout.sizes()
        self.gridcells: list[GridcellView] = [self._gridcell(g) for g in range(layout.count("g"))]

    def _gridcell(self, g: int) -> GridcellView:
        y = self.layout
        act = {lev: y.active(lev) for lev in ("t", "l", "c", "p")}
        wt = {lev: y.weight(lev, {"t": "g", "l": "t", "c": "l", "p": "c"}[lev]) for lev in ("t", "l", "c", "p")}
        wg = y.weight("p", "g")
        ctype, pft, ltype = y.itype("c"), y.itype("p"), y.itype("l")
        from spel.proptest.hierarchy import column_role

        def kids(level: str, k: int, finer: str) -> range:
            return range(int(y.first_child(level, finer)[k]) - 1, int(y.last_child(level, finer)[k]))

        tops = []
        for t in kids("g", g, "t"):
            lus = []
            for lu in kids("t", t, "l"):
                decl = y.levels["l"].decl[lu]
                cols = []
                for c in kids("l", lu, "c"):
                    pats = tuple(PatchView(p + 1, int(pft[p]), float(wt["p"][p]), float(wg[p]), bool(act["p"][p]))
                                 for p in kids("c", c, "p"))
                    cols.append(ColumnView(c + 1, int(ctype[c]), column_role(int(ctype[c])), float(wt["c"][c]),
                                           bool(act["c"][c]), pats))
                lus.append(LandunitView(lu + 1, decl.kind, decl.density, int(ltype[lu]), float(wt["l"][lu]),
                                        bool(act["l"][lu]), tuple(cols)))
            tops.append(TopounitView(t + 1, y.topo_number[t], float(wt["t"][t]), bool(act["t"][t]),
                                     y.elevation[t], tuple(lus)))
        return GridcellView(g + 1, g // y.per_set_gridcells, y.levels["g"].decl[g].label, tuple(tops))

    def table(self, level: str):
        """pandas DataFrame, one row per element (hierarchy.Layout.table)."""
        return self.layout.table(level)

    def select(self, level: str, **where: Any) -> np.ndarray:
        """Fortran indices of the elements whose table columns equal the given values
        (`select("p", name="mixed", kind="soil", pft=13)`; a list means any of)."""
        df = self.table(level)
        mask = np.ones(len(df), dtype=bool)
        for key, val in where.items():
            if key not in df.columns:
                raise KeyError(f"{key}: not a column of the {level} table ({list(df.columns)})")
            vals = val if isinstance(val, (list, tuple, set)) else [val]
            vals = [v.value if hasattr(v, "value") else v for v in vals]
            mask &= df[key].isin(vals).to_numpy()
        return df.index.to_numpy()[mask]

    def expected_filters(self) -> dict[str, np.ndarray]:
        """The members setFilters will build from this topology (domain.EXPECTED_FILTERS)."""
        return expected_filters(self.layout)


# -- scenarios and the spec


class Scenario:
    """A `[scenario.<name>]` table, checked against the case as it is built."""

    def __init__(self, spec: "Spec", name: str, over: str = "p", extends: Sequence[str | Path] = (),
                 inputs: Optional[dict[str, Rule | Any]] = None, derive: Optional[dict[str, str]] = None,
                 assume: Sequence[str] = (), namelist: Optional[dict[str, Any]] = None, trap_fpe: bool = False,
                 steps: int = 1, time: Optional[dict[str, Any]] = None,
                 forcing: Optional[dict[str, Rule | Any | str]] = None, between: Optional[dict[str, str]] = None,
                 keep: Sequence[str] = (), carry: Optional[dict[str, Rule | Any]] = None):
        """Multi-step runs (SPEC.md "Multi-step runs"): `steps` calls per example; `time` of step 1;
        `forcing`: a rule (drawn for every step) or an expression (str) per boundary input;
        `between`: expressions applied after every step; `keep`: boundary variables left to the
        process (fnmatch patterns); `carry`: outputs that keep their last value where a step does
        not write them (not NaN), with their value before step 1."""
        self.spec = spec
        self.name = name
        self.over = over
        self.extends = [Path(p).resolve() for p in ([extends] if isinstance(extends, (str, Path)) else extends)]
        self.inputs: dict[str, Rule | Any] = {}
        self.derive_: dict[str, str] = {}
        self.assume_: list[str] = []
        self.namelist = dict(namelist or {})
        self.trap_fpe = trap_fpe
        self.steps = steps
        self.time = dict(time or {})
        self.forcing: dict[str, Rule | Any | str] = {spec.case.canonical(k): v for k, v in (forcing or {}).items()}
        self.between: dict[str, str] = {spec.case.canonical(k): v for k, v in (between or {}).items()}
        self.keep = list(keep)
        self.carry: dict[str, Rule | Any] = {spec.case.canonical(k): v for k, v in (carry or {}).items()}
        for k, r in (inputs or {}).items():
            self.inputs[spec.case.canonical(k)] = r
        for k, e in (derive or {}).items():
            self.derive_[spec.case.canonical(k)] = e
        self.assume_ = list(assume)
        self.check()

    def input(self, var: str, rule: Rule | Any) -> "Scenario":
        self.inputs[self.spec.case.canonical(var)] = rule
        self.check()
        return self

    def derive(self, var: str, expr: str) -> "Scenario":
        self.derive_[self.spec.case.canonical(var)] = expr
        self.check()
        return self

    def assume(self, expr: str) -> "Scenario":
        self.assume_.append(expr)
        self.check()
        return self

    def to_table(self, base: Optional[Path] = None) -> dict:
        out: dict[str, Any] = {"over": self.over}
        if self.extends:
            ext = [_relative(p, base) for p in self.extends]
            out["extends"] = ext[0] if len(ext) == 1 else ext
        if self.trap_fpe:
            out["trap_fpe"] = True
        if self.namelist:
            out["namelist"] = dict(self.namelist)
        if self.assume_:
            out["assume"] = list(self.assume_)
        if self.inputs:
            out["inputs"] = {k: _rule_table(r) for k, r in self.inputs.items()}
        if self.derive_:
            out["derive"] = dict(self.derive_)
        if self.steps != 1:
            out["steps"] = self.steps
        if self.time:
            out["time"] = dict(self.time)
        if self.forcing:
            out["forcing"] = {k: (v if isinstance(v, str) else _rule_table(v)) for k, v in self.forcing.items()}
        if self.between:
            out["between"] = dict(self.between)
        if self.keep:
            out["keep"] = list(self.keep)
        if self.carry:
            out["carry"] = {k: _rule_table(r) for k, r in self.carry.items()}
        return out

    def check(self) -> sp.Scenario:
        """The scenario through the spec reader (SpecError with the location otherwise)."""
        return sp._scenario(self.name, self.to_table(), Path("/"), self.spec.case.inv, self.spec.names())

    def __str__(self) -> str:
        return self.name


class Spec:
    """A whole spec: case, domain, namelist, scenarios, checks; `to_toml`, `validate`, `session`."""

    def __init__(self, case: Case, domain: Domain, namelist: Optional[dict[str, Any]] = None,
                 max_examples: int = 50, settings: Optional[dict[str, Any]] = None,
                 units: Optional[dict[str, str]] = None, functions: Sequence[str | Path] = (),
                 compare: Optional[dict[str, Any]] = None, build: Optional[dict[str, Any]] = None,
                 path: Optional[str | Path] = None):
        self.case = case
        self.domain = domain
        self.namelist = dict(namelist or {})
        self.settings = {"max_examples": max_examples, **(settings or {})}
        self.units = dict(units or {})
        self.functions = [Path(f).resolve() for f in functions]
        self.compare = dict(compare) if compare else None
        self.build = dict(build or {})
        self.path = Path(path).resolve() if path else case.dir.parent / f"{case.name}.proptest.toml"
        self.scenarios: dict[str, Scenario] = {}
        self.checks: list[dict[str, Any]] = []  # (table key, table)
        self.comments: dict[tuple, str] = {}
        domain.spec(case, self.path.parent)  # checked now
        sp.namelist_table(self.namelist, case.inv, "[namelist]")

    def names(self) -> sp.Names:
        from spel.proptest import functions as fn

        funcs = fn.load({"modules": [str(f) for f in self.functions]} if self.functions else {}, self.path.parent)
        return sp.Names(self.case.inv, set(funcs), self.domain.spec(self.case, self.path.parent))

    # -- pieces
    def scenario(self, name: str, *, over: str = "p", extends: Sequence[str | Path] | str | Path = (),
                 inputs: Optional[dict[str, Rule | Any]] = None, derive: Optional[dict[str, str]] = None,
                 assume: Sequence[str] = (), namelist: Optional[dict[str, Any]] = None,
                 trap_fpe: bool = False, comment: str = "", steps: int = 1, time: Optional[dict[str, Any]] = None,
                 forcing: Optional[dict[str, Rule | Any | str]] = None, between: Optional[dict[str, str]] = None,
                 keep: Sequence[str] = (), carry: Optional[dict[str, Rule | Any]] = None) -> Scenario:
        if name in self.scenarios:
            raise SpecError(f"scenario {name!r} exists")
        s = Scenario(self, name, over, extends, inputs, derive, assume, namelist, trap_fpe, steps, time, forcing,
                     between, keep, carry)
        self.scenarios[name] = s
        if comment:
            self.comments[("scenario", name)] = comment
        return s

    def property(self, name: str, expr: str, *, scenarios: Sequence[str | Scenario] = ("*",),
                 over: Optional[str] = None, comment: str = "", steps: Any = "all") -> None:
        """steps (multi-step scenarios): "all", "last", a step, or [first, last]."""
        self._check("property", {"name": name, "expr": expr}, scenarios, over, comment, steps)

    def relation(self, name: str, expect: str, *, follow: Optional[dict[str, str]] = None, permute: bool = False,
                 namelist: Optional[dict[str, Any]] = None, scenarios: Sequence[str | Scenario] = ("*",),
                 over: Optional[str] = None, comment: str = "", steps: Any = "all") -> None:
        t: dict[str, Any] = {"name": name}
        if follow:
            t["follow"] = {self.case.canonical(k): v for k, v in follow.items()}
        t["expect"] = expect
        if permute:
            t["permute"] = True
        if namelist:
            t["namelist"] = dict(namelist)
        self._check("relation", t, scenarios, over, comment, steps)

    def differential(self, name: str, *, outputs: str | Sequence[str] = "all", rtol: Optional[float] = None,
                     atol: Optional[float] = None, expect: Optional[str] = None,
                     scenarios: Sequence[str | Scenario] = ("*",), over: Optional[str] = None,
                     comment: str = "", steps: Any = "all") -> None:
        """Needs `compare` on the Spec (another case or build)."""
        t: dict[str, Any] = {"name": name}
        if expect is not None:
            t["expect"] = expect
        else:
            t["outputs"] = outputs if isinstance(outputs, str) else [self.case.canonical(o) for o in outputs]
            t.update({k: v for k, v in (("rtol", rtol), ("atol", atol)) if v is not None})
        self._check("differential", t, scenarios, over, comment, steps)

    def _check(self, key: str, t: dict, scenarios: Sequence[str | Scenario], over: Optional[str], comment: str,
               steps: Any = "all"):
        t = dict(t)
        if steps != "all":
            t["steps"] = list(steps) if isinstance(steps, tuple) else steps
        names = [str(s) for s in scenarios]
        if names != ["*"]:
            t["scenarios"] = names
        if over is not None:
            t["over"] = over
        if any(c["name"] == t["name"] for _, c in self.checks):
            raise SpecError(f"a check named {t['name']!r} exists")
        chk.KINDS[key].from_table(t, self.names())  # the reader's checks, with its messages
        self.checks.append((key, t))
        if comment:
            self.comments[(key, len([k for k, _ in self.checks if k == key]) - 1)] = comment

    # -- output
    def to_dict(self) -> dict:
        base = self.path.parent
        under = self.case.dir.resolve().parent == _unittests()
        out: dict[str, Any] = {"case": {"name": self.case.name} if under else {"dir": _relative(self.case.dir, base)}}
        if self.build:
            out["case"]["build"] = dict(self.build)
        out["settings"] = dict(self.settings)
        out["domain"] = self.domain.to_table()
        if self.namelist:
            out["namelist"] = dict(self.namelist)
        if self.units:
            out["units"] = dict(self.units)
        if self.functions:
            out["functions"] = {"modules": [_relative(f, base) for f in self.functions]}
        if self.compare:
            out["compare"] = dict(self.compare)
        out["scenario"] = {n: s.to_table(base) for n, s in self.scenarios.items()}
        for key, t in self.checks:
            out.setdefault(key, []).append(dict(t))
        return out

    def to_toml(self, path: Optional[str | Path] = None) -> str:
        """The TOML text; written to `path` (default: the Spec's path) when given or set."""
        if path is not None:
            self.path = Path(path).resolve()
        text = tomlw.dumps(self.to_dict(), self.comments)
        if path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(text)
        return text

    def validate(self) -> sp.Spec:
        """The spec as `spel proptest` reads it (SpecError otherwise)."""
        return sp.parse(self.to_dict(), self.path, self.case.dir, self.case.inv)

    def session(self, **kw: Any):
        """A runner.Session on the validated spec (keyword arguments as Session's)."""
        from spel.proptest.runner import Session

        return Session(self.validate(), self.case.inv, **kw)

    def view(self) -> DomainView:
        return self.domain.view(self.case)

    @classmethod
    def from_toml(cls, path: str | Path, inv: Optional[Inventory] = None) -> "Spec":
        """An existing spec file, to change it in Python (inv: the case's inventory, if loaded)."""
        import tomllib

        path = Path(path).resolve()
        data = tomllib.loads(path.read_text())
        parsed, inv = sp.load(path, inv=inv)
        case = Case.of(inv, parsed.domain.reference)
        dom = Domain.from_spec(parsed.domain, case.input_data)
        dom.reference = data.get("domain", {}).get("reference")
        out = cls(case, dom, namelist=data.get("namelist", {}), settings=data.get("settings", {}),
                  units=data.get("units", {}), compare=data.get("compare"),
                  build=data.get("case", {}).get("build"), path=path,
                  functions=[_absolute(f, path.parent) for f in data.get("functions", {}).get("modules", [])],
                  max_examples=data.get("settings", {}).get("max_examples", 50))
        for name, t in data.get("scenario", {}).items():
            ext = t.get("extends", [])
            ext = [ext] if isinstance(ext, str) else ext
            assume = t.get("assume", [])
            out.scenario(name, over=t.get("over", "p"), extends=[_absolute(e, path.parent) for e in ext],
                         inputs={k: Rule.from_table(v) for k, v in t.get("inputs", {}).items()},
                         derive=t.get("derive", {}), assume=[assume] if isinstance(assume, str) else assume,
                         namelist=t.get("namelist"), trap_fpe=t.get("trap_fpe", False),
                         steps=t.get("steps", 1), time=t.get("time"),
                         forcing={k: (v if isinstance(v, str) else Rule.from_table(v))
                                  for k, v in t.get("forcing", {}).items()},
                         between=t.get("between"), keep=t.get("keep", []),
                         carry={k: Rule.from_table(v) for k, v in t.get("carry", {}).items()})
        for key in chk.KINDS:
            for t in data.get(key, []):
                t = dict(t)
                scen = t.pop("scenarios", ["*"])
                over = t.pop("over", None)
                steps = t.pop("steps", "all")
                if key == "property":
                    out.property(t["name"], t["expr"], scenarios=scen, over=over, steps=steps)
                elif key == "relation":
                    out.relation(t["name"], t["expect"], follow=t.get("follow"), permute=t.get("permute", False),
                                 namelist=t.get("namelist"), scenarios=scen, over=over, steps=steps)
                else:
                    out.differential(t["name"], outputs=t.get("outputs", "all"), rtol=t.get("rtol"),
                                     atol=t.get("atol"), expect=t.get("expect"), scenarios=scen, over=over,
                                     steps=steps)
        return out


def _unittests() -> Path:
    from spel.scripts.config import unittests_dir

    return Path(unittests_dir).resolve()


def _relative(p: Path, base: Optional[Path]) -> str:
    if base is None:
        return str(p)
    rel = os.path.relpath(p, base)
    return str(p) if rel.count("..") > 2 else rel


def _absolute(p: str, base: Path) -> Path:
    q = Path(p)
    return q if q.is_absolute() else (base / q).resolve()
