"""
What an extracted routine reads and writes, from SPEL's analysis of the E3SM sources.

Sources (no text scanning):
- the variables ReadWriteMod captures: `netcdf_io.case_io_variables` (type_dict components of
  the active instances, with their namelist guards);
- read/write status: the static trace of the routines (`db_export.trace.StaticTrace`, dummies bound
  to elm_drv's actuals) categorized like the database does (`fieldstate.categorize`): a read before
  the first *unconditional* write, or only conditional writes, is `rw` (the routine needs the value);
  read only `r`; an unconditional write first `w`. Variables the trace doesn't reach fall back to the
  routines' `elmtype_access_summary` (`validate_access.access_manifest`/`expand_nested`, first access
  in source order);
- namelist configurations: every traced access keeps its *gates*, the namelist conditions of the
  branches around it and around every call on its chain (`transforms.core.config.gates`, from the
  analysis' `flat_ifs`). `Inventory.configured(values)` classifies again from the accesses that
  run under those namelist values, so a variable only used under `use_c13` is `unused` when it is
  off. Without values every branch counts as running;
- dimensions: each component's bounds, parsed with SPEL's parser (`fexpr.parse_expr`);
  the subgrid level is the `bounds_type` member a dimension starts at (`begp` ...);
- associate aliases: `Subroutine.record.frames`, dummy bases mapped to the actuals elm_drv
  binds (`fut.driver_bindings`);
- elm_drv's scalar arguments the routines take (`fut.driver_args`, e.g. `nextsw_cday`): scalar
  inputs under their own name, stored in ReadWriteMod's `spel_drv_<name>(1)`;
- constants: the FUTConstantsMod globals (`non_parameter_global_vars`) and module parameters
  (`module_dict[...].global_vars`, values parsed with the parser);
- units: `hist_addfld*`/`restartvar` registrations of the full elm_drv analysis
  (`registrations.Registration`), matched to instances through their type.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from spel.proptest.errors import ProptestError
from spel.proptest.fexpr import ExprError, designator, evaluate_bound, parse_expr, uses, Evaluator
from spel.proptest.paths import PROPTEST_DIR, work_dir
from spel.scripts.fortran_parser.spel_ast import BoundsExpression, Expression

log = logging.getLogger("spel.proptest")

# bounds_type members -> subgrid level (decompMod)
LEVEL_OF_BOUND = {
    "begg": "g", "endg": "g", "begt": "t", "endt": "t", "begl": "l", "endl": "l",
    "begc": "c", "endc": "c", "begp": "p", "endp": "p",
    "begg_all": "g", "endg_all": "g", "begl_all": "l", "endl_all": "l",
    "begc_all": "c", "endc_all": "c", "begp_all": "p", "endp_all": "p",
}
LEVELS = ("g", "t", "l", "c", "p")
LEVEL_NAMES = {"g": "gridcell", "t": "topounit", "l": "landunit", "c": "column", "p": "patch"}
# nc_io's subgrid dimension names (nc_subgrid_level)
NC_DIM = {"g": "gridcell", "t": "topo", "l": "landunit", "c": "column", "p": "patch"}
# real fields of the subgrid hierarchy the domain builder computes (weights, areas)
TOPOLOGY_REALS = {
    "wtgcell", "wtlunit", "wtcol", "wttopounit", "wt_ed", "wtgcell_save",
}

# integer fields of the subgrid types that are physical data, not structure: ruled inputs
# (col_pp%nlevbed, the depth to bedrock as a layer index, comes from the surface data)
PHYSICAL_INTEGERS = {"nlevbed"}

KINDS = ("topology", "parameter", "input", "inout", "output", "unused")
UNITS_FILE = "units.json"


@dataclass
class Dim:
    """One dimension: Fortran bounds as parsed expressions (start None = 1)."""

    text: str
    start: Optional[Expression]
    end: Optional[Expression]
    level: Optional[str]  # subgrid level, None for others

    def extent(self, env: dict[str, int]) -> tuple[int, int]:
        lo = 1 if self.start is None else evaluate_bound(self.start, env)
        if self.end is None:
            raise ExprError(f"dimension {self.text} has no upper bound")
        return lo, evaluate_bound(self.end, env)


@dataclass
class VarInfo:
    name: str  # inst%field (lower case)
    instance: str
    field: str
    type: str  # real | integer | logical
    rank: int
    dims: list[Dim]
    status: Optional[str]  # r | w | rw (first access), None: captured but not accessed
    kind: str = ""
    guard: Optional[str] = None  # Fortran condition under which ReadWriteMod reads it
    ptrscalar: bool = False
    units: str = ""
    long_name: str = ""
    driver: bool = False  # a scalar argument of elm_drv (ReadWriteMod's spel_drv_<name>(1))
    # for a variable `unused` under a configuration: the namelist options whose branches hold its accesses
    dead_under: tuple[str, ...] = ()

    @property
    def nc_name(self) -> str:
        if self.driver:
            return f"spel_drv_{self.field}"
        return f"{self.instance}__{self.field.replace('%', '__')}"

    @property
    def level(self) -> Optional[str]:
        return next((d.level for d in self.dims if d.level), None)

    @property
    def subgrid_axis(self) -> Optional[int]:
        return next((k for k, d in enumerate(self.dims) if d.level), None)

    def summary(self) -> dict:
        return {
            "name": self.name, "kind": self.kind, "type": self.type, "rank": self.rank,
            "dims": [d.text for d in self.dims], "level": self.level, "status": self.status,
            "guard": self.guard, "units": self.units, "long_name": self.long_name,
        }


@dataclass
class TracedAccess:
    """One access of a variable in the static step, with what decides whether it runs."""

    status: str  # r | w | rw
    conditional: bool  # inside a non-namelist if
    gates: tuple = ()  # namelist conditions (ConditionExpectation) it runs under


@dataclass
class ConstInfo:
    name: str
    module: str
    type: str
    rank: int
    dims: list[Dim]
    status: Optional[str]  # read by the routine (r) or not (None)


def parse_dims(bounds: str, rank: int, where: str) -> list[Dim]:
    """`begp:endp,1:nlevcan` (a component's bounds) -> Dims, through the parser."""
    if rank == 0:
        return []
    text = bounds.strip()
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1]
    try:
        ref = parse_expr(f"x({text})")
    except ExprError as exc:
        raise ExprError(f"{where}: bounds {bounds!r}: {exc}") from None
    args = getattr(ref, "args", None)
    if args is None or len(args) != rank:
        raise ExprError(f"{where}: bounds {bounds!r} do not give {rank} dimensions")
    dims = []
    for a in args:
        start, end = (a.start, a.end) if isinstance(a, BoundsExpression) else (None, a)
        level = None
        for e in (start, end):
            path = designator(e) if e is not None else None
            if path and path[-1] in LEVEL_OF_BOUND:
                level = LEVEL_OF_BOUND[path[-1]]
                break
        dims.append(Dim(text=str(a), start=start, end=end, level=level))
    return dims


@dataclass
class Inventory:
    case_dir: Path
    routines: list[str]
    vars: dict[str, VarInfo]
    constants: dict[str, ConstInfo]
    parameters: dict[str, Any]  # module parameters with a numeric value
    aliases: dict[str, str]  # associate name -> inst%field
    filters: dict[str, str]  # filter member -> integer type rank ('nolakeurbanp': 1, ...)
    instances: dict[str, str]  # instance -> type
    unit_test: Any = field(default=None, repr=False)
    accesses: dict[str, list[TracedAccess]] = field(default_factory=dict, repr=False)  # inst%field -> in step order
    fallback: dict[str, Optional[str]] = field(default_factory=dict, repr=False)  # status of untraced variables
    namelist: dict[str, Any] = field(default_factory=dict)  # option the routines test -> declared default (or None)
    config: dict[str, Any] = field(default_factory=dict)  # the namelist values this inventory is classified under
    _configured: dict = field(default_factory=dict, repr=False)

    # -- loading
    @classmethod
    def load(cls, case_dir: Path, units: bool = True) -> "Inventory":
        from spel.scripts.export_objects import unpickle_case

        case_dir = Path(case_dir)
        if not (case_dir / "fut.pkl").is_file():
            raise ProptestError(f"{case_dir} is not a unit-test case (no fut.pkl): make it with `spel create`")
        ut = unpickle_case(str(case_dir))
        inv = cls.from_unit_test(ut, case_dir)
        if units:
            inv.attach_units(load_units(case_dir, ut))
        return inv

    @classmethod
    def from_unit_test(cls, ut, case_dir: Path) -> "Inventory":
        from spel.scripts.io.netcdf_io import SUBGRID_INDEX_FIELDS, case_io_variables
        from spel.scripts.validate_access import access_manifest, expand_nested

        inst_types = ut.instance_to_type_map()
        subs = list(ut.primary_subroutines.values())
        fallback = {k.lower(): v for k, v in expand_nested(access_manifest(subs), ut.type_dict, inst_types).items()}
        traced = trace_accesses(ut)
        io_vars, guards = case_io_variables(ut)

        out: dict[str, VarInfo] = {}
        for name, var in io_vars.items():
            key = name.lower()
            inst, _, comp = key.partition("%")
            guard = guards.get(name)
            info = VarInfo(
                name=key, instance=inst, field=comp, type=var.type.lower(), rank=var.dim,
                dims=parse_dims(var.bounds, var.dim, key), status=status_of(traced.get(key), fallback.get(key), {})[0],
                guard=guard.to_fortran() if guard is not None else None, ptrscalar=bool(var.ptrscalar),
            )
            info.kind = classify(info, inst_types.get(inst, ""), set(SUBGRID_INDEX_FIELDS))
            out[key] = info

        for arg, typ in (getattr(ut, "driver_args", None) or {}).items():
            key = arg.lower()
            out[key] = VarInfo(name=key, instance="", field=key, type=str(typ).lower(), rank=0, dims=[],
                               status="r", kind="input", driver=True)

        consts: dict[str, ConstInfo] = {}
        read_globals = set()
        for s in subs:
            read_globals |= {k.lower() for k in s.record_access.all_globals()}
        for name, var in ut.non_parameter_global_vars.items():
            consts[name.lower()] = ConstInfo(
                name=name.lower(), module=var.declaration, type=var.type.lower(), rank=var.dim,
                dims=_const_dims(var), status="r" if name.lower() in read_globals else None,
            )

        filters = {}
        clumpfilter = ut.type_dict.get("clumpfilter")
        if clumpfilter is not None:
            filters = {k.lower(): v.dim for k, v in clumpfilter.components.items() if v.type.lower() == "integer"}

        from spel.transforms.core.config import namelist_defaults, namelist_names

        defaults = namelist_defaults(ut)
        return cls(
            case_dir=Path(case_dir), routines=list(ut.primary_subroutines), vars=out, constants=consts,
            parameters=module_parameters(ut), aliases=routine_aliases(ut), filters=filters,
            instances=dict(inst_types), unit_test=ut, accesses=traced, fallback=fallback,
            namelist={n: defaults.get(n) for n in sorted(namelist_names(ut))},
        )

    # -- namelist configurations
    def gate_names(self) -> set[str]:
        """The namelist variables that decide whether some access runs."""
        if "_gate_names" not in self.__dict__:
            names: set[str] = set()
            for accesses in self.accesses.values():
                for a in accesses:
                    for g in a.gates:
                        names.update(n.lower() for n in g.variable_names())
            self.__dict__["_gate_names"] = names
        return self.__dict__["_gate_names"]

    def options(self) -> dict[str, Any]:
        """The namelist options a spec can set: tested by the routines and a scalar constant of
        the case (so the constants file carries their value). option -> declared default."""
        return {n: d for n, d in self.namelist.items() if n in self.constants and self.constants[n].rank == 0}

    def configured(self, values: dict[str, Any]) -> "Inventory":
        """This inventory with every variable classified under namelist values (see module doc):
        `values` are the constants of a run (names in lower case; other entries are ignored)."""
        from spel.scripts.io.netcdf_io import SUBGRID_INDEX_FIELDS

        relevant = {k: _scalar(values[k]) for k in sorted(self.gate_names()) if k in values and _scalar(values[k]) is not None}
        key = tuple(relevant.items())
        if key not in self._configured:
            physical = set(SUBGRID_INDEX_FIELDS)
            new: dict[str, VarInfo] = {}
            for name, v in self.vars.items():
                if v.driver or v.instance == "bounds" or not self.accesses.get(name):
                    new[name] = v  # no traced access to decide: as it is
                    continue
                status, dead = status_of(self.accesses[name], self.fallback.get(name), relevant)
                info = dataclasses.replace(v, status=status, dead_under=dead)
                info.kind = classify(info, self.instances.get(v.instance, ""), physical)
                new[name] = info
            self._configured[key] = dataclasses.replace(self, vars=new, config=relevant)
        return self._configured[key]

    def fortran_value(self, name: str, value: Any) -> str:
        """A constant's value as Fortran writes it (logical constants are stored as integers)."""
        c = self.constants.get(name)
        if isinstance(value, bool) or (c is not None and c.type == "logical"):
            return ".true." if value else ".false."
        return str(value)

    def why_unused(self, name: str) -> str:
        """Why a variable has no value to offer under this inventory's configuration."""
        v = self.vars[name]
        if v.dead_under:
            opts = ", ".join(f"{o} = {self.fortran_value(o, self.config[o])}" if o in self.config else o
                             for o in v.dead_under)
            return f"it is only used under other namelist options (here {opts})"
        return "the routines do not use it"

    # -- queries
    def of_kind(self, *kinds: str) -> list[VarInfo]:
        return [v for v in self.vars.values() if v.kind in kinds]

    def canonical(self, path: list[str] | tuple[str, ...]) -> Optional[str]:
        """`inst%field`, an associate alias, or a flat netCDF name -> inst%field."""
        name = "%".join(path).lower()
        if name in self.vars:
            return name
        if len(path) == 1 and name in self.aliases and self.aliases[name] in self.vars:
            return self.aliases[name]
        flat = {v.nc_name: v.name for v in self.vars.values()}
        return flat.get(name)

    def attach_units(self, units: dict[str, tuple[str, str]]) -> None:
        for name, (u, long_name) in units.items():
            if name in self.vars:
                self.vars[name].units = u
                self.vars[name].long_name = long_name

    def bound_env(self, sizes: dict[str, int], constants: dict[str, Any]) -> dict[str, int]:
        """Names a dimension may use: bounds_type members (from domain sizes), parameters, constants."""
        env: dict[str, int] = {}
        for k, v in self.parameters.items():
            if isinstance(v, (int,)):
                env[k] = v
        for k, v in constants.items():
            try:
                if getattr(v, "ndim", 0) == 0 and float(v) == int(v):
                    env[k] = int(v)
            except (TypeError, ValueError, OverflowError):
                pass
        for lev in LEVELS:
            n = sizes.get(lev)
            if n is None:
                continue
            for b, l in LEVEL_OF_BOUND.items():
                if l == lev:
                    env[b] = 1 if b.startswith("beg") else n
        return env

    def bound_names(self) -> set[str]:
        """Names the dimension bounds use besides the bounds_type members (`nlevsno`, `numrad`):
        what `[domain.dims]` may set."""
        names: set[str] = set()
        for v in [*self.vars.values(), *self.constants.values()]:
            for d in v.dims:
                for e in (d.start, d.end):
                    if e is not None:
                        names |= {"%".join(u.path) for u in uses(e) if u.path[-1] not in LEVEL_OF_BOUND}
        return names

    def time_names(self) -> Optional[set[str]]:
        """The case's per-step scalars (`[domain.time]` keys); None without the case's analysis."""
        if self.unit_test is None:
            return None
        return {s.name.lower() for s in getattr(self.unit_test, "time_scalars", None) or []}

    def guard_active(self, var: VarInfo, constants: dict[str, Any]) -> bool:
        """Evaluate ReadWriteMod's namelist guard of `var` over the constants (parsed)."""
        if not var.guard:
            return True

        def resolve(path):
            name = "%".join(path)
            if name not in constants:
                raise KeyError(name)
            return constants[name]

        try:
            return bool(Evaluator(resolve)(parse_expr(var.guard)))
        except ExprError as exc:
            raise ExprError(f"{var.name}: guard {var.guard}: {exc}") from None

    def save(self, path: Path) -> None:
        data = {
            "routines": self.routines,
            "variables": [v.summary() for v in self.vars.values()],
            "constants": [{"name": c.name, "module": c.module, "type": c.type, "rank": c.rank,
                           "status": c.status} for c in self.constants.values()],
            "aliases": self.aliases,
        }
        path.write_text(json.dumps(data, indent=1))


CATEGORY_STATUS = {"prognostic": "rw", "input": "r", "diagnostic": "w", "intermediate": "w"}


def _scalar(v: Any) -> Any:
    """A constant's value as a gate can use it (None for arrays)."""
    if getattr(v, "ndim", 0):
        return None
    return v.item() if hasattr(v, "item") else v


def status_of(accesses: Optional[list[TracedAccess]], fallback: Optional[str],
              values: dict[str, Any]) -> tuple[Optional[str], tuple[str, ...]]:
    """(r | w | rw | None, options hiding it) of a variable under namelist values: categorized
    from the accesses none of whose gates is decided false. An untraced variable keeps the
    access manifest's status. None with option names: every access is in a branch that is off."""
    from spel.scripts.db_export.fieldstate import categorize
    from spel.scripts.validate_access import evaluate_guard

    if not accesses:
        return fallback, ()
    live, hidden = [], set()
    for a in accesses:
        off = [g for g in a.gates if evaluate_guard(g, values) is False]
        if off:
            for g in off:
                hidden.update(n.lower() for n in g.variable_names())
        else:
            live.append((a.status, a.conditional))
    if not live:
        return None, tuple(sorted(hidden))
    return CATEGORY_STATUS.get(categorize(live, False)), ()


def trace_status(ut) -> dict[str, str]:
    """`inst%member` -> r | w | rw from the static trace of the case's routines, every branch
    counted as running (see module doc)."""
    out = {}
    for key, accesses in trace_accesses(ut).items():
        status = status_of(accesses, None, {})[0]
        if status:
            out[key] = status
    return out


def trace_accesses(ut) -> dict[str, list[TracedAccess]]:
    """`inst%member` -> its accesses in the static trace of the case's routines, in step order,
    each with the namelist gates of its line and of the calls leading to it."""
    from spel.proptest.shim import global_module, instance_modules
    from spel.scripts.db_export.context import ExportContext
    from spel.scripts.db_export.trace import StaticTrace
    from spel.scripts.fortran_parser.symbols import Origin
    from spel.transforms.compact.index_forms import IndexContext
    from spel.transforms.core.config import gates

    ctx = ExportContext(ut, {}, Path(getattr(ut, "e3sm_srcroot", "/") or "/"))
    trace = StaticTrace(ctx, [])
    inst_mod = instance_modules(ut)
    bindings: dict[str, list] = {}
    for b in getattr(ut, "driver_bindings", []):
        bindings.setdefault(b.callee, []).append(b)
    for sub_id, sub in ut.primary_subroutines.items():
        env: dict[str, frozenset] = {}
        for b in bindings.get(sub_id, []):
            if b.origin not in (Origin.GLOBAL, Origin.EXTERNAL):
                continue
            base, _, rest = b.actual.lower().partition("%")
            mod = global_module(ut, base) or inst_mod.get(base)
            if mod:
                env[b.dummy.lower()] = frozenset({f"{mod}::{base}" + (f"%{rest}" if rest else "")})
        trace.visit(sub, env, None, (sub.id,), None, None, False, IndexContext(sub.id, sub.record, {}, {}))
    subs = {s.id: s for s in ut.subroutine_dict.values()}
    chain: dict[int, tuple] = {}  # trace node -> gates of the calls leading to it

    def chain_gates(order: int) -> tuple:
        if order not in chain:
            node = trace.nodes[order]
            parent = node["parent_order"]
            if parent == "" or parent is None:
                chain[order] = ()
            else:
                caller = subs.get(trace.nodes[parent]["subroutine"])
                here = tuple(gates(caller, node["call_ln"])) if caller is not None and node["call_ln"] != "" else ()
                chain[order] = chain_gates(parent) + here
        return chain[order]

    seq: dict[str, list[TracedAccess]] = {}
    for a in sorted(trace.accesses, key=lambda a: a["order"]):
        if a["instance"] and a["member"]:
            sub_ = subs.get(trace.nodes[a["node_order"]]["subroutine"])
            own = tuple(gates(sub_, a["ln"])) if sub_ is not None else ()
            seq.setdefault(f"{a['instance']}%{a['member']}".lower(), []).append(
                TracedAccess(a["status"], bool(a["conditional"]), chain_gates(a["node_order"]) + own))
    return seq


def classify(info: VarInfo, type_name: str, physical_types: set[str]) -> str:
    """topology | parameter | input | inout | output | unused (see module doc)."""
    if info.status is None:
        return "unused"
    if type_name in physical_types and (info.type in ("integer", "logical") or info.field in TOPOLOGY_REALS) \
            and info.field not in PHYSICAL_INTEGERS:
        return "topology"
    if info.level is None:
        return "parameter" if "r" in info.status or info.status != "w" else "output"
    return {"r": "input", "w": "output", "rw": "inout"}.get(info.status, "inout")


def _const_dims(var) -> list[Dim]:
    if var.dim == 0:
        return []
    try:
        return parse_dims(var.bounds, var.dim, var.name)
    except ExprError:
        return [Dim(text=":", start=None, end=None, level=None) for _ in range(var.dim)]


def module_parameters(ut) -> dict[str, Any]:
    """Integer/real module parameters with a literal (or parameter-only) value, evaluated."""
    raw: dict[str, Expression] = {}
    for mod in ut.module_dict.values():
        for name, var in getattr(mod, "global_vars", {}).items():
            if getattr(var, "parameter", False) and var.default_value not in (None, ""):
                try:
                    raw[name.lower()] = parse_expr(str(var.default_value))
                except ExprError:
                    continue
    values: dict[str, Any] = {}
    for _ in range(4):  # parameters defined from other parameters
        for name, node in raw.items():
            if name in values:
                continue

            def resolve(path):
                key = "%".join(path)
                if key not in values:
                    raise KeyError(key)
                return values[key]

            try:
                v = Evaluator(resolve)(node)
            except ExprError:
                continue
            if getattr(v, "ndim", 1) == 0 and getattr(v, "dtype", None) is not None and v.dtype.kind in "iuf":
                values[name] = int(v) if v.dtype.kind in "iu" else float(v)
    return values


def routine_aliases(ut) -> dict[str, str]:
    """Associate names of the routines -> inst%field, dummy bases mapped to elm_drv's actuals."""
    actual_of: dict[str, dict[str, str]] = {}
    for b in getattr(ut, "driver_bindings", []):
        actual_of.setdefault(b.callee, {})[b.dummy.lower()] = b.actual.lower()
    out: dict[str, str] = {}
    for sub_id, sub in ut.primary_subroutines.items():
        dummies = actual_of.get(sub_id, {})
        for frame in sub.record.frames:
            for alias, target in frame.names.items():
                base, sep, rest = target.lower().partition("%")
                if not sep:
                    continue
                base = dummies.get(base, base)
                out.setdefault(alias.lower(), f"{base}%{rest}")
    return out


# -- units from registrations (full analysis, cached per case)


def load_units(case_dir: Path, ut) -> dict[str, tuple[str, str]]:
    cache = work_dir(case_dir) / UNITS_FILE
    if cache.is_file():
        return {k: tuple(v) for k, v in json.loads(cache.read_text()).items()}
    units = registration_units(ut)
    if units is None:
        return {}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(units, indent=1, sort_keys=True))
    return units


def registration_units(ut) -> Optional[dict[str, tuple[str, str]]]:
    """inst%field -> (units, long_name) from the full elm_drv analysis' registrations."""
    from spel.scripts.analysis_cache import ANALYSIS_PICKLE, cache_dir
    from spel.scripts.export_objects import _load_pickle

    srcroot = getattr(ut, "e3sm_srcroot", None)
    path = cache_dir(Path(srcroot)) / ANALYSIS_PICKLE if srcroot else None
    if path is None or not path.is_file():
        log.warning("no elm_drv analysis for units (%s): run `spel analyze`", path)
        return None
    analysis = _load_pickle(path)
    by_type: dict[str, list[str]] = {}
    for inst, typ in ut.instance_to_type_map().items():
        by_type.setdefault(typ.lower(), []).append(inst.lower())
    out: dict[str, tuple[str, str]] = {}
    for sub in analysis.subroutine_dict.values():
        for r in getattr(sub, "registrations", None) or []:
            if not r.field or not r.units:
                continue
            dummy = sub.arguments.get(r.base)
            if dummy is not None:
                # dummies are typed `class(t)`/`type(t)` (same unwrapping as db_export/routines.py)
                tname = dummy.type.lower().replace("class(", "").replace("type(", "").rstrip(")")
                instances = [i for i in by_type.get(tname, []) if r.applies_to(i)]
            else:
                instances = [r.base.lower()]
            for inst in instances:
                out.setdefault(f"{inst}%{r.field.lower()}", (r.units, r.long_name or ""))
    return out
