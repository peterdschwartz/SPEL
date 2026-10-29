"""
Generate the device data layer and the driver calls for a full-model OpenACC run.

The modules go to `components/elm/src/data_types/`: one `DeepCopy<Group>Mod.F90` per
family of types (`GROUPS`: parameters, subgrid, coupling, biogeophysics, vegetation and
soil nutrients), each with four routines per type (`deepcopy_<type>`, `attach_<type>`,
`update_<type>`, `check_<type>`) and the entry points for the group's instances, and
`ElmDeviceDataMod.F90`, which calls the groups. It gives ELM three entry points, all
no-ops without `_OPENACC`:

- `elm_device_init()`: device selection and `acc_init`, called first in `initialize1`
  (before any module data is touched by ELM; data in `declare` directives was already
  placed on the default device at program start, so the run must give each rank one
  visible device, e.g. `srun --gpus-per-task=1`).
- `elm_device_copyin()`: the one-time deep copy of every type instance of the analysis
  to the device, called in `elm_drv` on the first time step (after initialization and
  restart reads), guarded by `elm_device_resident`. Instances with `!$acc declare
  create` get their components copied under the already present object; the others get
  `enter data copyin` of the object first. Pointer components the analysis records as
  aliases (`=>` to another component) are attached instead of copied. After the copy
  every array is checked with `acc_is_present` and the misses are reported.
- `elm_device_update(to_device)`: `update device`/`update self` of every component of
  every instance (brute force: the whole state), called at the step boundaries and
  around the host-side sections of `elm_drv`.

Everything is generated from `fut.type_dict` (instances, components, nesting, pointer
targets) so it follows the sources it is applied to.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

MODULE_NAME = "ElmDeviceDataMod"
TAG = "! spel device-data"
INTRINSIC = {"real", "integer", "logical", "character", "double", "complex", "double precision"}
MAX_DEPTH = 5


class DeviceDataError(RuntimeError):
    pass


@dataclass
class Instance:
    name: str
    module: str
    type_name: str
    dim: int
    allocatable: bool
    pointer: bool
    declared: bool  # has an `!$acc declare` in its module


@dataclass
class Plan:
    instances: list[Instance]
    types: dict[str, object]  # type name -> DerivedType (only the ones reached)
    skipped: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    groups: dict[str, list[str]] = field(default_factory=dict)  # group key -> type names
    globals: list = field(default_factory=list)  # GlobalVar: declared module variables
    notes_guards: dict[str, str] = field(default_factory=dict)  # instance -> namelist guard
    ncomp_guards: int = 0  # type components under a namelist guard


# --- analysis side -------------------------------------------------------------------


def _module_path(mod) -> Path:
    """A module's source file; analyses store it absolute or relative to the E3SM srcroot."""
    from spel.scripts.config import E3SM_SRCROOT

    p = Path(mod.filepath)
    return p if p.is_absolute() else Path(E3SM_SRCROOT) / p


def _spec(path: Optional[Path]):
    """The parsed specification part of a module file (module_spec), or None."""
    from spel.scripts.fortran_parser.module_spec import module_spec

    if path is None or not Path(path).exists():
        return None
    try:
        return module_spec(Path(path))
    except Exception:  # the parser rejects the file: callers treat it as unknown
        return None


def _acc_declared_names(path: Path) -> set[str]:
    """Names in the `!$acc declare` directives of a source file."""
    spec = _spec(path)
    return set(spec.declared) if spec else set()


def type_is_public(dtype, fut) -> bool:
    """Whether the type can be `use`d from another module (from the parsed definition)."""
    mod = fut.module_dict.get(str(dtype.declaration).lower())
    spec = _spec(_module_path(mod)) if mod is not None else None
    t = spec.types.get(dtype.type_name.lower()) if spec else None
    return t is not None and t.access == "public"


def variable_is_public(name: str, mod) -> bool:
    """Whether module variable `name` can be `use`d (from the parsed declarations)."""
    spec = _spec(_module_path(mod))
    v = spec.variables.get(name.lower()) if spec else None
    return v is not None and v.access == "public"


def plan(fut, only_instances: Optional[set[str]] = None, source_of=None) -> Plan:
    """
    Every module-level instance of an analyzed derived type, with what the copy needs.
    `only_instances` restricts the instances (a unit test's); `source_of(mod)` gives the
    file whose `!$acc declare` directives count (a unit test's copy of the module).
    """
    private = {k for k, t in fut.type_dict.items() if not type_is_public(t, fut)}
    type_dict = {k: t for k, t in fut.type_dict.items() if k not in private}
    declared_cache: dict[str, set[str]] = {}
    instances: list[Instance] = []
    skipped: list[str] = []
    reached: dict[str, object] = {}

    def reach(tname: str, depth: int = 0) -> None:
        t = type_dict.get(tname.lower())
        if t is None or tname.lower() in reached or depth > MAX_DEPTH:
            return
        reached[tname.lower()] = t
        for c in t.components.values():
            if str(c.type).lower() not in INTRINSIC:
                reach(str(c.type), depth + 1)

    for k in sorted(private):
        skipped.append(f"type {k}: private in its module (and its instances)")
    for dtype in type_dict.values():
        for inst in dtype.instances.values():
            mod = (inst.declaration or "").lower()
            if not mod or mod not in fut.module_dict:
                skipped.append(f"{inst.name}: no declaring module ({inst.declaration!r})")
                continue
            if getattr(inst, "private", False):
                skipped.append(f"{mod}::{inst.name}: private")
                continue
            gv = fut.module_dict[mod].global_vars
            if inst.name not in gv and inst.name.lower() not in {g.lower() for g in gv}:
                # the analysis also registers nested-type components (veg_cf%core) as
                # instances of the component's type; only module variables are copied here
                skipped.append(f"{mod}::{inst.name}: not a module variable (nested component)")
                continue
            if getattr(inst, "parameter", False):
                skipped.append(f"{mod}::{inst.name}: parameter (constant, cannot be a data-clause item)")
                continue
            if not variable_is_public(inst.name, fut.module_dict[mod]):
                skipped.append(f"{mod}::{inst.name}: private module variable")
                continue
            if not dtype.components:
                skipped.append(f"{mod}::{inst.name}: type {dtype.type_name} has no components")
                continue
            if only_instances is not None and inst.name.lower() not in only_instances:
                continue
            fpath = source_of(fut.module_dict[mod]) if source_of else _module_path(fut.module_dict[mod])
            if fpath is None:
                skipped.append(f"{mod}::{inst.name}: module not in the case")
                continue
            if mod not in declared_cache:
                declared_cache[mod] = _acc_declared_names(fpath) if fpath.exists() else set()
            instances.append(
                Instance(
                    name=inst.name,
                    module=mod,
                    type_name=dtype.type_name,
                    dim=int(inst.dim or 0),
                    allocatable=bool(getattr(inst, "allocatable", False)),
                    pointer=bool(getattr(inst, "ptrscalar", False)),
                    declared=inst.name.lower() in declared_cache[mod],
                )
            )
            reach(dtype.type_name)
    instances.sort(key=lambda i: (i.module, i.name))
    return Plan(instances=instances, types=reached, skipped=skipped, notes=sorted(private))


# modules that move their own module data (history buffers, compact groups)
GLOBALS_EXCLUDE = {"histcomputemod", "compactlayoutmod"}
GLOBALS_MODULE = "DeepCopyGlobalsMod"


@dataclass
class GlobalVar:
    name: str
    module: str
    dim: int
    allocatable: bool
    pointer: bool = False


def plan_globals(fut, type_dict, source_of=None) -> tuple[list[GlobalVar], list[str]]:
    from spel.scripts.io.netcdf_io import alloc_status

    """
    Module variables named in an `!$acc declare` (create or copyin) of an analyzed
    module: they live on the device from program start, but their values are set at run
    time (namelists, parameter files, per-step time info), so the device copy must be
    updated. Derived-type instances are handled by the type groups; parameters never
    change; private variables cannot be named here (their modules must update them).
    """
    from spel.scripts.io.netcdf_io import alloc_status

    out: list[GlobalVar] = []
    skipped: list[str] = []
    for mname in sorted(fut.module_dict):
        mod = fut.module_dict[mname]
        path = source_of(mod) if source_of else _module_path(mod)
        if mname in GLOBALS_EXCLUDE or path is None or not path.exists():
            continue
        names = _acc_declared_names(path)
        if not names:
            continue
        gv = {k.lower(): g for k, g in mod.global_vars.items()}
        for n in sorted(names):
            g = gv.get(n)
            if g is None:
                skipped.append(f"global {mname}::{n}: declared but not a module variable of the analysis")
            elif getattr(g, "parameter", False) or str(g.type).lower() in type_dict:
                continue
            elif str(g.type).lower().startswith("character"):
                skipped.append(f"global {mname}::{n}: character (stays on the host)")
            elif getattr(g, "private", False) or not variable_is_public(g.name, mod):
                skipped.append(f"global {mname}::{n}: private (its module must update the device copy)")
            else:
                # storage test from the parsed declaration (the analysis flags some fixed-size
                # arrays with initializers as allocatable, e.g. timeinfoMod::days_per_mon)
                pspec = _spec(path)
                test = pspec.storage_test(g.name) if pspec else alloc_status(g)
                out.append(GlobalVar(g.name, mname, int(g.dim or 0), bool(test and test.startswith("allocated")),
                                     bool(test and test.startswith("associated"))))
    return out, skipped


@dataclass
class Guards:
    """Namelist conditions from the access analysis of elm_drv's call tree."""

    components: dict[int, str] = field(default_factory=dict)  # id(type component) -> Fortran condition
    cond_flags: dict[str, set[str]] = field(default_factory=dict)  # Fortran condition -> its flag names
    instances: dict[str, str] = field(default_factory=dict)  # instance (lower) -> Fortran condition
    flags: dict[str, str] = field(default_factory=dict)  # flag name -> declaring module
    notes: list[str] = field(default_factory=list)


def _history_components(fut) -> set[tuple[str, str]]:
    """(type, first component) of every history registration: hist_update_hbuf (outside the
    analysis) reads these on the device whatever the analyzed routines do."""
    out: set[tuple[str, str]] = set()
    inst_type = {i.name.lower(): t.type_name.lower() for t in fut.type_dict.values() for i in t.instances.values()}
    for sub in fut.subroutine_dict.values():
        for r in getattr(sub, "registrations", []) or []:
            if r.kind != "hist" or not r.field:
                continue
            first = r.field.split("%")[0].lower()
            var = sub.arguments.get(r.base) if hasattr(sub, "arguments") else None
            if var is not None:
                tname = str(var.type).lower().replace("class(", "").replace("type(", "").rstrip(")")
            else:
                tname = inst_type.get(r.base.lower(), "")
            if tname:
                out.add((tname, first))
    return out


def _fortran(cond) -> str:
    """A (minimized) condition as Fortran, without the redundant parentheses of to_fortran."""
    from spel.scripts.fortran_parser.boolen_expression import AllOf, AnyOf, Expectation

    match cond:
        case Expectation():
            return cond.to_fortran()
        case AllOf(items):
            if not items:
                return ".true."
            return " .and. ".join(f"({_fortran(i)})" if isinstance(i, AnyOf) else _fortran(i) for i in items)
        case AnyOf(items):
            if not items:
                return ".false."
            # factor out the literals every product shares: (a .and. x) .or. (b .and. x) -> x .and. (a .or. b)
            prods = [set(i.items) if isinstance(i, AllOf) else {i} for i in items]
            common = set.intersection(*prods)
            rests = [p - common for p in prods]
            if common and all(rests):
                order = [e for e in (items[0].items if isinstance(items[0], AllOf) else (items[0],)) if e in common]
                rest = AnyOf(tuple(next(iter(r)) if len(r) == 1 else AllOf(tuple(e for e in i.items if e in r))
                                   for r, i in zip(rests, items)))
                return " .and. ".join([_fortran(e) for e in order] + [f"({_fortran(rest)})"])
            return " .or. ".join(f"({_fortran(i)})" if isinstance(i, AllOf) and len(i.items) > 1 else _fortran(i)
                                 for i in items)
    return cond.to_fortran()


def access_guards(fut, type_dict) -> Guards:
    """
    Namelist guards for the copies and updates, from SPEL's access analysis
    (`nml_guards`: the namelist ifs around every access in elm_drv's tree, through the
    call sites). A type component accessed only under a namelist condition, by every
    instance that accesses it, is copied/updated only under that condition (shared by the
    instances: the per-type routines serve them all). An instance whose accessed fields are
    all gated is handled only under the or of their conditions (c13_col_cf: use_c13).
    Kept unguarded: history-registered components (the hbuf kernels read them), and
    conditions with string comparisons (no spel_streq in ELM).
    """
    from spel.scripts.fortran_parser.boolen_expression import AnyOf, minimize, simplify
    from spel.scripts.nml.analyze_namelist import check_subs_for_nml_guarded_vars, nml_guards

    g = Guards()
    drv = fut.subroutine_dict.get("elm_driver::elm_drv")
    if drv is None:
        g.notes.append("guards: elm_drv not in the analysis, no namelist guards")
        return g
    instance_dict = {i.name: t for t in fut.type_dict.values() for i in t.instances.values()}
    try:
        by_cond = check_subs_for_nml_guarded_vars([drv], instance_dict)
        raw = nml_guards(drv)
    except Exception as exc:
        g.notes.append(f"guards: namelist analysis failed ({exc}), no namelist guards")
        return g
    hist = _history_components(fut)
    owner = {id(c): t.type_name.lower() for t in type_dict.values() for c in t.components.values()}
    names: set[str] = set()

    def usable(cond) -> Optional[str]:
        cond = minimize(cond)
        text = _fortran(cond)
        if "spel_streq" in text or text in (".true.", ".false."):
            return None
        g.cond_flags[text] = {n.lower() for n in cond.variable_names()}
        return text

    for cond, comps in by_cond.items():
        text = usable(cond)
        if text is None:
            continue
        for c in comps:
            if (owner.get(id(c), ""), c.name.lower().split("%")[-1]) in hist:
                continue
            g.components[id(c)] = text
            names |= cond.variable_names()
    accessed = getattr(drv, "elmtype_access_by_ln", {})
    fields_of: dict[str, list[str]] = {}
    for v in accessed:
        fields_of.setdefault(v.split("%")[0].lower(), []).append(v)
    for inst, fields in fields_of.items():
        conds = [raw.get(v) for v in fields]
        if any(c is None for c in conds):
            continue
        cond = simplify(AnyOf(tuple(dict.fromkeys(conds))))
        text = usable(cond)
        if text is not None:
            g.instances[inst] = text
            names |= cond.variable_names()
    for n in sorted(names):
        mods = [m for m, mod in fut.module_dict.items() if n in {k.lower() for k in mod.global_vars}]
        if not mods:
            g.notes.append(f"guards: flag {n} has no declaring module; conditions using it dropped")
            continue
        g.flags[n] = "elm_varctl" if "elm_varctl" in mods else sorted(mods)[0]
    missing = set(names) - set(g.flags)
    if missing:
        g.components = {k: v for k, v in g.components.items() if not g.cond_flags.get(v, set()) & missing}
        g.instances = {k: v for k, v in g.instances.items() if not g.cond_flags.get(v, set()) & missing}
    return g


def _wrap_cond(head: str, cond: str, width: int = 120) -> list[str]:
    """`<head>if (cond) then`, continued at .and./.or. when long."""
    line = f"{head}if ({cond}) then"
    if len(line) <= width:
        return [line]
    pieces = re.split(r"(?=\s\.(?:and|or)\.\s)", cond)
    out, cur = [], f"{head}if ("
    for p in pieces:
        if len(cur) + len(p) > width - 2 and cur.strip() != "if (":
            out.append(cur.rstrip() + " &")
            cur = head + "     " + p.lstrip()
        else:
            cur += p
    out.append(cur + ") then")
    return out


def _gated(cond: Optional[str], stmts: list[str]) -> list[str]:
    """`stmts` (each indented by 4) under `if (cond)` when there is a condition."""
    if not cond or not stmts:
        return stmts
    return _wrap_cond("    ", cond) + ["   " + l for l in stmts] + ["    end if"]


# --- code generation -----------------------------------------------------------------


def _assoc(c) -> Optional[str]:
    """The Fortran test that `this%c` has storage, or None when it always has."""
    if id(c) in _STORAGE:  # from the parsed type definition
        return _STORAGE[id(c)]
    if getattr(c, "ptrscalar", False):
        return "associated"
    if getattr(c, "allocatable", False):
        return "allocated"
    return None


def _is_alias(c) -> bool:
    return bool(getattr(c, "ptrscalar", False) and getattr(c, "pointer", None))


def private_components(t, fut) -> set[str]:
    """Component names of type `t` that are private (cannot be named outside its module)."""
    mod = fut.module_dict.get(str(t.declaration).lower())
    spec = _spec(_module_path(mod)) if mod is not None else None
    info = spec.types.get(t.type_name.lower()) if spec else None
    return {c for c, acc in info.components.items() if acc == "private"} if info else set()


_STORAGE: dict[int, Optional[str]] = {}  # id(type component) -> storage test from the parsed definition


def _record_storage(type_dict, fut) -> None:
    """The allocated/associated test of every component, from the type definitions."""
    _STORAGE.clear()
    for t in type_dict.values():
        mod = fut.module_dict.get(str(t.declaration).lower())
        spec = _spec(_module_path(mod)) if mod is not None else None
        info = spec.types.get(t.type_name.lower()) if spec else None
        if info is None:
            continue
        for c in t.components.values():
            name = c.name.lower().split("%")[-1]
            if name in info.component_attrs:
                _STORAGE[id(c)] = info.storage_test(name)


def _components(t, fut=None):
    if fut is None:
        return list(t.components.values())
    hidden = private_components(t, fut)
    return [c for c in t.components.values() if c.name.lower() not in hidden]


# --- groups: one module per family of types --------------------------------------------

# (key, module, description), in `use` order: a group may only nest types of earlier groups.
GROUPS = [
    ("params", "DeepCopyParamsMod", "parameter and constant types (pft/soil-order constants, BGC parameters)"),
    ("subgrid", "DeepCopySubgridMod", "subgrid hierarchy, decomposition, filters and averaging views"),
    ("coupling", "DeepCopyCouplingMod", "coupler and external-model interface types (atm2lnd, lnd2atm, ...)"),
    ("biogeophys", "DeepCopyBiogeophysMod", "biogeophysics: energy and water state/flux types and the physics *_vars"),
    ("veg_nutrient", "DeepCopyVegNutrientMod", "vegetation carbon/nitrogen/phosphorus state and flux types, CN and crop state"),
    ("soil_nutrient", "DeepCopySoilNutrientMod", "column/gridcell carbon/nitrogen/phosphorus types, CH4, BGC interface"),
]
GROUP_MODULE = {k: m for k, m, _ in GROUPS}
GROUP_ORDER = {k: i for i, (k, _, _) in enumerate(GROUPS)}

_PARAMS = {"decomp_cascade_type", "soilordercon_type", "vegetation_properties_type", "urbanparams_type", "paramssharetype"}
_SUBGRID = {
    "clumpfilter", "clump_type", "decomp_type", "processor_type", "domain_type", "subgrid_ave_view",
    "col_connection_set_type",
}
_COUPLING = {
    "atm2lnd_type", "lnd2atm_type", "glc2lnd_type", "lnd2glc_type", "iac2lnd_type", "lnd2iac_type", "ocn2lnd_type",
    "glc_diagnostics_type", "topounit_atmospheric_flux", "topounit_atmospheric_state", "sedflux_type", "dust_type",
    "drydepvel_type", "chemstate_type",
}
_VEG_BGC = {"cnstate_type", "crop_type"}
_SOIL_BGC = {"ch4_type", "elm_interface_bgc_datatype", "elm_interface_data_type", "elm_interface_th_datatype", "conctransporttype"}
_NUTRIENT_WORDS = ("carbon", "nitrogen", "phosphorus", "nutrient", "mineral", "retrans", "uptake")


def type_group(tname: str) -> Optional[str]:
    """The group of a type by name, or None (then it follows the type that nests it)."""
    n = tname.lower()
    if n in _PARAMS or n.endswith("paramstype") or n.endswith("params_type"):
        return "params"
    if n in _SUBGRID or "physical_properties" in n:
        return "subgrid"
    if n in _COUPLING:
        return "coupling"
    if n in _VEG_BGC or (n.startswith("vegetation_") and any(w in n for w in _NUTRIENT_WORDS)):
        return "veg_nutrient"
    if n in _SOIL_BGC or any(w in n for w in _NUTRIENT_WORDS):
        return "soil_nutrient"
    return None


def assign_groups(p: "Plan", type_dict) -> dict[str, str]:
    """Group of every planned type. Types without a rule take the group of the first type
    (in instance order) that nests them, else biogeophys. Raises if a type nests a type of a
    later group (the modules would use each other)."""
    groups = {k: g for k in p.types if (g := type_group(k))}

    def nest(tname: str, parent_group: str, depth: int = 0) -> None:
        t = p.types.get(tname)
        if t is None or depth > MAX_DEPTH:
            return
        groups.setdefault(tname, parent_group)
        for c in t.components.values():
            ct = str(c.type).lower()
            if ct in p.types:
                nest(ct, groups[tname], depth + 1)

    for inst in p.instances:
        nest(inst.type_name.lower(), groups.get(inst.type_name.lower(), "biogeophys"))
    for k in p.types:
        groups.setdefault(k, "biogeophys")
    for k, t in p.types.items():
        for c in t.components.values():
            ct = str(c.type).lower()
            if ct in groups and GROUP_ORDER[groups[ct]] > GROUP_ORDER[groups[k]]:
                raise DeviceDataError(
                    f"type {k} ({groups[k]}) nests {ct} ({groups[ct]}): a group may only nest types of earlier groups"
                )
    return groups


# --- per-type routines and modules ------------------------------------------------------


def _wrap(line: str, width: int = 120) -> list[str]:
    """Split a long Fortran statement at `, ` with `&` continuations."""
    if len(line) <= width:
        return [line]
    indent = " " * (len(line) - len(line.lstrip()) + 5)
    out, cur = [], ""
    for piece in re.split(r"(?<=, )", line):
        if cur and len(cur) + len(piece) > width - 2:
            out.append(cur.rstrip() + " &")
            cur = indent + piece
        else:
            cur += piece
    out.append(cur)
    return out


def _guarded(test: Optional[str], path: str, body: list[str]) -> list[str]:
    """`body` (lines indented by 4) inside `if (test(path)) then ... end if` when there is a test."""
    if not test:
        return ["    " + b for b in body]
    return [f"    if ({test}({path})) then"] + ["       " + b for b in body] + ["    end if"]


def _elementwise(test: Optional[str], path: str, call: str) -> list[str]:
    """`call` (with `(i_)` already applied) over the elements of an array of types."""
    return _guarded(test, path, [f"do i_ = lbound({path}, 1), ubound({path}, 1)", f"   {call}", "end do"])


_COMP_GUARDS: dict[int, str] = {}  # set by generate(): id(type component) -> condition


def _comp(c, stmts: list[str]) -> list[str]:
    """One component's statements, under its namelist guard if it has one."""
    return _gated(_COMP_GUARDS.get(id(c)), stmts)


def _routine(kind: str, t, args: str, decls: list[str], body: list[str]) -> list[str]:
    name = f"{kind}_{t.type_name}"
    return (
        [f"  subroutine {name}({args})"]
        + decls
        + ["    integer :: i_", "#ifdef _OPENACC"]
        + (body or ["    continue"])
        + ["#endif", f"  end subroutine {name}", ""]
    )


def _copyin_routine(t, type_dict, fut) -> list[str]:
    """`deepcopy_<type>(this)`: enter data for the arrays, recursion into nested types."""
    body: list[str] = []
    for c in _components(t, fut):
        ctype = str(c.type).lower()
        path = f"this%{c.name}"
        test = _assoc(c)
        if ctype in INTRINSIC:
            if ctype.startswith("character") or _is_alias(c):
                continue  # strings stay on the host; aliases are attached
            if not test:
                # inline in the object (scalars and fixed-size arrays): an object created by `declare create` never got its
                # values, and `enter data copyin` of an inline member of a present object is
                # a no-op, so send the value (redundant, and cheap, for copied-in objects)
                body += _comp(c, [f"    !$acc update device({path})"])
            else:
                body += _comp(c, _guarded(test, path, [f"!$acc enter data copyin({path})"]))
        elif ctype in type_dict:
            if c.dim == 0:
                body += _comp(c, [f"    call deepcopy_{ctype}({path})"])
            else:
                body += _comp(c, _guarded(test, path, [
                    f"!$acc enter data copyin({path})",
                    f"do i_ = lbound({path}, 1), ubound({path}, 1)",
                    f"   call deepcopy_{ctype}({path}(i_))",
                    "end do",
                ]))
    return _routine("deepcopy", t, "this", [f"    type({t.type_name}), intent(inout), target :: this"], body)


def _attach_routine(t, type_dict, fut) -> list[str]:
    """`attach_<type>(this)`: attach the alias pointer components once their targets are present."""
    body: list[str] = []
    for c in _components(t, fut):
        ctype = str(c.type).lower()
        path = f"this%{c.name}"
        if ctype in INTRINSIC:
            if _is_alias(c) and c.dim > 0:
                body += _comp(c, _guarded("associated", path, [f"!$acc enter data attach({path})"]))
        elif ctype in type_dict and _has_alias(ctype, type_dict, fut):
            if c.dim == 0:
                body += _comp(c, [f"    call attach_{ctype}({path})"])
            else:
                body += _comp(c, _elementwise(_assoc(c), path, f"call attach_{ctype}({path}(i_))"))
    return _routine("attach", t, "this", [f"    type({t.type_name}), intent(inout), target :: this"], body)


_ALIAS_CACHE: dict[str, bool] = {}


def _has_alias(tname: str, type_dict, fut, depth: int = 0) -> bool:
    """Whether a type (or a type it nests) has an alias pointer component: only those get attach calls."""
    if tname in _ALIAS_CACHE:
        return _ALIAS_CACHE[tname]
    t = type_dict.get(tname)
    found = False
    if t is not None and depth <= MAX_DEPTH:
        for c in _components(t, fut):
            ct = str(c.type).lower()
            if (ct in INTRINSIC and _is_alias(c) and c.dim > 0) or (
                ct in type_dict and ct != tname and _has_alias(ct, type_dict, fut, depth + 1)
            ):
                found = True
                break
    _ALIAS_CACHE[tname] = found
    return found


def _update_routine(t, type_dict, fut) -> list[str]:
    """`update_<type>(this, to_device)`: update device/self of every component."""
    body: list[str] = []
    for c in _components(t, fut):
        ctype = str(c.type).lower()
        path = f"this%{c.name}"
        if ctype in INTRINSIC:
            if ctype.startswith("character") or _is_alias(c):
                continue
            body += _comp(c, _guarded(_assoc(c), path, [  # pointer/allocatable scalars too
                f"!$acc update device({path}) if(to_device)",
                f"!$acc update self({path}) if(.not. to_device)",
            ]))
        elif ctype in type_dict:
            if c.dim == 0:
                body += _comp(c, [f"    call update_{ctype}({path}, to_device)"])
            else:
                body += _comp(c, _elementwise(_assoc(c), path, f"call update_{ctype}({path}(i_), to_device)"))
    decls = [f"    type({t.type_name}), intent(inout), target :: this", "    logical, intent(in) :: to_device"]
    return _routine("update", t, "this, to_device", decls, body)


def _check_routine(t, type_dict, fut) -> list[str]:
    """`check_<type>(this, label, nmiss)`: count array components not present on the device."""
    body: list[str] = []
    for c in _components(t, fut):
        ctype = str(c.type).lower()
        path = f"this%{c.name}"
        test = _assoc(c)
        if ctype in INTRINSIC:
            if (c.dim == 0 and not test) or ctype.startswith("character"):
                continue  # inline scalars live in the (present) object
            # scalars need the byte count: acc_is_present(a) is the array form
            present = f"acc_is_present({path})" if c.dim > 0 else f"acc_is_present({path}, storage_size({path})/8)"
            stmt = f"call dd_miss({present}, label, '{c.name}', nmiss)"
            body += _comp(c, _wrap(f"    if ({test}({path})) {stmt}" if test else f"    {stmt}"))
        elif ctype in type_dict:
            if c.dim == 0:
                body += _comp(c, _wrap(f"    call check_{ctype}({path}, trim(label)//'%{c.name}', nmiss)"))
            else:
                body += _comp(c, _elementwise(test, path, f"call check_{ctype}({path}(i_), trim(label)//'%{c.name}(i)', nmiss)"))
    decls = [
        f"    type({t.type_name}), intent(in), target :: this",
        "    character(len=*), intent(in) :: label",
        "    integer, intent(inout) :: nmiss",
    ]
    return _routine("check", t, "this, label, nmiss", decls, body)


def _instance_call(inst: Instance, what: str, extra: str = "") -> list[str]:
    """Statements applying `<what>_<type>` to one instance (array instances element-wise)."""
    if inst.dim == 0:
        return [f"    call {what}_{inst.type_name}({inst.name}{extra})"]
    test = "associated" if inst.pointer else ("allocated" if inst.allocatable else None)
    return _elementwise(test, inst.name, f"call {what}_{inst.type_name}({inst.name}(i_){extra})")


def _header(name: str, purpose: list[str], srcroot: Path, commit: str) -> list[str]:
    return [
        f"module {name}",
        "  !",
        "  ! GENERATED by `spel device-data` from the elm_drv analysis of the sources",
        f"  ! ({srcroot}, {commit}). Do not edit; regenerate.",
        "  !",
    ] + [f"  ! {s}" for s in purpose] + [
        "  ! Everything is a no-op without _OPENACC.",
        "  !",
        "#ifdef _OPENACC",
        "  use openacc",
        "#endif",
    ]


def _group_module(key: str, mod_name: str, desc: str, types: list, instances: list[Instance], groups, p, type_dict, fut,
                  srcroot: Path, commit: str, guards: "Guards") -> str:
    """One group module: the per-type routines of its types and the group's instance entry points."""
    L = _header(mod_name, [
        f"Device deep copy of the {desc}.",
        f"Per type: deepcopy_<type> (enter data of every array, nested types recursively),",
        "attach_<type> (alias pointer components, after all targets are present),",
        "update_<type> (update device/self of every component) and check_<type> (presence).",
        f"Instances: {key}_copyin/_attach/_check/_update, called from ElmDeviceDataMod.",
    ], srcroot, commit)
    # the flags of every condition used in this module (instances and type components)
    conds = [guards.instances[i.name.lower()] for i in instances if i.name.lower() in guards.instances]
    conds += [guards.components[id(c)] for t in types for c in t.components.values() if id(c) in guards.components]
    used = {n for cnd in conds for n in guards.cond_flags.get(cnd, set()) if n in guards.flags}
    flag_mods: dict[str, list[str]] = {"elm_varctl": ["iulog"]}
    for n in sorted(used):
        flag_mods.setdefault(guards.flags[n], []).append(n)
    for m, ns in flag_mods.items():
        L += _wrap(f"  use {m}, only : {', '.join(ns)}")
    # types of earlier groups this group's types nest
    nested: dict[str, set[str]] = {}
    for t in types:
        for c in t.components.values():
            ct = str(c.type).lower()
            if ct in groups and groups[ct] != key and ct in type_dict:
                nested.setdefault(GROUP_MODULE[groups[ct]], set()).update(
                    {f"deepcopy_{ct}", f"update_{ct}", f"check_{ct}"} | ({f"attach_{ct}"} if _has_alias(ct, type_dict, fut) else set())
                )
    for _, m, _ in GROUPS:
        if m in nested:
            L += _wrap(f"  use {m}, only : {', '.join(sorted(nested[m]))}")
    imports: dict[str, list[str]] = {}  # module -> its types, then its instances
    for t in types:
        imports.setdefault(str(t.declaration).lower(), []).append(t.type_name)
    for inst in instances:
        imports.setdefault(inst.module, []).append(inst.name)
    for mod in sorted(imports):
        L += _wrap(f"  use {mod}, only : {', '.join(imports[mod])}")
    public = [f"{key}_copyin", f"{key}_attach", f"{key}_check", f"{key}_update"]
    for t in types:
        n = t.type_name.lower()
        public += [f"deepcopy_{n}", f"update_{n}", f"check_{n}"] + ([f"attach_{n}"] if _has_alias(n, type_dict, fut) else [])
    L += ["  implicit none", "  private"]
    L += _wrap(f"  public :: {', '.join(public)}")
    L += ["contains", ""]

    def entry(what: str, args: str, decls: list[str], call: str, extra: str, inst_filter=lambda i: True) -> list[str]:
        out = [f"  subroutine {key}_{what}({args})"] + decls + ["    integer :: i_", "#ifdef _OPENACC"]
        body: list[str] = []
        for inst in instances:
            if not inst_filter(inst):
                continue
            stmts: list[str] = []
            if what == "copyin":
                body.append(f"    ! {inst.module}::{inst.name} ({inst.type_name}{', declare' if inst.declared else ''})")
                if not inst.declared:
                    stmts.append(f"    !$acc enter data copyin({inst.name})")
            stmts += _instance_call(inst, call, extra(inst))
            # the instance is used only when its feature is switched on
            body += _gated(guards.instances.get(inst.name.lower()), stmts)
        return out + (body or ["    continue"]) + ["#endif", f"  end subroutine {key}_{what}", ""]

    L += entry("copyin", "", [], "deepcopy", lambda i: "")
    L += entry("attach", "", [], "attach", lambda i: "", lambda i: _has_alias(i.type_name.lower(), type_dict, fut))
    L += entry("check", "nmiss", ["    integer, intent(inout) :: nmiss"], "check", lambda i: f", '{i.name}', nmiss")
    L += entry("update", "to_device", ["    logical, intent(in) :: to_device"], "update", lambda i: ", to_device")
    for t in types:
        L += [f"  ! ---- {t.type_name} ({t.declaration})", ""]
        L += _copyin_routine(t, type_dict, fut)
        if _has_alias(t.type_name.lower(), type_dict, fut):
            L += _attach_routine(t, type_dict, fut)
        L += _update_routine(t, type_dict, fut)
        L += _check_routine(t, type_dict, fut)
    L += [
        "  subroutine dd_miss(present_, label, comp, nmiss)",
        "    ! count (and report the first 50 of) the arrays not present on the device",
        "    logical, intent(in) :: present_",
        "    character(len=*), intent(in) :: label, comp",
        "    integer, intent(inout) :: nmiss",
        "    if (present_) return",
        "    nmiss = nmiss + 1",
        "    if (nmiss <= 50) write(iulog,*) 'elm_device: not present: ', trim(label)//'%'//comp",
        "  end subroutine dd_miss",
        "",
        f"end module {mod_name}",
    ]
    return "\n".join(L) + "\n"


def _driver_module(group_keys: list[str], srcroot: Path, commit: str, spmd: bool = True,
                   filters: Optional[list[Instance]] = None, groups: Optional[dict[str, str]] = None) -> str:
    """
    filters: the clumpfilter instances (elm_device_filters_update); groups: type -> group key.
    """
    filters = filters or []
    L = _header(MODULE_NAME, [
        "Device data layer for full-model OpenACC runs: device initialization, the",
        "one-time deep copy of every type instance to the device (first time step),",
        "and the host/device transitions of the driver. The per-type routines are in",
        "the DeepCopy*Mod group modules.",
        "",
        "Transitions: by default (targeted) elm_device_update only switches where the",
        "compute constructs run (elm_device_ready) and the driver's acc-sync updates move",
        "the data its host segments use; SPEL_DEVICE_FULLSYNC=1 in the environment makes",
        "every transition copy the whole state instead (the reference for A/B runs).",
    ], srcroot, commit)
    L += ["  use elm_varctl, only : iulog, elm_device_ready"]
    if spmd:
        L += ["  use perf_mod,   only : t_startf, t_stopf"]
    if spmd:
        L += ["  use spmdMod,    only : masterproc, iam"]
    for k in group_keys:
        L.append(f"  use {GROUP_MODULE[k]}, only : {k}_copyin, {k}_attach, {k}_check, {k}_update")
    L.append(f"  use {GLOBALS_MODULE}, only : globals_copyin, globals_update")
    for mod in sorted({f.module for f in filters}):
        L.append(f"  use {mod}, only : {', '.join(f.name for f in filters if f.module == mod)}")
    if filters:
        L.append(f"  use {GROUP_MODULE[groups[filters[0].type_name.lower()]]}, only : update_{filters[0].type_name.lower()}")
    L += [
        "  implicit none",
        "  private",
        "  public :: elm_device_init, elm_device_copyin, elm_device_update, elm_device_resident",
        "  public :: elm_device_fullsync, elm_device_pull_all, elm_device_filters_update",
        "  logical, save :: elm_device_resident = .false.  ! the state lives on the device (public above)",
        "  ! every transition copies the whole state: until the first host -> device transition of",
        "  ! the run (the initialization writes host data after elm_device_copyin, and pulls what",
        "  ! its kernels compute), then as requested (SPEL_DEVICE_FULLSYNC=1, read in elm_device_init)",
        "  logical, save :: elm_device_fullsync = .true.",
        "  logical, save :: fullsync_requested = .false.",
        "  ! debugging targeted transitions: the first full_until transitions copy the whole state",
        "  ! (SPEL_DEVICE_FULL_UNTIL=K; bisect K for a missing sync), SPEL_DEVICE_TRACE=1 prints them",
        "  integer, save :: full_until = 0, ntransitions = 0",
        "  logical, save :: trace = .false.",
    ]
    if not spmd:  # unit tests: one process
        L += ["  logical, parameter :: masterproc = .true.", "  integer, parameter :: iam = 0"]
    L += [
        "contains",
        "",
        "  subroutine elm_device_init()",
        "    ! Select the device and initialize the runtime; called first in initialize1.",
        "    ! Data in `declare` directives was placed on the default device when the",
        "    ! program started, so each rank must see exactly one device (srun",
        "    ! --gpus-per-task=1); with several visible devices the local rank picks one",
        "    ! and the declared module data of the other device is lost: warn loudly.",
        "#ifdef _OPENACC",
        "    integer :: ndev, dev, localid, ios",
        "    character(len=32) :: s",
        "    call get_environment_variable('SPEL_DEVICE_FULLSYNC', s, status=ios)",
        "    fullsync_requested = ios == 0 .and. (trim(s) == '1' .or. trim(s) == 'true' .or. trim(s) == 'TRUE')",
        "    call get_environment_variable('SPEL_DEVICE_FULL_UNTIL', s, status=ios)",
        "    if (ios == 0) read(s, *, iostat=ios) full_until",
        "    call get_environment_variable('SPEL_DEVICE_TRACE', s, status=ios)",
        "    trace = ios == 0 .and. trim(s) == '1'",
        "    if (masterproc .and. full_until > 0) write(iulog,*) 'elm_device_init: whole state for the first', &",
        "         full_until, 'transitions'",
        "    if (masterproc) write(iulog,*) 'elm_device_init: host/device transitions ', &",
        "         merge('whole state', 'targeted   ', fullsync_requested), ' after the first step start'",
        "    ndev = acc_get_num_devices(acc_device_nvidia)",
        "    if (ndev > 1) then",
        "       call get_environment_variable('SLURM_LOCALID', s, status=ios)",
        "       localid = 0",
        "       if (ios == 0) read(s, *, iostat=ios) localid",
        "       dev = mod(max(localid, 0), ndev)",
        "       call acc_set_device_num(dev, acc_device_nvidia)",
        "       write(iulog,*) 'elm_device_init: WARNING rank', iam, 'sees', ndev, 'devices; picked', dev, &",
        "            ' - module data in acc declare directives may be on another device; bind one GPU per rank'",
        "    end if",
        "    call acc_init(acc_device_nvidia)",
        "    if (masterproc .or. ndev > 1) write(iulog,*) 'elm_device_init: rank', iam, 'device', &",
        "         acc_get_device_num(acc_device_nvidia), 'of', ndev",
        "#endif",
        "  end subroutine elm_device_init",
        "",
        "  subroutine elm_device_copyin()",
        "    ! One-time deep copy of every type instance, then the alias attachments",
        "    ! (their targets may be in other groups) and the presence check.",
        "#ifdef _OPENACC",
        "    integer :: nmiss",
        "    if (elm_device_resident) return",
    ]
    L += ["    call globals_copyin()"]
    L += [f"    call {k}_copyin()" for k in group_keys]
    L += [f"    call {k}_attach()" for k in group_keys]
    L += ["    nmiss = 0"] + [f"    call {k}_check(nmiss)" for k in group_keys]
    L += [
        "    if (nmiss > 0) then",
        "       write(iulog,*) 'elm_device_copyin: rank', iam, ':', nmiss, 'arrays NOT present on the device'",
        "    else if (masterproc) then",
        "       write(iulog,*) 'elm_device_copyin: every array of every instance is present on the device'",
        "    end if",
        "    elm_device_resident = .true.",
        "    ! from now on the compute constructs (if(elm_device_ready)) run on the device",
        "    elm_device_ready = .true.",
        "    !$acc update device(elm_device_ready)",
        "#endif",
        "  end subroutine elm_device_copyin",
        "",
        "  subroutine elm_device_update(to_device, line)",
        "    ! A transition of the driver: the next code runs on the device (to_device) or on the",
        "    ! host. Targeted: switch elm_device_ready only (the driver's acc-sync updates move the",
        "    ! data). Whole state (elm_device_fullsync): update device / update self of everything.",
        "    logical, intent(in) :: to_device",
        "    integer, intent(in), optional :: line  ! the caller's __LINE__ (SPEL_DEVICE_TRACE)",
        "#ifdef _OPENACC",
        "    integer :: ln",
        "    if (.not. elm_device_resident) return",
        "    ! the state already lives there (a transition inside a branch that was not taken, or",
        "    ! two in a row): nothing to copy",
        "    if (elm_device_ready .eqv. to_device) return",
        "    ntransitions = ntransitions + 1",
        "    ln = -1",
        "    if (present(line)) ln = line",
        "    if (trace .and. masterproc) write(iulog,'(a,i6,a,i6,a,l2,a,l2)') ' elm_device_update: transition', &",
        "         ntransitions, ' line', ln, ' to_device', to_device, ' whole', &",
        "         elm_device_fullsync .or. ntransitions <= full_until",
        "    if (elm_device_fullsync .or. ntransitions <= full_until) then",
    ]
    if spmd:
        L += ["       call t_startf(merge('device_push_whole', 'device_pull_whole', to_device))"]
    L += ["       call globals_update(to_device)"]
    L += [f"       call {k}_update(to_device)" for k in group_keys]
    if spmd:
        L += ["       call t_stopf(merge('device_push_whole', 'device_pull_whole', to_device))"]
    L += [
        "       ! host and device agree after the first whole push: targeted from here on",
        "       if (to_device) elm_device_fullsync = fullsync_requested",
        "    end if",
    ]
    L += [
        "    ! the state now lives where the next code runs: a host segment of elm_drv (to_device",
        "    ! false) runs its compute constructs (if(elm_device_ready)) on the host too, so a routine",
        "    ! the transitions treat as host code cannot leave results on the device that the next",
        "    ! host -> device update overwrites",
        "    elm_device_ready = to_device",
        "    !$acc update device(elm_device_ready)",
    ]
    L += ["#endif", "  end subroutine elm_device_update", ""]
    L += [
        "  subroutine elm_device_pull_all()",
        "    ! device -> host of the whole state, for host code the analysis does not see into",
        "    ! (restFile_write reads the state through its registered pointers). Nothing to do",
        "    ! with whole-state transitions: the preceding elm_device_update already copied it.",
        "#ifdef _OPENACC",
        "    if (.not. elm_device_resident .or. elm_device_fullsync) return",
    ]
    if spmd:
        L += ["    call t_startf('device_pull_all')"]
    L += ["    call globals_update(.false.)"]
    L += [f"    call {k}_update(.false.)" for k in group_keys]
    if spmd:
        L += ["    call t_stopf('device_pull_all')"]
    L += ["#endif", "  end subroutine elm_device_pull_all", ""]
    L += [
        "  subroutine elm_device_filters_update(to_device)",
        "    ! the filters of every clump (arrays of derived types, which acc-sync cannot",
        "    ! update member by member): host code that builds or reads them",
        "    logical, intent(in) :: to_device",
        "#ifdef _OPENACC",
        "    integer :: i_",
        "    if (.not. elm_device_resident .or. elm_device_fullsync) return",
    ]
    for f in filters:
        upd = f"update_{f.type_name.lower()}"
        if f.dim > 0:
            L += [f"    do i_ = lbound({f.name}, 1), ubound({f.name}, 1)",
                  f"       call {upd}({f.name}(i_), to_device)", "    end do"]
        else:
            L += [f"    call {upd}({f.name}, to_device)"]
    L += ["#endif", "  end subroutine elm_device_filters_update", "", f"end module {MODULE_NAME}"]
    return "\n".join(L) + "\n"


def _globals_module(gvars: list[GlobalVar], srcroot: Path, commit: str) -> str:
    L = _header(GLOBALS_MODULE, [
        "Device copies of the module variables named in `!$acc declare` directives:",
        "globals_copyin uploads them once (after initialization and restart reads),",
        "globals_update moves them with the rest of the state at the host/device",
        "transitions (the per-step time info changes every step).",
    ], srcroot, commit)
    # two modules may declare variables of the same name (CanopyStateType and SoilMoistStressMod
    # both have `perchroot`): those are imported under a module-qualified local name
    counts: dict[str, int] = {}
    for g in gvars:
        counts[g.name] = counts.get(g.name, 0) + 1
    local = {(g.module, g.name): (f"{g.name}_{g.module}" if counts[g.name] > 1 else g.name) for g in gvars}
    by_mod: dict[str, list[str]] = {}
    for g in gvars:
        ln = local[(g.module, g.name)]
        by_mod.setdefault(g.module, []).append(g.name if ln == g.name else f"{ln} => {g.name}")
    for mod in sorted(by_mod):
        L += _wrap(f"  use {mod}, only : {', '.join(by_mod[mod])}")
    L += ["  implicit none", "  private", "  public :: globals_copyin, globals_update", "contains", ""]

    def guard(g: GlobalVar) -> Optional[str]:
        return "associated" if g.pointer else ("allocated" if g.allocatable else None)

    def tail(g: GlobalVar) -> str:
        # a module pointer has a device copy only when it was allocated (not when `=>`ed)
        return " if_present" if g.pointer else ""

    copy: list[str] = []
    for g in gvars:
        name = local[(g.module, g.name)]
        copy += _guarded(guard(g), name, [f"!$acc update device({name}){tail(g)}"])
    L += ["  subroutine globals_copyin()", "#ifdef _OPENACC"] + (copy or ["    continue"])
    L += ["#endif", "  end subroutine globals_copyin", ""]
    L += ["  subroutine globals_update(to_device)", "    logical, intent(in) :: to_device", "#ifdef _OPENACC"]
    out: list[str] = []
    for g in gvars:
        name = local[(g.module, g.name)]
        out += _guarded(guard(g), name, [
            f"!$acc update device({name}) if(to_device){tail(g)}",
            f"!$acc update self({name}) if(.not. to_device){tail(g)}",
        ])
    L += (out or ["    continue"]) + ["#endif", "  end subroutine globals_update", "", f"end module {GLOBALS_MODULE}"]
    return "\n".join(L) + "\n"


def generate(fut, srcroot: Path, only_instances: Optional[set[str]] = None, source_of=None,
             guards: Optional["Guards"] = None, spmd: bool = True) -> tuple[dict[str, str], Plan]:
    """The module texts (file name -> text) and the plan they were generated from."""
    p = plan(fut, only_instances, source_of)
    # components of private types are opaque to the generated code (no `use` possible)
    type_dict = {k.lower(): v for k, v in fut.type_dict.items() if k.lower() not in set(p.notes)}
    _ALIAS_CACHE.clear()
    # allocated/associated tests of the components from the type definitions (the analysis's
    # Variables do not record `allocatable`): without them allocatable components would be
    # treated as inline and only updated, never copied in
    _record_storage(type_dict, fut)
    try:
        commit = subprocess.run(
            ["git", "-C", str(srcroot), "rev-parse", "--short=10", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
    except Exception:
        commit = "?"
    groups = assign_groups(p, type_dict)
    if guards is None:
        guards = access_guards(fut, type_dict)
        # full model: no per-component guards. They come from elm_drv's analysis, but code
        # outside it uses components unconditionally too (the initialization's export state,
        # lnd2atm_minimal, reads lnd2atm albd_grc, which elm_drv reads only with
        # use_finetop_rad); a component guard only saves device memory
        guards.notes.append(f"component guards dropped for the full model: {len(guards.components)}")
        guards.components = {}
    _COMP_GUARDS.clear()
    _COMP_GUARDS.update(guards.components)
    p.notes_guards = {i.name: guards.instances[i.name.lower()] for i in p.instances if i.name.lower() in guards.instances}
    p.ncomp_guards = len(guards.components)
    p.skipped += guards.notes
    files: dict[str, str] = {}
    keys: list[str] = []
    for key, mod_name, desc in GROUPS:
        types = [p.types[k] for k in sorted(p.types) if groups[k] == key]
        if not types:
            continue
        instances = [i for i in p.instances if groups[i.type_name.lower()] == key]
        files[f"{mod_name}.F90"] = _group_module(key, mod_name, desc, types, instances, groups, p, type_dict, fut,
                                                 srcroot, commit, guards)
        keys.append(key)
    gvars, gskipped = plan_globals(fut, type_dict, source_of)
    p.skipped += gskipped
    p.globals = gvars
    files[f"{GLOBALS_MODULE}.F90"] = _globals_module(gvars, srcroot, commit)
    filters = [i for i in p.instances if i.type_name.lower() == "clumpfilter"]
    files[f"{MODULE_NAME}.F90"] = _driver_module(keys, srcroot, commit, spmd, filters, groups)
    p.groups = {k: [t for t in sorted(p.types) if groups[t] == k] for k in keys}
    return files, p


# --- unit tests ------------------------------------------------------------------------


def unit_test_guards(unit_test, type_dict) -> Guards:
    """The unit test's own namelist guards (`guarded_usage_dict`, as ReadWriteMod uses them)."""
    from spel.scripts.fortran_parser.boolen_expression import minimize

    g = Guards()
    for cond, variables in getattr(unit_test, "guarded_usage_dict", {}).items():
        text = _fortran(minimize(cond))
        if "spel_streq" in text or text in (".true.", ".false."):
            continue
        g.cond_flags[text] = {n.lower() for n in cond.variable_names()}
        for v in variables:
            g.components[id(v)] = text
        for n in cond.variable_names():
            var = unit_test.active_global_vars.get(n)
            if var is not None:
                g.flags[n.lower()] = str(var.declaration).lower()
    return g


def generate_for_unit_test(unit_test) -> list[str]:
    """
    Write the device data layer for a unit-test case (`spel create --device-data`): the
    same generator as for the full model, restricted to the case's instances, reading the
    `!$acc declare` directives of the case's copies of the modules, with the case's
    namelist guards. Returns the file names, in compile order.
    """
    case_dir = Path(unit_test.case_dir)

    def source_of(mod):
        path = case_dir / Path(mod.filepath).name
        return path if path.exists() else None

    # every instance the case's copy of its module declares: the captured ones, the ones
    # init hooks build at run time (transport_ptr_list), the filters the harness builds
    only: set[str] = set()
    for dtype in unit_test.type_dict.values():
        for inst in dtype.instances.values():
            mod = unit_test.module_dict.get(str(inst.declaration or "").lower())
            spec = _spec(source_of(mod)) if mod is not None else None
            if spec is not None and inst.name.lower() in spec.variables:
                only.add(inst.name.lower())

    type_dict = {k.lower(): v for k, v in unit_test.type_dict.items()}
    from spel.scripts.config import E3SM_SRCROOT

    import os

    # SPEL_DEVICE_NO_GUARDS=1: copy every component whatever the namelist (a debugging aid)
    guards = Guards() if os.environ.get("SPEL_DEVICE_NO_GUARDS") else unit_test_guards(unit_test, type_dict)
    texts, p = generate(unit_test, Path(E3SM_SRCROOT), only_instances=only, source_of=source_of,
                        guards=guards, spmd=False)
    order = [f"{m}.F90" for _, m, _ in GROUPS] + [f"{GLOBALS_MODULE}.F90", f"{MODULE_NAME}.F90"]
    written = []
    for name in order:
        if name in texts:
            (case_dir / name).write_text(texts[name])
            written.append(name)
    unit_test.logger.info(
        f"device data layer: {len(p.instances)} instances of {len(p.types)} types, {len(p.globals)} globals"
    )
    return written


# --- applying to the sources -----------------------------------------------------------


@dataclass
class Target:
    """
    A statement located in the parse tree (spel.scripts.fortran_parser.module_spec):
    `use` of module `name`, a `call` of routine `name` (with first string argument
    `arg`), or the first `if` construct whose body calls `name`; in `routine`, or in the
    module's specification part when `routine` is None.
    """

    kind: str  # "use" | "call" | "if_calling"
    name: str
    routine: Optional[str] = None
    arg: Optional[str] = None

    def __str__(self) -> str:
        where = f" in {self.routine}" if self.routine else ""
        what = {"use": "use", "call": "call", "if_calling": "if-block calling"}[self.kind]
        return f"{what} {self.name}{f'({self.arg!r})' if self.arg else ''}{where}"


@dataclass
class Edit:
    path: Path
    target: Optional[Target]  # the statement to insert before/after (its first match)
    lines: list[str]
    after: bool = False
    at_line: Optional[int] = None  # insert before this 0-based line instead (a recorded call site)


def _locate(path: Path, target: Target, after: bool) -> int:
    """The 0-based line at which to insert before/after the target statement."""
    from spel.scripts.fortran_parser import spel_ast as A
    from spel.scripts.fortran_parser.module_spec import (
        callee_name, contains_call, first_string_arg, parse_modules, statements_with_next,
    )

    mods = parse_modules(path)
    if not mods:
        raise DeviceDataError(f"{path.name}: no module")
    mod = mods[0]
    if target.routine is None:
        block, end = mod.spec, mod.contains_ln if mod.contains_ln is not None else mod.end_ln
    else:
        sub = next((p for p in mod.subprograms if p.name.lower() == target.routine), None)
        if sub is None:
            raise DeviceDataError(f"{path.name}: no routine {target.routine}")
        block, end = sub.body, sub.end_ln
    for st, next_ln in statements_with_next(block, end):
        if target.kind == "use":
            hit = isinstance(st, A.UseStatement) and str(st.module).lower() == target.name
        elif target.kind == "call":
            hit = callee_name(st) == target.name and (target.arg is None or first_string_arg(st) == target.arg)
        else:
            hit = isinstance(st, A.IfConstruct) and contains_call(st, target.name)
        if hit:
            return next_ln if after else st.lineno
    raise DeviceDataError(f"{path.name}: no {target}")


# routines that run kernels the analysis does not see: their modules are excluded from it
# (hist_update_hbuf), or the kernels are in such a module (UpdateAccVars: accumulMod's
# update/extract, accumComputeMod)
DEVICE_EXTRA = {"hist_update_hbuf", "updateaccvars"}
# calls that touch no model state: they never start a host or a device section
NEUTRAL = {
    "t_startf", "t_stopf", "get_clump_bounds", "get_proc_bounds", "get_proc_clumps", "shr_sys_flush",
    "endrun", "get_curr_date", "get_nstep", "is_end_curr_day", "is_end_curr_month", "get_step_size",
    "elm_device_update", "elm_device_copyin", "mpi_barrier", "shr_sys_abort",
    "elm_device_pull_all", "elm_device_filters_update",
}
# device routines run as host segments anyway: their kernels are not ported through
# (dynSubgrid is left out of the first GPU port; its ZeroDWT/conservation kernels run on the host).
# The UpdateAccVars run on the device: accumulMod's update/extract are kernels (accumComputeMod).
HOST_SEGMENTS = {"dynsubgrid_driver"}


def call_kind(base: str, device: set[str]) -> str:
    """'device' or 'host' by name only (device_routines), for calls call_kinds cannot resolve."""
    return "device" if base in device and base not in HOST_SEGMENTS else "host"


def call_kinds(fut) -> list[tuple[int, str, str]]:
    """
    (line, callee name, 'device' | 'host') of every elm_drv call, in source order. A call runs
    on the device when the routine it resolves to (scope, call descriptions, or the binding of
    a type-bound call: `col_cs%Summary` -> col_cs_summary) has a compute construct in its call
    tree. By name, type-bound calls never matched their implementations (the summaries, the
    balance checks ran as host segments). HOST_SEGMENTS and DEVICE_EXTRA override.
    """
    from spel.scripts.driver_callsites import type_bound_callee
    from spel.scripts.record_access import AccessMapper
    from spel.transforms.acc_apply import _Facts

    drv = fut.subroutine_dict["elm_driver::elm_drv"]
    facts = _Facts(fut)
    mapper = AccessMapper(fut.subroutine_dict)
    memo: dict[str, bool] = {}

    def resolve(sub, event):
        return mapper.callee(sub, sub.record, event) or type_bound_callee(
            event, fut.module_dict, fut.subroutine_dict, fut.type_dict)

    def kernels(sid: str, stack: tuple = ()) -> bool:
        if sid in memo:
            return memo[sid]
        sub = fut.subroutine_dict.get(sid)
        if sub is None or sub.record is None or sid in stack:
            return False
        found = facts.has_compute(sub) or sid.split("::")[-1] in DEVICE_EXTRA
        for e in [] if found else sub.record.calls:
            callee = resolve(sub, e)
            if callee is not None and kernels(callee.id, stack + (sid,)):
                found = True
                break
        memo[sid] = found
        return found

    by_name = device_routines(fut)
    out = []
    for e in sorted(drv.record.calls, key=lambda e: e.ln):
        base = e.name.split("%")[-1].lower()
        callee = resolve(drv, e)
        if base in HOST_SEGMENTS:
            kind = "host"
        elif base in DEVICE_EXTRA:
            kind = "device"
        elif callee is not None:
            kind = "device" if kernels(callee.id) else "host"
        else:
            kind = call_kind(base, by_name)
        out.append((e.ln, base, kind))
    return out


def device_routines(fut) -> set[str]:
    """
    Basenames of the analyzed routines whose call tree contains an OpenACC compute
    construct (`parallel`, `kernels`, `serial`): calling one runs device kernels.
    The directives are comments to the parser; `acc_directives` parses them.
    """
    from spel.transforms.acc_directives import find_directives

    subs = fut.subroutine_dict
    has_region: set[str] = set()
    callees: dict[str, set[str]] = {}
    text_cache: dict[str, list[str]] = {}
    for sid, sub in subs.items():
        base = sid.split("::")[-1].lower()
        try:
            if sub.filepath not in text_cache:
                text_cache[sub.filepath] = Path(sub.filepath).read_text(errors="ignore").split("\n")
            if any(d.is_compute_region or d.is_compute_loop
                   for d in find_directives(text_cache[sub.filepath], sub.startline, sub.endline)):
                has_region.add(base)
        except Exception:
            pass
        rec = getattr(sub, "record", None)
        if rec is not None:
            callees.setdefault(base, set()).update(str(c.name).split("%")[-1].lower() for c in rec.calls)
    device = set(has_region) | DEVICE_EXTRA
    changed = True
    while changed:
        changed = False
        for base, cs in callees.items():
            if base not in device and cs & device:
                device.add(base)
                changed = True
    return device


def _transition_edits(fut, drv: Path) -> tuple[list[Edit], list[str]]:
    """
    Whole-state syncs before every call of elm_drv: `update self` before each host routine,
    `update device` before each device routine. elm_device_update does nothing when the state
    already lives there, so only the actual host/device transitions copy, and a transition
    inside a branch that is not taken cannot leave the next routine with the wrong copy
    (calls in if blocks: the budgets, nstep > 0). Brute force; the refinement is per-routine
    variable lists from the access analysis.
    """
    if fut.subroutine_dict.get("elm_driver::elm_drv") is None:
        raise DeviceDataError("elm_driver::elm_drv not in the analysis")
    from spel.scripts.fortran_parser.module_spec import callee_name, parse_modules, statements_with_next

    # the calls and their lines come from the driver file being edited, not from the analysis:
    # its line numbers go stale as soon as the driver is edited after `spel analyze`, and
    # transitions placed by them land before the wrong calls
    mod = parse_modules(drv)[0]
    body = next(p for p in mod.subprograms if p.name.lower() == "elm_drv")
    calls = sorted({(st.lineno, name) for st, _ in statements_with_next(body.body, body.end_ln)
                    if (name := callee_name(st)) is not None})
    # the analysis's calls in source order give the kinds; matched one by one to the file's
    # calls (the analysis read the driver with its transitions, the file may be stripped)
    # calls into modules the analysis excludes (ndep_interp, hist_*, restFile_write) are not in
    # it: they keep the kind by name (DEVICE_EXTRA for hist_update_hbuf)
    kinds = [(b, k) for _, b, k in call_kinds(fut) if b not in NEUTRAL]
    by_name = device_routines(fut)
    placed: list[tuple[int, str, str]] = []
    i = 0
    for ln, base in calls:
        if base in NEUTRAL:
            continue
        if i < len(kinds) and kinds[i][0] == base:
            placed.append((ln, base, kinds[i][1]))
            i += 1
        else:
            placed.append((ln, base, "device" if base in DEVICE_EXTRA else call_kind(base, by_name)))
    if i != len(kinds):
        raise DeviceDataError(f"elm_drv's calls in the driver file differ from the analysis at {kinds[i][0]}: "
                              "run `spel analyze`")
    edits: list[Edit] = []
    report: list[str] = []
    state = "host"
    for ln, base, kind in placed:
        to_device = kind == "device"
        edits.append(
            Edit(
                drv,
                None,
                [
                    "#ifdef _OPENACC",
                    f"    call elm_device_update(to_device={'.true.' if to_device else '.false.'}, line=__LINE__)  {TAG}: {'host -> device' if to_device else 'device -> host'} before {base}",
                    "#endif",
                ],
                at_line=ln,
            )
        )
        report.append(f"  {ln + 1}: {'update device' if to_device else 'update self  '} before {base}")
        state = kind
    return edits, report


def _driver_edits(fut, src: Path) -> tuple[list[Edit], list[str]]:
    drv = src / "main" / "elm_driver.F90"
    ini = src / "main" / "elm_initializeMod.F90"
    use_line = f"  use {MODULE_NAME}, only : elm_device_init, elm_device_copyin, elm_device_update, elm_device_resident  {TAG}"
    edits = [
        Edit(ini, Target("use", "shr_kind_mod"),
             [f"  use {MODULE_NAME}, only : elm_device_init, elm_device_copyin, elm_device_update  {TAG}"]),
        Edit(ini, Target("call", "t_startf", "initialize1", "elm_init1"),
             ["#ifdef _OPENACC", f"    call elm_device_init()  {TAG}", "#endif"], after=True),
        # the end of initialize2 runs kernels (lnd2atm_minimal's subgrid averages, lnd2glc):
        # copy the state there, after every restart read, and hand it back to the host after
        Edit(
            ini,
            Target("if_calling", "lnd2atm_minimal", "initialize2"),
            [
                "#ifdef _OPENACC",
                f"    {TAG}: the whole state to the device (after the restart reads), before the",
                "    ! kernels that compute the initial export state",
                "    call t_startf('device_copyin')",
                "    call elm_device_copyin()",
                "    call t_stopf('device_copyin')",
                "#endif",
            ],
        ),
        Edit(
            ini,
            Target("if_calling", "update_lnd2glc", "initialize2"),
            [
                "#ifdef _OPENACC",
                f"    call elm_device_update(to_device=.false.)  {TAG}: the step starts on the host",
                "#endif",
            ],
            after=True,
        ),
        Edit(drv, Target("use", "shr_kind_mod"), [use_line]),
        Edit(
            drv,
            Target("call", "get_proc_bounds", "elm_drv"),
            [
                "#ifdef _OPENACC",
                f"    {TAG}: the whole state to the device if initialization did not (fallback)",
                "    if (.not. elm_device_resident) then",
                "       call t_startf('device_copyin')",
                "       call elm_device_copyin()",
                "       call t_stopf('device_copyin')",
                "    end if",
                "#endif",
            ],
            after=True,
        ),
    ]
    # subgrid weights changed (first step, transient runs): the filters and the subgrid data
    # were rebuilt on the host inside a device section of elm_drv; refresh the device copies
    rew = src / "main" / "reweightMod.F90"
    edits += [
        Edit(
            rew,
            Target("use", "filtermod", "reweight_wrapup"),
            [
                "#ifdef _OPENACC",
                f"    use openacc           , only : acc_is_present  {TAG}",
                "    use filterMod         , only : filter",
                "    use DeepCopySubgridMod, only : subgrid_update",
                "#endif",
            ],
            after=True,
        ),
        Edit(
            rew,
            Target("call", "setfilters", "reweight_wrapup"),
            [
                "#ifdef _OPENACC",
                f"    {TAG}: the filters are built on the host; once the state is on the device",
                "    ! (after elm_device_copyin) send the rebuilt filters and subgrid data there",
                "    if (acc_is_present(filter)) call subgrid_update(to_device=.true.)",
                "#endif",
            ],
            after=True,
        ),
    ]
    trans, report = _transition_edits(fut, drv)
    return edits + trans, report


def apply(fut, srcroot: Path, out_dir: Optional[Path] = None, dry_run: bool = False) -> tuple[Plan, list[str]]:
    """
    Write the modules to `data_types/` and insert the calls into elm_initializeMod and
    elm_driver. With `out_dir`, the edited files and the modules go there (a case's
    SourceMods/src.elm); otherwise the sources are edited in place. Refuses sources
    that already carry the calls (revert the two driver files first); generated modules
    from an earlier run are overwritten.
    """
    src = srcroot / "components" / "elm" / "src"
    texts, p = generate(fut, srcroot)
    edits, report = _driver_edits(fut, src)
    files: dict[Path, list[str]] = {}
    for e in edits:
        if e.path not in files:
            if not e.path.exists():
                raise DeviceDataError(f"{e.path}: not found")
            content = e.path.read_text()
            if TAG in content or f"use {MODULE_NAME}" in content:
                raise DeviceDataError(f"{e.path}: already carries the device data calls")
            files[e.path] = content.split("\n")
    # resolve every insertion point on the original text, then insert bottom-up
    placed: list[tuple[Path, int, list[str], str]] = []
    for e in edits:
        if e.at_line is not None:
            i = e.at_line
            where = f"line {i + 1}"
        else:
            i = _locate(e.path, e.target, e.after)
            where = f"{'after' if e.after else 'before'} {e.target} (line {i + 1})"
        placed.append((e.path, i, e.lines, where))
    for path, i, new, where in sorted(placed, key=lambda t: -t[1]):
        files[path][i:i] = new
        report.append(f"{path.name}: {where}")
    dest_dir = out_dir or src / "data_types"
    for name, text in texts.items():
        dest = dest_dir / name
        if dest.exists() and out_dir is None and "GENERATED by `spel device-data`" not in dest.read_text():
            raise DeviceDataError(f"{dest} exists and was not generated here")
    stale = src / "main" / f"{MODULE_NAME}.F90"  # first version's location
    if out_dir is None and stale.exists() and "GENERATED by `spel device-data`" in stale.read_text():
        report.append(f"removed {stale} (the modules now live in data_types/)")
        if not dry_run:
            stale.unlink()
    if not dry_run:
        dest_dir.mkdir(parents=True, exist_ok=True)
        for name, text in texts.items():
            (dest_dir / name).write_text(text)
        for path, lines in files.items():
            dest = (out_dir / path.name) if out_dir else path
            dest.write_text("\n".join(lines))
    head = [f"{len(p.instances)} instances of {len(p.types)} types; {len(p.skipped)} instances skipped"]
    for name, text in texts.items():
        key = next((k for k, m, _ in GROUPS if f"{m}.F90" == name), None)
        ntypes = f", {len(p.groups.get(key, []))} types" if key else ""
        head.append(f"{dest_dir / name}: {text.count(chr(10))} lines{ntypes}")
    head.append(f"namelist guards: {len(p.notes_guards)} instances, {p.ncomp_guards} type components")
    head += [f"guard {n}: if ({c})" for n, c in sorted(p.notes_guards.items())]
    report[:0] = head
    return p, report
