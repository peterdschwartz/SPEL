"""
Generate `SpelPyMod.F90`, a `bind(C)` layer over a unit-test case, and its manifest
(`spel_py_manifest.json`), from the case's analysis (fut.pkl) and the inventory.

The library built from the case (CMake `SPEL_PYLIB=ON`) exports:

  spel_py_init(path, n)            spel_io_init + elm_init(nsets=1, nclumps=1) + time sync:
                                   reads the domain the builder wrote under `path`
  spel_py_compute()                the routines under test over the clumps, called with the
                                   actuals elm_drv binds (fut.driver_bindings)
  spel_py_view(id, addr, lb, ub)   address and bounds of variable `id` (zero-copy numpy view);
                                   returns its rank, -1 if not allocated, -2 for an unknown id
  spel_py_true()                   the integer image of .true. (logicals are viewed as int32)
  spel_py_trap(on)                 halt on invalid/divide-by-zero/overflow (ieee_set_halting_mode)

and, for cases made with spel_varMod (the manifest's `direct`), the init without an inputs file:

  spel_py_begin(path, n)           spel_io_init + read_constants (the constants file under `path`)
  spel_py_set_bounds(id, ...)      bounds and fill value of variable `id` (spel_varMod::spel_var_desc)
  spel_py_allocate()               allocate every variable that has bounds (spel_var_allocate);
                                   Python then writes topology and parameters through the views
  spel_py_setup(ng, nt, nl, nc, np) process bounds (elements per level), elm_setup(nsets=1, nclumps=1),
                                   time sync

Nothing here reads Fortran text: names, types, ranks and modules come from the inventory, the
derived-type instances (`helper.get_active_instances`), the routine's `record.dummy_args`, and
the driver's `ArgBinding`s. Fortran lines come from `spel.transforms.core.fortran`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from spel.proptest.errors import ProptestError
from spel.proptest.inventory import Inventory, VarInfo
from spel.transforms.core.fortran import declaration, if_block, use_stmt, wrap_statement

SHIM_FILE = "SpelPyMod.F90"
MANIFEST_FILE = "spel_py_manifest.json"
MODULE = "SpelPyMod"

# Fortran type spec and helper suffix per inventory type
FTYPES = {"real": ("real(r8)", "r8"), "integer": ("integer", "int"), "logical": ("logical", "lgc")}
CLUMP_ARRAYS = {"filter", "filter_inactive_and_active"}  # filterMod: one element per clump


class ShimError(ProptestError):
    pass


@dataclass
class Entry:
    """One variable Python can view."""

    id: int
    name: str  # inst%field, filter%member, or a module variable
    ref: str  # its Fortran designator in the shim
    type: str  # real | integer | logical
    rank: int
    kind: str  # inventory kind, or filter | constant | time
    module: str  # module the shim uses it from
    use_name: str  # name in the use statement
    check: Optional[str] = None  # allocated()/associated() test
    nc_name: str = ""
    level: Optional[str] = None
    dims: list[str] = field(default_factory=list)
    units: str = ""
    alloc: bool = False  # allocated through a descriptor by spel_py_allocate (direct init)


def instance_modules(ut) -> dict[str, str]:
    """instance -> module declaring it (the same instances ReadWriteMod handles)."""
    from spel.scripts.io.helper import get_active_instances

    out = {name.lower(): var.declaration.lower() for name, var in get_active_instances(ut.type_dict).items()}
    for typ in ut.type_dict.values():
        for name, var in typ.instances.items():
            out.setdefault(name.lower(), (var.declaration or "").lower())
    return out


def global_module(ut, name: str) -> Optional[str]:
    for mod_name, mod in ut.module_dict.items():
        if name in getattr(mod, "global_vars", {}):
            return mod_name
    return None


def var_check(rank: int, pointer: bool, ref: str) -> Optional[str]:
    """`associated` for pointer components (`Variable.ptrscalar`: declared with the pointer
    attribute), `allocated` for other arrays, nothing for plain scalars."""
    if pointer:
        return f"associated({ref})"
    return f"allocated({ref})" if rank > 0 else None


def collect_entries(inv: Inventory) -> tuple[list[Entry], list[str]]:
    """Every variable the shim exposes, and the ones it cannot (with the reason)."""
    from spel.scripts.io.netcdf_io import alloc_status

    ut = inv.unit_test
    inst_mod = instance_modules(ut)
    entries: list[Entry] = []
    skipped: list[str] = []

    def add(**kw):
        entries.append(Entry(id=len(entries) + 1, **kw))

    for v in inv.vars.values():
        if v.type not in FTYPES:
            skipped.append(f"{v.name}: type {v.type}")
            continue
        if v.driver:
            # ReadWriteMod keeps elm_drv's scalar arguments as `spel_drv_<name>(1)`, logicals as integers
            add(
                name=v.name, ref=f"{v.nc_name}(1)", type="real" if v.type == "real" else "integer", rank=0,
                kind=v.kind, module="readwritemod", use_name=v.nc_name, nc_name=v.nc_name, units=v.units,
            )
            continue
        mod = inst_mod.get(v.instance)
        if v.instance == "bounds" or not mod:
            skipped.append(f"{v.name}: no module variable {v.instance}")
            continue
        check = var_check(v.rank, v.ptrscalar, v.name)
        add(
            name=v.name, ref=v.name, type=v.type, rank=v.rank, kind=v.kind, module=mod, use_name=v.instance,
            check=check, nc_name=v.nc_name, level=v.level, dims=[d.text for d in v.dims],
            units=v.units, alloc=check is not None,
        )

    filter_mod = global_module(ut, "filter") or "filtermod"
    clumpfilter = ut.type_dict.get("clumpfilter")
    for member, rank in sorted(inv.filters.items()):
        ref = f"filter(1)%{member}"
        comp = clumpfilter.components.get(member) if clumpfilter is not None else None
        add(
            name=f"filter%{member}", ref=ref, type="integer", rank=rank, kind="filter", module=filter_mod,
            use_name="filter", check=var_check(rank, bool(comp is not None and comp.ptrscalar), ref),
        )

    seen = {e.name for e in entries}
    for c in inv.constants.values():
        if c.type not in FTYPES or c.name in seen:
            continue
        var = ut.non_parameter_global_vars.get(c.name)
        add(
            name=c.name, ref=c.name, type=c.type, rank=c.rank, kind="constant", module=c.module,
            use_name=c.name, check=alloc_status(var) if var is not None else None, nc_name=c.name,
            dims=[d.text for d in c.dims],
        )
        seen.add(c.name)
    for s in getattr(ut, "time_scalars", []) or []:
        if s.name in seen or s.type not in FTYPES:
            continue
        add(name=s.name, ref=s.name, type=s.type, rank=0, kind="time", module=s.module, use_name=s.name)
        seen.add(s.name)
    # ReadWriteMod's per-step arrays `spel_t_<name>(1)`, which the time sync copies into the
    # scalars: a direct init sets them as the inputs file would
    for s in getattr(ut, "time_scalars", []) or []:
        if s.array in seen:
            continue
        add(name=s.array, ref=f"{s.array}(1)", type="real" if s.io_type == "real" else "integer", rank=0,
            kind="time_io", module="readwritemod", use_name=s.array, nc_name=s.array)
        seen.add(s.array)
    return entries, skipped


def direct_init(inv: Inventory) -> bool:
    """Whether the case can be initialized without an inputs file: made since spel_varMod (and
    elm_initializeMod::elm_setup) exist. Older cases need `spel create` for it."""
    return (inv.case_dir / "spel_varMod.F90").is_file()


# -- routine call, from the analysis


def routine_call(inv: Inventory, sub_id: str, bounds: str, clump: str) -> tuple[list[str], dict[str, set[str]]]:
    """`call <routine>(actuals)` as elm_drv makes it, and the uses the actuals need.

    One actual per dummy of the callee (`record.dummy_args`), taken from the driver's binding of
    that dummy: the clump bounds for the driver's bounds local, `filter(nc)%member` for clump
    arrays, module variables by name, captured driver scalars as `spel_drv_<name>(1)`.
    """
    from spel.scripts.fortran_parser.symbols import Origin

    ut = inv.unit_test
    sub = ut.primary_subroutines[sub_id]
    if getattr(sub, "type_bound", False) or getattr(sub, "bound_to", None):
        raise ShimError(f"{sub_id}: type-bound calls are not supported yet")
    bind = {b.dummy.lower(): b for b in ut.driver_bindings if b.callee == sub_id}
    site = ut.driver_sites.calls.get(sub_id, [None])[0] if getattr(ut, "driver_sites", None) else None
    drv_args = {k.lower(): v for k, v in (getattr(ut, "driver_args", {}) or {}).items()}

    uses: dict[str, set[str]] = {}
    actuals: list[str] = []
    for d in sub.record.dummy_args:
        b = bind.get(d.name.lower())
        if b is None:
            if getattr(d, "optional", False) or "optional" in (getattr(d, "attrs", None) or ()):
                continue
            raise ShimError(f"{sub_id}: elm_drv's call binds nothing to dummy `{d.name}`")
        actual = b.actual.lower()
        base, sep, rest = actual.partition("%")
        if b.origin == Origin.LOCAL and site is not None and actual == site.bounds.lower():
            actuals.append(bounds)
        elif base in drv_args:
            typ = drv_args[base]
            actuals.append(f"(spel_drv_{base}(1) /= 0)" if typ == "logical" else f"spel_drv_{base}(1)")
            uses.setdefault("readwritemod", set()).add(f"spel_drv_{base}")
        elif b.origin == Origin.GLOBAL:
            mod = global_module(ut, base) or instance_modules(ut).get(base)
            if not mod:
                raise ShimError(f"{sub_id}: no module declares `{base}` (actual of `{d.name}`)")
            uses.setdefault(mod, set()).add(base)
            actuals.append(f"{base}({clump})%{rest}" if base in CLUMP_ARRAYS and sep else actual)
        else:
            raise ShimError(f"{sub_id}: actual `{b.actual}` of `{d.name}` ({b.origin}) is not supported")
    name = sub_id.partition("::")[2] or sub.name
    uses.setdefault(sub_id.partition("::")[0], set()).add(name)
    return wrap_statement(f"call {name}({', '.join(actuals)})"), uses


# -- module text


def _view_case(e: Entry) -> list[str]:
    _, sfx = FTYPES[e.type]
    body = [f"call spel_py_addr_{sfx}{'0' if e.rank == 0 else ''}({e.ref}{'' if e.rank == 0 else f', size({e.ref})'}, addr)"]
    body.append(f"rank = {e.rank}")
    if e.rank:
        body += [f"lb(1:{e.rank}) = lbound({e.ref})", f"ub(1:{e.rank}) = ubound({e.ref})"]
    lines = [f"case ({e.id})  ! {e.name}"]
    if e.check:
        lines += if_block("   ", e.check, body)
    else:
        lines += [f"   {b}" for b in body]
    return lines


def _direct_routines(entries: list[Entry], sync_body: list[str]) -> list[str]:
    """spel_py_begin, spel_py_set_bounds, spel_py_allocate, spel_py_setup (see module doc)."""
    out = [
        "subroutine spel_py_begin(path, n) bind(C, name='spel_py_begin')",
        "   integer(c_int), value, intent(in) :: n",
        "   character(kind=c_char), intent(in) :: path(n)",
        "   character(len=n) :: dir",
        "   integer :: i",
        "   do i = 1, n",
        "      dir(i:i) = path(i)",
        "   end do",
        "   call spel_io_init(dir, read_io=.true.)",
        "   call read_constants(io_constants)",
        "end subroutine spel_py_begin",
        "",
        "subroutine spel_py_set_bounds(id, name, nlen, rank, lb, ub, has_fill, fill_r8, fill_int) &",
        "     bind(C, name='spel_py_set_bounds')",
        "   integer(c_int), value, intent(in) :: id, nlen, rank, has_fill, fill_int",
        "   character(kind=c_char), intent(in) :: name(nlen)",
        "   integer(c_int), intent(in) :: lb(*), ub(*)",
        "   real(c_double), value, intent(in) :: fill_r8",
        "   integer :: i",
        "   if (id < 1 .or. id > size(spel_vars)) stop 'spel_py_set_bounds: unknown id'",
        "   if (rank < 0 .or. rank > SPEL_MAXRANK) stop 'spel_py_set_bounds: rank out of range'",
        "   spel_vars(id)%name = ''",
        "   do i = 1, min(nlen, len(spel_vars(id)%name))",
        "      spel_vars(id)%name(i:i) = name(i)",
        "   end do",
        "   spel_vars(id)%rank = rank",
        "   if (rank > 0) then",
        "      spel_vars(id)%lb(1:rank) = lb(1:rank)",
        "      spel_vars(id)%ub(1:rank) = ub(1:rank)",
        "   end if",
        "   spel_vars(id)%has_bounds = .true.",
        "   spel_vars(id)%has_fill = has_fill /= 0",
        "   spel_vars(id)%fill_r8 = fill_r8",
        "   spel_vars(id)%fill_int = fill_int",
        "end subroutine spel_py_set_bounds",
        "",
        "subroutine spel_py_allocate() bind(C, name='spel_py_allocate')",
    ]
    for e in entries:
        if e.alloc:
            out += wrap_statement(f"   if (spel_vars({e.id})%has_bounds) call spel_var_allocate({e.ref}, spel_vars({e.id}))")
    out += [
        "end subroutine spel_py_allocate",
        "",
        "subroutine spel_py_setup(ng, nt, nl, nc, np) bind(C, name='spel_py_setup')",
        "   integer(c_int), value, intent(in) :: ng, nt, nl, nc, np",
        "   type(bounds_type) :: bounds_proc",
        "   bounds_proc%begg = 1; bounds_proc%endg = ng",
        "   bounds_proc%begt = 1; bounds_proc%endt = nt",
        "   bounds_proc%begl = 1; bounds_proc%endl = nl",
        "   bounds_proc%begc = 1; bounds_proc%endc = nc",
        "   bounds_proc%begp = 1; bounds_proc%endp = np",
        "   call elm_setup(1, bounds_proc, 1)",
        *sync_body,
        "end subroutine spel_py_setup",
        "",
    ]
    return out


def _helpers() -> list[str]:
    out = []
    for ftype, sfx in FTYPES.values():
        out += [
            f"subroutine spel_py_addr_{sfx}(x, n, addr)",
            "   integer, intent(in) :: n",
            f"   {ftype}, target, intent(in) :: x(*)",
            "   type(c_ptr), intent(out) :: addr",
            "   addr = c_null_ptr",
            "   if (n > 0) addr = c_loc(x(1))",
            f"end subroutine spel_py_addr_{sfx}",
            "",
            f"subroutine spel_py_addr_{sfx}0(x, addr)",
            f"   {ftype}, target, intent(in) :: x",
            "   type(c_ptr), intent(out) :: addr",
            "   addr = c_loc(x)",
            f"end subroutine spel_py_addr_{sfx}0",
            "",
        ]
    return out


def generate(inv: Inventory) -> tuple[str, dict]:
    """(SpelPyMod.F90 text, manifest)."""
    from spel.scripts.io.time_scalars import sync_lines

    ut = inv.unit_test
    entries, skipped = collect_entries(inv)

    uses: dict[str, set[str]] = {}
    for e in entries:
        uses.setdefault(e.module.lower(), set()).add(e.use_name)
    calls: list[str] = []
    for sub_id in inv.routines:
        lines, u = routine_call(inv, sub_id, "bounds_clump", "nc")
        calls += lines
        for m, names in u.items():
            uses.setdefault(m.lower(), set()).update(names)

    scalars = list(getattr(ut, "time_scalars", []) or [])
    _, sync_body = sync_lines(scalars) if scalars else ([], [])
    for s in scalars:  # the uses sync_lines' statements need, from the TimeScalar objects
        uses.setdefault("readwritemod", set()).add(s.array)
        uses.setdefault(s.module.lower(), set()).add(s.name)
    uses.setdefault("timeinfomod", set()).update({"dtime_mod", "year_curr"})  # elm_init's arguments
    direct = direct_init(inv)

    out = [f"module {MODULE}", "! generated by spel proptest (spel/proptest/shim.py): do not edit"]
    out += use_stmt("   ", "iso_c_binding")
    out += use_stmt("   ", "ieee_exceptions", ["ieee_set_halting_mode", "ieee_invalid", "ieee_divide_by_zero",
                                                "ieee_overflow"])
    out += use_stmt("   ", "shr_kind_mod", ["r8 => shr_kind_r8"])
    out += use_stmt("   ", "decompMod", ["bounds_type", "get_clump_bounds", "procinfo"])
    out += use_stmt("   ", "nc_io", ["spel_io_init"])
    out += use_stmt("   ", "elm_initializeMod", ["elm_init", "elm_setup"] if direct else ["elm_init"])
    if direct:
        out += use_stmt("   ", "FUTConstantsMod", ["read_constants"])
        out += use_stmt("   ", "nc_io", ["io_constants"])
        out += use_stmt("   ", "spel_varMod", ["spel_var_desc", "spel_var_allocate", "SPEL_MAXRANK"])
    for mod in sorted(uses):
        out += use_stmt("   ", mod, sorted(uses[mod]))
    public = ["spel_py_init", "spel_py_compute", "spel_py_view", "spel_py_true", "spel_py_trap"]
    if direct:
        public += ["spel_py_begin", "spel_py_set_bounds", "spel_py_allocate", "spel_py_setup"]
    out += ["   implicit none", "   private", *wrap_statement(f"   public :: {', '.join(public)}")]
    if direct:
        out += [f"   type(spel_var_desc), save :: spel_vars({len(entries)})  ! by manifest id"]
    out += ["contains", ""]

    out += [
        "subroutine spel_py_init(path, n) bind(C, name='spel_py_init')",
        "   integer(c_int), value, intent(in) :: n",
        "   character(kind=c_char), intent(in) :: path(n)",
    ]
    out += [
        "   character(len=n) :: dir",
        "   type(bounds_type) :: bounds_proc",
        "   integer :: i",
        "   do i = 1, n",
        "      dir(i:i) = path(i)",
        "   end do",
        "   call spel_io_init(dir, read_io=.true.)",
        "   call elm_init(1, dtime_mod, year_curr, bounds_proc, 1)",
    ]
    out += [b.rstrip("\n") for b in sync_body]
    out += ["end subroutine spel_py_init", ""]
    if direct:
        out += _direct_routines(entries, [b.rstrip("\n") for b in sync_body])

    out += [
        "subroutine spel_py_compute() bind(C, name='spel_py_compute')",
        "   type(bounds_type) :: bounds_clump",
        "   integer :: nc",
        "   do nc = 1, procinfo%nclumps",
        "      call get_clump_bounds(nc, bounds_clump)",
        *[f"      {c}" for c in calls],
        "   end do",
        "end subroutine spel_py_compute",
        "",
        "subroutine spel_py_trap(on) bind(C, name='spel_py_trap')",
        "   integer(c_int), value, intent(in) :: on",
        "   call ieee_set_halting_mode([ieee_invalid, ieee_divide_by_zero, ieee_overflow], on /= 0)",
        "end subroutine spel_py_trap",
        "",
        "function spel_py_true() result(t) bind(C, name='spel_py_true')",
        "   integer(c_int) :: t",
        "   t = transfer(.true., 0_c_int)",
        "end function spel_py_true",
        "",
        "function spel_py_view(id, addr, lb, ub) result(rank) bind(C, name='spel_py_view')",
        "   integer(c_int), value, intent(in) :: id",
        "   type(c_ptr), intent(out) :: addr",
        "   integer(c_int), intent(out) :: lb(7), ub(7)",
        "   integer(c_int) :: rank",
        "   addr = c_null_ptr",
        "   lb = 1",
        "   ub = 0",
        "   rank = -1",
        "   select case (id)",
    ]
    for e in entries:
        out += ["   " + line for line in _view_case(e)]
    out += ["   case default", "      rank = -2", "   end select", "end function spel_py_view", ""]
    out += _helpers()
    out += [f"end module {MODULE}", ""]

    manifest = {
        "case": inv.case_dir.name,
        "routines": inv.routines,
        "entries": [asdict(e) for e in entries],
        "skipped": skipped,
        "aliases": inv.aliases,
        "time_scalars": [{"name": s.name, "array": s.array, "type": s.type, "io_type": s.io_type,
                          "fallback": s.fallback} for s in scalars],
        "driver_args": dict(getattr(ut, "driver_args", {}) or {}),
        "direct": direct,
    }
    return "\n".join(out), manifest


def write(inv: Inventory) -> tuple[Path, Path]:
    text, manifest = generate(inv)
    src = inv.case_dir / SHIM_FILE
    man = inv.case_dir / MANIFEST_FILE
    # unchanged files keep their time stamps, so an up-to-date library is not rebuilt
    for path, new in ((src, text), (man, json.dumps(manifest, indent=1))):
        if not path.is_file() or path.read_text() != new:
            path.write_text(new)
    return src, man


def current(inv: Inventory) -> bool:
    """Whether the case's shim and manifest are what this inventory generates."""
    text, manifest = generate(inv)
    src, man = inv.case_dir / SHIM_FILE, inv.case_dir / MANIFEST_FILE
    return (src.is_file() and man.is_file() and src.read_text() == text
            and man.read_text() == json.dumps(manifest, indent=1))
