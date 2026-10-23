"""
Unit-test and capture support for sources rewritten by `spel compact rewrite`
(the manifest `.spel-compact.json` in components/elm/src/main lists the groups).

A generated unit test reads and writes the full layout (nc_io allocates the
type components over the test's bounds and duplicate_sets offsets them). The
compact fields, however, must alias their group buffers while the routines
run. The case gets a generated module `SpelCompactMod` (write_compact_module)
that keeps the IO arrays as shadows and switches between the two around the
compute loop (the !#COMPACT_* tokens of main.F90):

  !#COMPACT_INIT    after elm_init: remember the IO arrays (shadows)
  !#COMPACT_BEFORE  first time: process filters from the clumps' filters, groups,
                    bind; every time: bind, pack shadow -> group per clump
  !#COMPACT_AFTER   unpack group -> shadow per clump, point the fields back at
                    the shadows (IO and duplicate_sets see the full layout)

The clump offset comes from get_clump_bounds (the rewritten decompMod). The
capture in ELM (instrument_elm) does the inverse: it gathers each compact
field into a full-layout scratch array through the inverse map before
write_elmtypes and rebinds after, so the reference files hold what a build
without the compact layout holds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

MANIFEST = "components/elm/src/main/.spel-compact.json"
MODULE = "SpelCompactMod"
MODULE_FILE = f"{MODULE}.F90"
# components the rewrite adds and the unit test sets itself: never captured
RUNTIME_COMPONENTS = {"cgrp_id"}
RUNTIME_TYPES: set[str] = set()  # (the v1 layout's buffer type; none in v2)
RUNTIME_MODULES = {"compactlayoutmod"}  # its instances (compact_groups) are run-time state
MAIN = {"p": "soilp", "c": "soilc"}
LEVEL = {"p": "compact_lev_patch", "c": "compact_lev_col"}


@dataclass(frozen=True)
class CompactField:
    """One compact field of an active instance: `instance%path%comp` (path "" for the instance's own type)."""

    instance: str
    path: str
    comp: str
    holder: str  # holder type (vegetation_nutrient_flux)
    holder_module: str  # module declaring the holder and `<holder>_compact_bind`
    instance_module: str
    subgrid: str  # p | c
    rank: int = 1

    @property
    def prefix(self) -> str:
        return f"{self.instance}%{self.path}" if self.path else self.instance

    @property
    def designator(self) -> str:
        return f"{self.prefix}%{self.comp}"

    @property
    def shadow(self) -> str:
        return "sh_" + "_".join(p for p in (self.instance, self.path, self.comp) if p)

    @property
    def scratch(self) -> str:
        return "cap_" + "_".join(p for p in (self.instance, self.path, self.comp) if p)

    @property
    def bind_call(self) -> str:
        return f"call {self.holder}_compact_bind({self.prefix})"


def load_manifest(srcroot) -> Optional[dict]:
    path = Path(srcroot) / MANIFEST
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def compact_fields(type_dict: dict, active_instances: dict, manifest: Optional[dict]) -> list[CompactField]:
    """
    The compact fields of the case: every field of the manifest's groups whose
    instance is active and whose component the test uses (nc_io allocates only
    those; the others stay bound to their group, which nothing reads).
    """
    if not manifest:
        return []
    out: list[CompactField] = []
    for tname, info in manifest.get("types", {}).items():
        holder = type_dict.get(tname)
        if holder is None:
            continue
        level = info.get("level", "p")
        embedded = info.get("embedded_in") or {inst: "" for inst in info.get("instances", [])}
        fields = [(f, g.get("rank", 1)) for g in info.get("groups", []) for f in g.get("fields", [])]
        for inst, path in embedded.items():
            var = active_instances.get(inst)
            if var is None:
                continue
            ctype = type_dict.get(var.type)
            for comp, rank in fields:
                c = holder.components.get(comp)
                # the analysis flattens the holder's fields into the concrete type (`core%x`)
                flat = ctype.components.get(f"{path}%{comp}") if (ctype is not None and path) else None
                active = (flat.active if flat is not None else False) or (c is not None and c.active and not path)
                if not active:
                    continue
                out.append(
                    CompactField(
                        inst, path, comp, holder.type_name, holder.declaration or "",
                        var.declaration or "", level, rank,
                    )
                )
    return out


def deactivate_runtime_components(type_dict: dict) -> list[str]:
    """
    Mark the rewrite's run-time state inactive, so no IO/alloc/copy code is
    made for it: the `cgrp_id` components, and the instances CompactLayoutMod
    declares (`compact_groups`) with their components. The unit test builds
    that state itself (SpelCompactMod).
    """
    dropped = []
    for dtype in type_dict.values():
        runtime_type = any(
            str(getattr(inst, "declaration", "")).lower() in RUNTIME_MODULES for inst in getattr(dtype, "instances", {}).values()
        )
        for inst in getattr(dtype, "instances", {}).values():
            if getattr(inst, "active", False) and str(getattr(inst, "declaration", "")).lower() in RUNTIME_MODULES:
                inst.active = False
                dropped.append(inst.name)
        for name, comp in dtype.components.items():
            base = name.split("%")[-1].lower()
            if comp.active and (runtime_type or base in RUNTIME_COMPONENTS or str(comp.type).lower() in RUNTIME_TYPES):
                comp.active = False
                dropped.append(f"{dtype.type_name}%{name}")
    return sorted(dropped)


def _uses(fields: list[CompactField]) -> list[str]:
    mods: dict[str, list[str]] = {}
    for f in fields:
        procs = mods.setdefault(f.holder_module, [])
        if f"{f.holder}_compact_bind" not in procs:
            procs.append(f"{f.holder}_compact_bind")
        insts = mods.setdefault(f.instance_module, [])
        if f.instance not in insts:
            insts.append(f.instance)
    return [f"  use {m}, only : {', '.join(n)}\n" for m, n in sorted(mods.items()) if m]


def _binds(fields: list[CompactField]) -> list[str]:
    seen: list[str] = []
    for f in fields:
        if f.bind_call not in seen:
            seen.append(f.bind_call)
    return seen


FILTERS_ROUTINE = """\
  subroutine filters_from_clumps(bounds)
    ! the process soil filters (counts, clump offsets, inverse maps) from the clumps' filters
    type(bounds_type), intent(in) :: bounds
    integer :: nc, nclumps, k, p, c
    integer, allocatable :: nump(:), numc(:), idxp(:), idxc(:)
    logical, allocatable :: specp(:), specc(:)
    nclumps = get_proc_clumps()
    allocate(nump(nclumps), numc(nclumps))
    do nc = 1, nclumps
       nump(nc) = filter(nc)%num_soilp
       numc(nc) = filter(nc)%num_soilc
    end do
    allocate(idxp(max(sum(nump), 1)), idxc(max(sum(numc), 1)))
    k = 0
    do nc = 1, nclumps
       idxp(k+1:k+nump(nc)) = filter(nc)%soilp(1:nump(nc))
       k = k + nump(nc)
    end do
    k = 0
    do nc = 1, nclumps
       idxc(k+1:k+numc(nc)) = filter(nc)%soilc(1:numc(nc))
       k = k + numc(nc)
    end do
    allocate(specp(bounds%begp:bounds%endp), specc(bounds%begc:bounds%endc))
    ! the special-landunit flags only matter for history (the group fill), which a unit test
    ! has not; the test may not have read lun_pp%ifspecial or the landunit maps
    specp(:) = .false.
    specc(:) = .false.
    if (associated(lun_pp%ifspecial) .and. associated(veg_pp%landunit) .and. associated(col_pp%landunit)) then
       do p = bounds%begp, bounds%endp
          specp(p) = lun_pp%ifspecial(veg_pp%landunit(p))
       end do
       do c = bounds%begc, bounds%endc
          specc(c) = lun_pp%ifspecial(col_pp%landunit(c))
       end do
    end if
    call compact_filters_set(compact_lev_patch, bounds%begp, bounds%endp, nump, idxp, specp)
    call compact_filters_set(compact_lev_col, bounds%begc, bounds%endc, numc, idxc, specc)
  end subroutine filters_from_clumps
"""


def compact_module(fields: list[CompactField]) -> str:
    """The unit test's SpelCompactMod (see the module docstring)."""
    uses = "".join(_uses(fields))
    shadows = "".join(
        f"  real(r8), pointer :: {f.shadow}({':,:' if f.rank == 2 else ':'}) => null()\n" for f in fields
    )
    save = "".join(
        f"    {f.shadow} => {f.designator}\n"
        f"    if (.not. associated({f.shadow})) call missing('{f.designator}')\n"
        for f in fields
    )
    restore = "".join(f"    {f.designator} => {f.shadow}\n" for f in fields)
    binds = "".join(f"    {b}\n" for b in _binds(fields))
    pack, unpack = [], []
    for sg in ("p", "c"):
        fs = [f for f in fields if f.subgrid == sg]
        if not fs:
            continue
        main = MAIN[sg]
        head = [
            f"    do nc = 1, nclumps\n",
            f"       off = compact_{main}_clump_off(nc)\n",
            f"       n = filter(nc)%num_{main}\n",
        ]
        def lev(f):
            return ", :" if f.rank == 2 else ""
        pack += head + [f"       {f.designator}(off+1:off+n{lev(f)}) = {f.shadow}(filter(nc)%{main}(1:n){lev(f)})\n" for f in fs] + ["    end do\n"]
        unpack += head + [f"       {f.shadow}(filter(nc)%{main}(1:n){lev(f)}) = {f.designator}(off+1:off+n{lev(f)})\n" for f in fs] + ["    end do\n"]
    return f"""module {MODULE}
  !!! Auto-generated by SPEL (spel.transforms.compact.unit_test): the unit test
  !!! reads and writes the full layout; the compact fields (spel compact rewrite)
  !!! are packed into their group buffers around the compute loop.
  use shr_kind_mod, only : r8 => shr_kind_r8
  use decompMod, only : bounds_type, get_proc_clumps
  use filterMod, only : filter
  use LandunitType, only : lun_pp
  use VegetationType, only : veg_pp
  use ColumnType, only : col_pp
  use CompactLayoutMod, only : compact_filters_set, compact_lev_patch, compact_lev_col, compact_bound, &
       compact_soilp_clump_off, compact_soilc_clump_off, compact_report
{uses}  implicit none
  private
  public :: spel_compact_init, spel_compact_pack, spel_compact_unpack
  type(bounds_type) :: bounds_saved
  logical :: layout_ready = .false.
  integer :: nmissing = 0
{shadows}contains

  subroutine spel_compact_init(bounds)
    ! after elm_init: the IO arrays become the shadows
    type(bounds_type), intent(in) :: bounds
    bounds_saved = bounds
{save}    if (nmissing > 0) stop "SpelCompactMod: compact fields the IO did not allocate (see above)"
  end subroutine spel_compact_init

  subroutine missing(name)
    character(len=*), intent(in) :: name
    nmissing = nmissing + 1
    write(*, '(a,a,a)') "SpelCompactMod: ", name, " is a compact field but the unit test's IO did not allocate it"
  end subroutine missing

  subroutine spel_compact_pack()
    ! before compute: bind the fields to their groups and fill each clump's slice
    integer :: nc, nclumps, off, n
    if (.not. layout_ready) then
       call filters_from_clumps(bounds_saved)
{binds}       compact_bound = .true.
       layout_ready = .true.
       call compact_report(6)
    else
{binds}    end if
    nclumps = get_proc_clumps()
{"".join(pack)}  end subroutine spel_compact_pack

  subroutine spel_compact_unpack()
    ! after compute: the clumps' slices back into the shadows, fields back to the full layout
    integer :: nc, nclumps, off, n
    nclumps = get_proc_clumps()
{"".join(unpack)}{restore}  end subroutine spel_compact_unpack

{FILTERS_ROUTINE}end module {MODULE}
"""


def write_compact_module(unit_test) -> Optional[str]:
    """
    Write the case's SpelCompactMod.F90 if its sources carry a compact
    rewrite and the test uses compact fields; remembers them on
    `unit_test.compact_fields` for main.F90. Returns the file name or None.
    """
    import spel.scripts.io.helper as hio

    manifest = load_manifest(unit_test.e3sm_srcroot)
    if not manifest:
        unit_test.compact_fields = []
        return None
    active_instances, _, _ = hio.get_var_usage_and_elm_inst_vars(unit_test.type_dict)
    fields = compact_fields(unit_test.type_dict, active_instances, manifest)
    unit_test.compact_fields = fields
    if not fields:
        return None
    (Path(unit_test.case_dir) / MODULE_FILE).write_text(compact_module(fields))
    return MODULE_FILE


def hook_lines(fields: list[CompactField]) -> dict[str, list[str]]:
    """Lines for the !#COMPACT_* tokens of main.F90 (empty if no field)."""
    if not fields:
        return {}
    # multicore OpenACC runs in host memory, so nothing more is needed; a real
    # device (SPEL_ACC_GPU, set by the generated CMake for ACC=GPU) would need
    # the groups and the rebinding on the device
    init = [
        "#if defined(_OPENACC) && defined(SPEL_ACC_GPU)\n",
        '   stop "compact layout (spel compact rewrite): no OpenACC GPU unit tests yet"\n',
        "#endif\n",
        "   call spel_compact_init(bounds_proc)\n",
    ]
    return {
        "!#COMPACT_INIT": init,
        "!#COMPACT_BEFORE": ["      call spel_compact_pack()\n"],
        "!#COMPACT_AFTER": ["      call spel_compact_unpack()\n"],
        "!#USE_START": [f"use {MODULE}, only : spel_compact_init, spel_compact_pack, spel_compact_unpack\n"],
    }


def capture_pieces(fields: list[CompactField]) -> tuple[str, str, str, str, str]:
    """
    For the ELM capture module (instrument_elm): (use lines, declarations,
    lines before a capture, lines after, contained procedures). Before a
    capture every compact field is gathered into a full-layout scratch array
    through the inverse map (members from the group, active special-landunit
    points the group's fill, others spval) and the component points at it;
    after, the fields are rebound to their groups. Captures must run with one
    thread per process (the components are shared).
    """
    if not fields:
        return "", "", "", "", ""
    use = "".join(_uses(fields))
    use += (
        "  use CompactLayoutMod, only : compact_cidx_p, compact_cidx_c, compact_groups, compact_locate, "
        "compact_locate2, compact_spval\n"
    )
    decl = "".join(
        f"  real(r8), allocatable, target :: {f.scratch}({':,:' if f.rank == 2 else ':'})\n" for f in fields
    )
    gather = []
    for f in fields:
        sg = f.subgrid
        r2 = "2" if f.rank == 2 else ""
        gather += [
            f"    call gather_{sg}{r2}({f.designator}, {f.scratch}, bounds%beg{sg}, bounds%end{sg}, proc%beg{sg}, proc%end{sg})\n",
            f"    {f.designator} => {f.scratch}\n",
        ]
    binds = "".join(f"    {b}\n" for b in _binds(fields))

    def gather_routine(sg: str) -> str:
        return f"""
  subroutine gather_{sg}(field, scratch, beg, end, pbeg, pend)
    ! the full layout of a compact field on beg:end (scratch allocated over the process range)
    real(r8), pointer, intent(in) :: field(:)
    real(r8), allocatable, target, intent(inout) :: scratch(:)
    integer, intent(in) :: beg, end, pbeg, pend
    integer :: i, k, g, s
    real(r8) :: fill
    if (.not. allocated(scratch)) then
       allocate(scratch(pbeg:pend))
       scratch(:) = compact_spval
    end if
    call compact_locate(field, g, s)
    fill = compact_spval
    if (g > 0) fill = compact_groups(g)%fill
    do i = beg, end
       k = compact_cidx_{sg}(i)
       if (k > 0) then
          scratch(i) = field(k)
       else if (k < 0) then
          scratch(i) = fill
       else
          scratch(i) = compact_spval
       end if
    end do
  end subroutine gather_{sg}
"""

    def gather_routine2(sg: str) -> str:
        return f"""
  subroutine gather_{sg}2(field, scratch, beg, end, pbeg, pend)
    ! the full layout of a rank-2 compact field on beg:end (scratch allocated over the process range)
    real(r8), pointer, intent(in) :: field(:,:)
    real(r8), allocatable, target, intent(inout) :: scratch(:,:)
    integer, intent(in) :: beg, end, pbeg, pend
    integer :: i, k, g, s
    real(r8) :: fill
    if (.not. allocated(scratch)) then
       allocate(scratch(pbeg:pend, size(field, 2)))
       scratch(:,:) = compact_spval
    end if
    call compact_locate2(field, g, s)
    fill = compact_spval
    if (g > 0) fill = compact_groups(g)%fill
    do i = beg, end
       k = compact_cidx_{sg}(i)
       if (k > 0) then
          scratch(i, :) = field(k, :)
       else if (k < 0) then
          scratch(i, :) = fill
       else
          scratch(i, :) = compact_spval
       end if
    end do
  end subroutine gather_{sg}2
"""

    levels = sorted({f.subgrid for f in fields if f.rank == 1})
    levels2 = sorted({f.subgrid for f in fields if f.rank == 2})
    procs = f"""
  subroutine compact_capture_begin(bounds)
    ! compact layout (spel compact rewrite): the full layout of the clump's compact fields
    type(bounds_type), intent(in) :: bounds
    type(bounds_type) :: proc
    call get_proc_bounds(proc)
{"".join(gather)}  end subroutine compact_capture_begin

  subroutine compact_capture_end()
    ! back to the compact storage
{binds}  end subroutine compact_capture_end
{"".join(gather_routine(sg) for sg in levels)}{"".join(gather_routine2(sg) for sg in levels2)}"""
    pre = "      call compact_capture_begin(bounds)\n"
    post = "      call compact_capture_end()\n"
    return use, decl, pre, post, procs
