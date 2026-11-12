"""
Gridcell subsets of the reference data.

A subset capture writes only some gridcells of the reference simulation,
with all of their topounits, landunits, columns and patches, renumbered
from 1, so the spel-*.nc files look like a capture of a smaller domain and
the unit test needs no changes.

The gridcells are chosen in ELM, at the first capture, by a mask
(`SubsetSpec`): named presets (`urban`, `lake`, `crop`, ...) or a Fortran
logical expression at one subgrid level, e.g.

    landunit: lun_pp%itype(l) == istsoil .and. lun_pp%wtgcell(l) > 0.5_r8

A gridcell matches if the expression is true for it or any of its
topounits/landunits/columns/patches. Up to `max_cells` matching gridcells
are kept, evenly spaced over the matches. With several MPI ranks, every rank
with matches writes its share (in proportion to its matches) to its own files,
<tag>.rankNNNNN.spel-*.nc, and the master rank lists those ranks in
<tag>.spel-subset-ranks.txt; `merge_rank_files` (run by `spel validate`,
`spel instrument --run-case`, or `spel merge-subset`) joins them into one
capture, gridcells ordered by global index and cut to `max_cells`.

nc_io does the writing (spel_subset_type, nc_subset_select): it keeps the
selected positions along each variable's subgrid dimension and renumbers the
subgrid index fields (netcdf_io.SUBGRID_INDEX_FIELDS). `subset_files` applies
the same rules offline to existing captures.
"""

from __future__ import annotations

import re
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MAX_CELLS = 100
# Per-rank subset captures: <tag>.rankNNNNN.spel-*.nc, listed in <tag>.spel-subset-ranks.txt
MANIFEST_SUFFIX = "spel-subset-ranks.txt"
RANK_WIDTH = 5

# level -> (loop index, Fortran index of its gridcell, process bounds suffix)
LEVELS = {
    "g": ("g", "g", "g"),
    "t": ("t", "top_pp%gridcell(t)", "t"),
    "l": ("l", "lun_pp%gridcell(l)", "l"),
    "c": ("c", "col_pp%gridcell(c)", "c"),
    "p": ("p", "veg_pp%gridcell(p)", "p"),
}
LEVEL_NAMES = {
    "gridcell": "g", "g": "g",
    "topounit": "t", "t": "t",
    "landunit": "l", "l": "l",
    "column": "c", "c": "c",
    "patch": "p", "p": "p",
}
# Instances the selection uses (always) and the modules custom masks may use
TYPE_USES = (
    "GridcellType, only : grc_pp",
    "TopounitType, only : top_pp",
    "LandunitType, only : lun_pp",
    "ColumnType, only : col_pp",
    "VegetationType, only : veg_pp",
)
CUSTOM_USES = ("landunit_varcon", "column_varcon", "pftvarcon")


class SubsetError(ValueError):
    pass


@dataclass(frozen=True)
class MaskTerm:
    """A gridcell matches if `expr` holds at `level` for it or one of its children."""

    level: str
    expr: str
    uses: tuple[str, ...] = ()


# Weights: zero-weight landunits exist (e.g. urban with run_zero_weight_urban),
# so presets require a positive weight on the gridcell.
PRESETS: dict[str, tuple[MaskTerm, str]] = {
    "all": (MaskTerm("g", ".true."), "every gridcell"),
    "urban": (
        MaskTerm("l", "lun_pp%urbpoi(l) .and. lun_pp%wtgcell(l) > 0._r8"),
        "an urban landunit with weight > 0",
    ),
    "lake": (
        MaskTerm("l", "lun_pp%lakpoi(l) .and. lun_pp%wtgcell(l) > 0._r8"),
        "a lake landunit with weight > 0",
    ),
    "crop": (
        MaskTerm(
            "l",
            "lun_pp%itype(l) == istcrop .and. lun_pp%wtgcell(l) > 0._r8",
            ("landunit_varcon, only : istcrop",),
        ),
        "a crop landunit (istcrop) with weight > 0",
    ),
    "glacier": (
        MaskTerm(
            "l",
            "(lun_pp%itype(l) == istice .or. lun_pp%itype(l) == istice_mec) "
            ".and. lun_pp%wtgcell(l) > 0._r8",
            ("landunit_varcon, only : istice, istice_mec",),
        ),
        "a glacier landunit with weight > 0",
    ),
    "wetland": (
        MaskTerm(
            "l",
            "lun_pp%itype(l) == istwet .and. lun_pp%wtgcell(l) > 0._r8",
            ("landunit_varcon, only : istwet",),
        ),
        "a wetland landunit with weight > 0",
    ),
}

_CUSTOM = re.compile(r"^\s*(\w+)\s*:(.*)$", re.S)
_MAX = re.compile(r";\s*max\s*=\s*(\d+)\s*$", re.I)


@dataclass(frozen=True)
class SubsetSpec:
    mask: str
    terms: tuple[MaskTerm, ...]
    max_cells: int = DEFAULT_MAX_CELLS
    custom: bool = False

    @classmethod
    def parse(cls, text: str, max_cells: int | None = None) -> "SubsetSpec":
        """
        `urban`, `urban,lake` (either), or `<level>: <Fortran expression>`;
        an optional `; max=N` suffix overrides `max_cells`.
        """
        if m := _MAX.search(text):
            max_cells = int(m.group(1))
            text = text[: m.start()]
        text = text.strip()
        if max_cells is None:
            max_cells = DEFAULT_MAX_CELLS
        if max_cells < 1:
            raise SubsetError(f"subset max must be >= 1, got {max_cells}")
        if (m := _CUSTOM.match(text)) and m.group(1).lower() in LEVEL_NAMES:
            level = LEVEL_NAMES[m.group(1).lower()]
            expr = " ".join(m.group(2).split())
            check_expression(expr, level)
            return cls(text, (MaskTerm(level, expr),), max_cells, custom=True)
        names = [n.strip().lower() for n in text.split(",")]
        if unknown := [n for n in names if n not in PRESETS]:
            raise SubsetError(
                f"unknown subset mask {', '.join(unknown)!s}: use one of "
                f"{', '.join(PRESETS)} or '<level>: <Fortran expression>' "
                f"with level one of gridcell, topounit, landunit, column, patch"
            )
        return cls(",".join(names), tuple(PRESETS[n][0] for n in names), max_cells)

    @property
    def description(self) -> str:
        return f"{self.mask} (max {self.max_cells} gridcells)"

    def to_json(self) -> dict:
        return {"mask": self.mask, "max_cells": self.max_cells}

    @classmethod
    def from_json(cls, data: dict) -> "SubsetSpec":
        return cls.parse(data["mask"], data.get("max_cells", DEFAULT_MAX_CELLS))

    def uses(self) -> list[str]:
        uses = list(TYPE_USES)
        for term in self.terms:
            uses += [u for u in term.uses if u not in uses]
        if self.custom:
            uses += list(CUSTOM_USES)
        return uses


def check_expression(expr: str, level: str) -> None:
    """Cheap checks; the ELM build is the real test of a custom expression."""
    if not expr:
        raise SubsetError("empty subset expression")
    if bad := set(expr) & set(";!&\"'\n"):
        raise SubsetError(f"subset expression may not contain {''.join(sorted(bad))!r}: {expr}")
    if expr.count("(") != expr.count(")"):
        raise SubsetError(f"unbalanced parentheses in subset expression: {expr}")
    idx = LEVELS[level][0]
    if not re.search(rf"\b{idx}\b", expr, re.I):
        raise SubsetError(f"a {level}-level subset expression must use the index {idx!r}: {expr}")


# --------------------------------------------------------------------------
# Capture module (ELM side)
# --------------------------------------------------------------------------
def fortran_string(text: str, width: int = 60) -> str:
    """A character literal, split into concatenated pieces to keep lines short."""
    pieces = [text[i : i + width] for i in range(0, len(text), width)] or [""]
    return " // &\n      ".join("'" + p.replace("'", "''") + "'" for p in pieces)


def wrap_expression(expr: str, width: int = 70) -> str:
    """Continue a long expression over several lines (custom masks have no strings)."""
    lines, line = [], ""
    for word in expr.split(" "):
        if line and len(line) + len(word) > width:
            lines.append(line)
            line = ""
        line += ("" if not line else " ") + word
    return " &\n          ".join(lines + [line])


def capture_select_routine(spec: SubsetSpec, tag: str, prefix: str | None = None) -> str:
    """
    spel_select_subset(bounds) for SpelCapture_<tag>: sets spel_sub (the
    subset), spel_bounds_sub (its bounds, from 1), spel_clump_begg (the
    clump it lies in) and spel_writer, and the master rank writes the
    manifest <prefix>spel-subset-ranks.txt (see merge_rank_files).
    """
    prefix = f"{tag}." if prefix is None else prefix
    loops = ""
    for term in spec.terms:
        idx, parent, b = LEVELS[term.level]
        loops += (
            f"    do {idx} = bp%beg{b}, bp%end{b}\n"
            f"      if ({wrap_expression(term.expr)}) match({parent}) = .true.\n"
            f"    end do\n"
        )
    mask = fortran_string(spec.mask[:500])
    return f"""
  subroutine spel_select_subset(bounds)
    ! Gridcells of the process matching the subset mask, restricted to the
    ! process's clump with the most matches: the capture runs around that
    ! clump's call (when an earlier clump's call returns, later clumps have no
    ! outputs yet). Every rank with matches keeps its share of
    ! spel_subset_max (in proportion to its matches, at least one), evenly
    ! spaced, with their topounits, landunits, columns and patches, and writes
    ! its own files (<tag>.rankNNNNN.spel-*.nc); `spel subset merge` joins them.
    ! The master rank lists the writing ranks in the manifest.
    type(bounds_type), intent(in) :: bounds
    type(bounds_type) :: bp, bc
    logical, allocatable :: match(:), keep(:)
    integer, allocatable :: cand(:), nkeep(:)
    integer :: g, t, l, c, p, k, n, nc, nmatch, ntotal, ierr
    call get_proc_bounds(bp)
    allocate (match(bp%begg:bp%endg), keep(bp%begg:bp%endg))
    match = .false.
    keep = .false.
{loops}    nmatch = 0
    spel_clump_begg = bounds%begg
    do nc = 1, get_proc_clumps()
      call get_clump_bounds(nc, bc)
      if (count(match(bc%begg:bc%endg)) > nmatch) then
        nmatch = count(match(bc%begg:bc%endg))
        spel_clump_begg = bc%begg
        k = bc%endg
      end if
    end do
    if (nmatch > 0) then
      match(bp%begg:spel_clump_begg - 1) = .false.
      match(k + 1:bp%endg) = .false.
    end if
    call mpi_allreduce(nmatch, ntotal, 1, MPI_INTEGER, MPI_SUM, mpicom, ierr)
    n = 0
    if (nmatch > 0) n = min(nmatch, ceiling(real(spel_subset_max, r8)*nmatch/ntotal))
    allocate (nkeep(0:npes - 1))
    nkeep = 0
    call mpi_gather(n, 1, MPI_INTEGER, nkeep, 1, MPI_INTEGER, 0, mpicom, ierr)
    if (masterproc) then
      call spel_write_rank_manifest(nkeep, ntotal)
      if (ntotal == 0) print *, 'SPEL subset ({tag}): no gridcell matches ', &
        {mask}
    end if
    spel_writer = n > 0
    if (.not. spel_writer) return

    cand = pack([(g, g = bp%begg, bp%endg)], match)
    do k = 1, n
      keep(cand(1 + ((k - 1)*nmatch)/n)) = .true.
    end do
    call nc_subset_set_level(spel_sub, 1, bp%begg, keep)
    call nc_subset_set_level(spel_sub, 2, bp%begt, keep(top_pp%gridcell(bp%begt:bp%endt)))
    call nc_subset_set_level(spel_sub, 3, bp%begl, keep(lun_pp%gridcell(bp%begl:bp%endl)))
    call nc_subset_set_level(spel_sub, 4, bp%begc, keep(col_pp%gridcell(bp%begc:bp%endc)))
    call nc_subset_set_level(spel_sub, 5, bp%begp, keep(veg_pp%gridcell(bp%begp:bp%endp)))
    spel_sub%mask = {mask}
    spel_sub%gindex = pack(grc_pp%gindex(bp%begg:bp%endg), keep)
    spel_sub%latdeg = pack(grc_pp%latdeg(bp%begg:bp%endg), keep)
    spel_sub%londeg = pack(grc_pp%londeg(bp%begg:bp%endg), keep)

    spel_bounds_sub = bounds
    spel_bounds_sub%begg = 1; spel_bounds_sub%endg = nc_subset_count(spel_sub, 1)
    spel_bounds_sub%begt = 1; spel_bounds_sub%endt = nc_subset_count(spel_sub, 2)
    spel_bounds_sub%begl = 1; spel_bounds_sub%endl = nc_subset_count(spel_sub, 3)
    spel_bounds_sub%begc = 1; spel_bounds_sub%endc = nc_subset_count(spel_sub, 4)
    spel_bounds_sub%begp = 1; spel_bounds_sub%endp = nc_subset_count(spel_sub, 5)
    print *, 'SPEL subset ({tag}): rank', iam, ' writes', n, ' of', nmatch, &
      ' matching gridcells of its clump starting at gridcell', spel_clump_begg
    print *, 'SPEL subset ({tag}): gindex', spel_sub%gindex
  end subroutine spel_select_subset

  subroutine spel_write_rank_manifest(nkeep, ntotal)
    ! Which ranks write a subset capture (`spel subset merge` reads exactly these)
    integer, intent(in) :: nkeep(0:), ntotal
    integer :: u, r
    open (newunit=u, file='{prefix}{MANIFEST_SUFFIX}', status='replace', action='write')
    write (u, '(a)') '# SPEL subset capture of case {tag}: rank, gridcells it writes'
    write (u, '(a,i0)') 'max ', spel_subset_max
    write (u, '(a,i0)') 'matches ', ntotal
    do r = 0, size(nkeep) - 1
      if (nkeep(r) > 0) write (u, '(i0,1x,i0)') r, nkeep(r)
    end do
    close (u)
  end subroutine spel_write_rank_manifest
"""


# --------------------------------------------------------------------------
# Offline subsets of existing captures
# --------------------------------------------------------------------------
SUBGRID_DIMS = {"gridcell": "g", "topo": "t", "landunit": "l", "column": "c", "patch": "p"}


LEVEL_PREFIX = {"t": "top_pp", "l": "lun_pp", "c": "col_pp", "p": "veg_pp"}
PARENT_FIELD = {"g": "gridcell", "t": "topounit", "l": "landunit", "c": "column"}


def gridcell_parents(ds, sizes: dict[str, int]) -> dict[str, "object"]:
    """
    0-based gridcell of every topounit/landunit/column/patch of capture file
    `ds` (netCDF4.Dataset; `sizes`: level -> entries, "g" included). A
    capture holds only the index fields its routines read, so the gridcell
    is taken directly (`col_pp__gridcell`), through a coarser level
    (`veg_pp__landunit` and the landunit's gridcell), or for a coarser level
    from its children (`col_pp__topounit` and the column's gridcell). A
    capture without topounit index fields and one topounit per gridcell has
    topounit t in gridcell t. Raises SubsetError if an entry stays unresolved.
    """
    import numpy as np

    order = SUBGRID_LEVEL_ORDER
    levels = [lev for lev in ("t", "l", "c", "p") if lev in sizes]

    def field(lev: str, parent: str):
        """0-based `parent` index of each `lev` entry (-1: not an index), or None."""
        name = f"{LEVEL_PREFIX[lev]}__{PARENT_FIELD[parent]}"
        if name not in ds.variables:
            return None
        v = ds[name]
        vals = np.asarray(v[0] if "time" in v.dimensions else v[...]).astype(np.int64).ravel()
        if vals.size != sizes[lev]:
            return None
        beg = int(ds[f"bounds__beg{parent}"][...]) if f"bounds__beg{parent}" in ds.variables else 1
        k = vals - beg
        return np.where((k >= 0) & (k < sizes[parent]), k, -1)

    out = {lev: np.full(sizes[lev], -1, np.int64) for lev in levels}
    if sizes["g"] == 1:
        return {lev: np.zeros(sizes[lev], np.int64) for lev in levels}
    for lev in levels:
        if (f := field(lev, "g")) is not None:
            out[lev] = f
    has_topounit_fields = any(
        f"{LEVEL_PREFIX[lev]}__topounit" in ds.variables for lev in levels
    ) or "top_pp__gridcell" in ds.variables
    if "t" in out and not has_topounit_fields and sizes["t"] == sizes["g"]:
        out["t"] = np.arange(sizes["g"])
    changed = True
    while changed:
        changed = False
        for lev in levels:
            cur = out[lev]
            if (cur >= 0).all():
                continue
            new = cur.copy()
            for other in levels:
                if other == lev:
                    continue
                if order.index(other) < order.index(lev):  # lev -> coarser `other`
                    if (f := field(lev, other)) is not None:
                        via = np.where(f >= 0, out[other][np.maximum(f, 0)], -1)
                        take = (new < 0) & (via >= 0)
                        new[take] = via[take]
                elif (f := field(other, lev)) is not None:  # finer `other` -> lev
                    ok = (f >= 0) & (out[other] >= 0)
                    via = np.full(sizes[lev], -1, np.int64)
                    via[f[ok]] = out[other][ok]
                    take = (new < 0) & (via >= 0)
                    new[take] = via[take]
            if (new != cur).any():
                out[lev] = new
                changed = True
    missing = [lev for lev in levels if (out[lev] < 0).any()]
    if missing:
        names = {"t": "topounits", "l": "landunits", "c": "columns", "p": "patches"}
        raise SubsetError(
            f"{Path(ds.filepath()).name}: can't tell the gridcell of every one of its "
            f"{', '.join(names[m] for m in missing)} from the index fields it holds"
        )
    return out


def level_keep(structure: Path, gridcells: Sequence[int]) -> dict:
    """
    Kept positions of each level for 1-based `gridcells`, from the index
    fields of a capture (an inputs/outputs file; time step 0).
    """
    import netCDF4
    import numpy as np

    with netCDF4.Dataset(structure) as ds:
        ds.set_auto_mask(False)
        ng = len(ds.dimensions["gridcell"]) if "gridcell" in ds.dimensions else None
        if ng is None:
            raise SubsetError(f"{structure} has no gridcell dimension")
        bad = [g for g in gridcells if not 1 <= g <= ng]
        if bad:
            raise SubsetError(f"gridcells {bad} out of range 1..{ng}")
        keep_g = np.zeros(ng, bool)
        keep_g[np.asarray(gridcells, int) - 1] = True
        sizes = {lev: len(ds.dimensions[dim]) for dim, lev in SUBGRID_DIMS.items() if dim in ds.dimensions}
        sizes["g"] = ng
        keep = {"g": keep_g}
        for lev, parent in gridcell_parents(ds, sizes).items():
            keep[lev] = keep_g[parent]
    return keep


FILL_INT = -9999  # nc_io's fill_int: an index of a dropped element


def _map_index(values, newidx, beg: int):
    """
    Index values through `newidx` (new 1-based index of each old position
    beg, beg+1, ...; 0 = dropped -> FILL_INT). Values outside the old range
    (-1, ispval, fill) are not indices and stay as they are, as in nc_io.
    """
    import numpy as np

    v = np.asarray(values).astype(np.int64)
    k = v - beg
    inside = (k >= 0) & (k < newidx.size)
    new = np.where(inside, newidx[np.clip(k, 0, max(newidx.size - 1, 0))], 0) if newidx.size else 0
    out = np.where(inside, np.where(new > 0, new, FILL_INT), v)
    return out.astype(np.asarray(values).dtype)


def _renumber(values, field: str, lev: str, keep: dict, beg: int):
    import numpy as np

    if field == "gindex":
        return np.broadcast_to(np.arange(1, values.shape[-1] + 1, dtype=values.dtype), values.shape)
    newidx = np.zeros(keep[lev].size, np.int64)
    newidx[keep[lev]] = np.arange(1, keep[lev].sum() + 1)
    return _map_index(values, newidx, beg)


def _subgrid_axis(dimensions) -> int | None:
    """netCDF axis of a variable's first Fortran subgrid dimension (netCDF order is reversed)."""
    axes = [i for i, d in enumerate(dimensions) if d in SUBGRID_DIMS]
    return axes[-1] if axes else None


def _set_subgrid_bounds(attrs: dict, dimensions, ax: int, count: int) -> None:
    """lbounds/ubounds attributes after the subgrid axis `ax` was cut to `count` entries."""
    import numpy as np

    fdims = [d for d in dimensions if d != "time"][::-1]
    fi = fdims.index(dimensions[ax])
    for key, val in (("lbounds", 1), ("ubounds", count)):
        if key in attrs:
            b = np.array(attrs[key], ndmin=1).copy()
            b[fi] = val
            attrs[key] = b


def subset_dataset(src: Path, dst: Path, keep: dict, mask: str = "", gindex=None) -> None:
    """Write the kept gridcells of capture file `src` to `dst`, as nc_io does in ELM."""
    import netCDF4
    import numpy as np

    from spel.scripts.io.netcdf_io import index_field_levels

    fields = index_field_levels()
    counts = {lev: int(k.sum()) for lev, k in keep.items()}
    with netCDF4.Dataset(src) as si, netCDF4.Dataset(dst, "w", format=si.data_model) as so:
        si.set_auto_mask(False)
        so.set_auto_mask(False)
        begs = {
            lev: int(si[f"bounds__beg{lev}"][...]) if f"bounds__beg{lev}" in si.variables else 1
            for lev in keep
        }
        so.setncatts({a: si.getncattr(a) for a in si.ncattrs()})
        so.setncattr("spel_subset_mask", mask)
        so.setncattr("spel_subset_gindex", np.asarray(gindex if gindex is not None else
                                                      np.flatnonzero(keep["g"]) + 1, np.int32))
        for name, dim in si.dimensions.items():
            lev = SUBGRID_DIMS.get(name)
            if dim.isunlimited():
                so.createDimension(name, None)
            elif lev is not None:
                if len(dim) != keep[lev].size:
                    raise SubsetError(f"{src}: {name} has {len(dim)} entries, structure has {keep[lev].size}")
                so.createDimension(name, counts[lev])
            else:
                so.createDimension(name, len(dim))
        for name, vi in si.variables.items():
            attrs = {a: vi.getncattr(a) for a in vi.ncattrs()}
            fill = attrs.pop("_FillValue", None)
            vo = so.createVariable(name, vi.datatype, vi.dimensions, fill_value=fill)
            data = vi[...]
            ax = _subgrid_axis(vi.dimensions)
            if ax is not None:
                lev = SUBGRID_DIMS[vi.dimensions[ax]]
                data = np.compress(keep[lev], data, axis=ax)
                _set_subgrid_bounds(attrs, vi.dimensions, ax, counts[lev])
            if (m := re.fullmatch(r"bounds__(beg|end)([gtlcp])", name)) and m.group(2) in counts:
                data = np.asarray(1 if m.group(1) == "beg" else counts[m.group(2)], vi.dtype)
            i = name.rfind("_pp__")
            if i >= 0 and name[i + 5:] in fields:
                lev = fields[name[i + 5:]]
                data = _renumber(np.asarray(data), name[i + 5:], lev, keep, begs[lev])
            vo.setncatts(attrs)
            vo[...] = data


def subset_files(src_dir: Path, dst_dir: Path, gridcells: Sequence[int], mask: str = "") -> list[Path]:
    """Subset every spel-*.nc capture in src_dir to the 1-based `gridcells`."""
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    files = sorted(src_dir.glob("spel-*.nc"))
    structure = next((f for f in files if not f.name.startswith("spel-constants")), None)
    if structure is None:
        raise SubsetError(f"no spel-inputs/outputs*.nc in {src_dir}")
    keep = level_keep(structure, gridcells)
    mask = mask or f"gridcells {','.join(map(str, gridcells))}"
    dst_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for f in files:
        subset_dataset(f, dst_dir / f.name, keep, mask=mask, gindex=sorted(gridcells))
        out.append(dst_dir / f.name)
    for extra in ("lnd_in",):
        if (src_dir / extra).is_file():
            shutil.copy2(src_dir / extra, dst_dir / extra)
    return out


# --------------------------------------------------------------------------
# Merging the per-rank files of a subset capture
# --------------------------------------------------------------------------
# With a subset, every MPI rank with matching gridcells writes its own files,
# <tag>.rankNNNNN.spel-*.nc, each one already a capture of a small domain
# numbered from 1; the master rank lists the writing ranks in
# <tag>.spel-subset-ranks.txt. The merge joins them into one capture: the
# gridcells ordered by their global index (so the result does not depend on
# the decomposition) and cut to spel_subset_max, evenly spaced as in ELM, each
# with its topounits, landunits, columns and patches, and the subgrid index
# fields renumbered. Gridcells never span ranks and the index fields point
# within a gridcell, so no index refers to another rank's file. Variables
# without a subgrid dimension must be identical on every rank.
_RANK_FILE = re.compile(r"^rank(\d+)\.(spel-.+\.nc)$")
SUBSET_ATTRS = ("spel_subset_gindex", "spel_subset_latdeg", "spel_subset_londeg")


@dataclass(frozen=True)
class RankManifest:
    max_cells: int
    matches: int
    ranks: dict[int, int]  # rank -> gridcells it writes

    @classmethod
    def read(cls, path: Path) -> "RankManifest":
        max_cells = matches = None
        ranks: dict[int, int] = {}
        for ln, line in enumerate(Path(path).read_text().splitlines(), 1):
            words = line.split()
            if not words or words[0].startswith("#"):
                continue
            try:
                if words[0] == "max":
                    max_cells = int(words[1])
                elif words[0] == "matches":
                    matches = int(words[1])
                else:
                    ranks[int(words[0])] = int(words[1])
            except (IndexError, ValueError):
                raise SubsetError(f"{path}:{ln}: can't read {line!r}") from None
        if max_cells is None or matches is None:
            raise SubsetError(f"{path}: no 'max' or 'matches' line")
        return cls(max_cells, matches, ranks)


@dataclass
class MergePlan:
    """Where each rank's elements go: `newidx[rank][lev]` = new 1-based index (0: dropped)."""

    ranks: list[int]
    newidx: dict[int, dict[str, "object"]]
    order: dict[str, list[tuple[int, int]]]  # lev -> [(rank, 0-based position)] in merged order
    gridcell_rank: list[int]
    attrs: dict[str, list]  # merged spel_subset_* attributes


def _evenly_spaced(n: int, m: int) -> list[int]:
    """0-based picks of m of n, as spel_select_subset's cand(1 + ((k-1)*n)/m)."""
    return list(range(n)) if m >= n else [(k * n) // m for k in range(m)]


def merge_plan(structures: dict[int, Path], max_cells: int | None = None) -> MergePlan:
    """The merge of the ranks' files, from one inputs/outputs file of each rank."""
    import netCDF4
    import numpy as np

    cells = []  # (gindex, rank, 0-based local gridcell)
    parents: dict[int, dict[str, np.ndarray]] = {}
    sizes: dict[int, dict[str, int]] = {}
    attrs: dict[int, dict[str, np.ndarray]] = {}
    for r, path in sorted(structures.items()):
        with netCDF4.Dataset(path) as ds:
            ds.set_auto_mask(False)
            if "spel_subset_gindex" not in ds.ncattrs():
                raise SubsetError(f"{path} has no spel_subset_gindex: not a subset capture")
            attrs[r] = {a: np.atleast_1d(ds.getncattr(a)) for a in SUBSET_ATTRS if a in ds.ncattrs()}
            gindex = attrs[r]["spel_subset_gindex"]
            ng = len(ds.dimensions["gridcell"]) if "gridcell" in ds.dimensions else gindex.size
            if gindex.size != ng:
                raise SubsetError(f"{path}: {gindex.size} gindex values for {ng} gridcells")
            sizes[r] = {lev: len(ds.dimensions[dim]) for dim, lev in SUBGRID_DIMS.items()
                        if dim in ds.dimensions}
            sizes[r]["g"] = ng
            parents[r] = gridcell_parents(ds, sizes[r])
            cells += [(int(gi), r, i) for i, gi in enumerate(gindex)]
    cells.sort()
    dup = sorted({gi for (gi, _, _), (gj, _, _) in zip(cells, cells[1:]) if gi == gj})
    if dup:
        raise SubsetError(f"gridcells {dup[:10]} are in the files of several ranks")
    if max_cells is not None:
        cells = [cells[k] for k in _evenly_spaced(len(cells), max_cells)]

    ranks = sorted(structures)
    gpos = {r: np.zeros(sizes[r]["g"], np.int64) for r in ranks}
    for new, (_, r, i) in enumerate(cells, 1):
        gpos[r][i] = new
    newidx = {r: {"g": gpos[r]} for r in ranks}
    order = {"g": [(r, i) for _, r, i in cells]}
    levels = sorted({lev for r in ranks for lev in parents[r]}, key=SUBGRID_LEVEL_ORDER.index)
    for lev in levels:
        if any(lev not in parents[r] for r in ranks):
            raise SubsetError(f"level {lev} is missing in some ranks' files")
        # elements of kept gridcells, by (merged gridcell, position within the rank)
        elems = sorted(
            (int(gpos[r][pg]), r, i)
            for r in ranks for i, pg in enumerate(parents[r][lev]) if gpos[r][pg] > 0
        )
        order[lev] = [(r, i) for _, r, i in elems]
        for r in ranks:
            newidx[r][lev] = np.zeros(sizes[r][lev], np.int64)
        for new, (_, r, i) in enumerate(elems, 1):
            newidx[r][lev][i] = new
    merged_attrs = {
        a: [attrs[r][a][i] for _, r, i in cells]
        for a in SUBSET_ATTRS if all(a in attrs[r] for r in ranks)
    }
    return MergePlan(ranks, newidx, order, [r for _, r, _ in cells], merged_attrs)


SUBGRID_LEVEL_ORDER = ("g", "t", "l", "c", "p")


def merge_datasets(
    srcs: dict[int, Path], dst: Path, plan: MergePlan, allow_differ: Sequence[str] = ()
) -> list[str]:
    """
    Write the merge of one capture file of every rank (`srcs`: rank -> file)
    to `dst`. Returns the variables without a subgrid dimension that differ
    between ranks and are in `allow_differ` (the first rank's values are
    kept); any other such difference is an error.
    """
    import netCDF4
    import numpy as np

    from spel.scripts.io.netcdf_io import index_field_levels

    fields = index_field_levels()
    ranks = plan.ranks
    if sorted(srcs) != ranks:
        raise SubsetError(f"{dst.name}: files of ranks {sorted(srcs)}, expected {ranks}")
    counts = {lev: len(o) for lev, o in plan.order.items()}
    differ: list[str] = []
    ds = {r: netCDF4.Dataset(srcs[r]) for r in ranks}
    try:
        for d in ds.values():
            d.set_auto_mask(False)
        first = ds[ranks[0]]
        bad = [f"{srcs[r].name}" for r in ranks if set(ds[r].variables) != set(first.variables)]
        if bad:
            raise SubsetError(f"{dst.name}: the variables of {', '.join(bad)} differ from {srcs[ranks[0]].name}'s")
        with netCDF4.Dataset(dst, "w", format=first.data_model) as so:
            so.set_auto_mask(False)
            so.setncatts({a: first.getncattr(a) for a in first.ncattrs()})
            for a, vals in plan.attrs.items():
                so.setncattr(a, np.asarray(vals, dtype=np.asarray(first.getncattr(a)).dtype))
            so.setncattr("spel_subset_rank", np.asarray(plan.gridcell_rank, np.int32))
            for name, dim in first.dimensions.items():
                lev = SUBGRID_DIMS.get(name)
                if dim.isunlimited():
                    so.createDimension(name, None)
                elif lev is not None:
                    for r in ranks:
                        n = len(ds[r].dimensions[name])
                        if n != plan.newidx[r][lev].size:
                            raise SubsetError(f"{srcs[r]}: {name} has {n} entries, its structure "
                                              f"{plan.newidx[r][lev].size}")
                    so.createDimension(name, counts[lev])
                else:
                    if any(len(ds[r].dimensions[name]) != len(dim) for r in ranks):
                        users = [v for v, vi in first.variables.items() if name in vi.dimensions]
                        if not users or any(u not in allow_differ for u in users):
                            raise SubsetError(
                                f"{dst.name}: dimension {name} (of {', '.join(users)}) has a different size "
                                f"on different ranks: not a subgrid dimension, so it can't be merged "
                                f"(`spel merge-subset --allow-differ {' '.join(users)}` keeps the first "
                                "rank's values)"
                            )
                    so.createDimension(name, len(dim))
            for name, vi in first.variables.items():
                attrs = {a: vi.getncattr(a) for a in vi.ncattrs()}
                fill = attrs.pop("_FillValue", None)
                vo = so.createVariable(name, vi.datatype, vi.dimensions, fill_value=fill)
                i = name.rfind("_pp__")
                field = name[i + 5:] if i >= 0 and name[i + 5:] in fields else None
                ax = _subgrid_axis(vi.dimensions)
                if ax is not None:
                    lev = SUBGRID_DIMS[vi.dimensions[ax]]
                    parts, offsets, off = [], {}, 0
                    for r in ranks:
                        if ds[r][name].dimensions != vi.dimensions:
                            raise SubsetError(f"{srcs[r]}: {name} has dimensions {ds[r][name].dimensions}")
                        x = np.asarray(ds[r][name][...])
                        if field and field != "gindex":
                            flev = fields[field]
                            x = _map_index(x, plan.newidx[r][flev], 1)
                        parts.append(x)
                        offsets[r] = off
                        off += x.shape[ax]
                    take = np.asarray([offsets[r] + k for r, k in plan.order[lev]], np.int64)
                    data = np.take(np.concatenate(parts, axis=ax), take, axis=ax)
                    if field == "gindex":
                        data = np.broadcast_to(
                            np.arange(1, data.shape[-1] + 1, dtype=data.dtype), data.shape
                        )
                    _set_subgrid_bounds(attrs, vi.dimensions, ax, counts[lev])
                elif (m := re.fullmatch(r"bounds__(beg|end)([gtlcp])", name)) and m.group(2) in counts:
                    data = np.asarray(1 if m.group(1) == "beg" else counts[m.group(2)], vi.dtype)
                else:
                    data = np.asarray(vi[...])
                    raw = data.tobytes()
                    odd = [r for r in ranks[1:] if np.asarray(ds[r][name][...]).tobytes() != raw]
                    if odd:
                        if name not in allow_differ:
                            raise SubsetError(
                                f"{dst.name}: {name} has no subgrid dimension but differs between ranks "
                                f"{ranks[0]} and {odd[:5]} (`spel merge-subset --allow-differ {name}` "
                                "keeps the first rank's values)"
                            )
                        differ.append(name)
                vo.setncatts(attrs)
                vo[...] = data
    finally:
        for d in ds.values():
            d.close()
    return differ


def rank_files(rundir: Path, prefix: str, manifest: RankManifest) -> dict[str, dict[int, Path]]:
    """{spel-*.nc name: {rank: path}} of the ranks the manifest lists."""
    out: dict[str, dict[int, Path]] = {}
    for f in Path(rundir).glob(f"{prefix}rank*.spel-*.nc"):
        m = _RANK_FILE.match(f.name[len(prefix):])
        if m and int(m.group(1)) in manifest.ranks:
            out.setdefault(m.group(2), {})[int(m.group(1))] = f
    want = set(manifest.ranks)
    for name, files in out.items():
        if set(files) != want:
            missing = sorted(want - set(files))[:10]
            raise SubsetError(f"{prefix}rank*.{name}: missing for ranks {missing} of the manifest")
    return out


def merge_rank_files(
    rundir: Path, dest: Path, prefix: str, allow_differ: Sequence[str] = ()
) -> list[Path] | None:
    """
    Merge the per-rank subset capture of case file prefix `prefix` (`<tag>.`)
    in `rundir` into dest/spel-*.nc. None when `rundir` has no manifest (not
    a per-rank subset capture); [] when no gridcell matched.
    """
    rundir, dest = Path(rundir), Path(dest)
    mpath = rundir / f"{prefix}{MANIFEST_SUFFIX}"
    if not mpath.is_file():
        return None
    manifest = RankManifest.read(mpath)
    if not manifest.ranks:
        return []
    files = rank_files(rundir, prefix, manifest)
    if not files:
        raise SubsetError(f"{mpath} lists ranks {sorted(manifest.ranks)[:10]}, but {rundir} has no "
                          f"{prefix}rank*.spel-*.nc")
    structure = next((n for n in sorted(files) if not n.startswith("spel-constants")), None)
    if structure is None:
        raise SubsetError(f"no {prefix}rank*.spel-inputs/outputs*.nc in {rundir}")
    plan = merge_plan(files[structure], manifest.max_cells)
    dest.mkdir(parents=True, exist_ok=True)
    out, differ = [], set()
    for name in sorted(files):
        differ.update(merge_datasets(files[name], dest / name, plan, allow_differ))
        out.append(dest / name)
    for name in sorted(differ):
        print(f"merge-subset: {name} differs between ranks; kept rank {plan.ranks[0]}'s values")
    print(f"merge-subset: {counts_line(plan)} from ranks {plan.ranks} "
          f"({manifest.matches} matching gridcells, max {manifest.max_cells})")
    return out


def counts_line(plan: MergePlan) -> str:
    return ", ".join(f"{len(o)} {name}" for name, o in zip(
        ("gridcells", "topounits", "landunits", "columns", "patches"),
        (plan.order.get(lev, []) for lev in SUBGRID_LEVEL_ORDER)))


def diff_datasets(a: Path, b: Path, ignore_attrs: Sequence[str] = ()) -> list[str]:
    """Differences between two netCDF files (dimensions, variables bit-for-bit, attributes)."""
    import netCDF4
    import numpy as np

    out: list[str] = []
    with netCDF4.Dataset(a) as da, netCDF4.Dataset(b) as db:
        da.set_auto_mask(False)
        db.set_auto_mask(False)
        for name in sorted(set(da.dimensions) | set(db.dimensions)):
            la = len(da.dimensions[name]) if name in da.dimensions else None
            lb = len(db.dimensions[name]) if name in db.dimensions else None
            if la != lb:
                out.append(f"dimension {name}: {la} vs {lb}")
        for att in sorted((set(da.ncattrs()) | set(db.ncattrs())) - set(ignore_attrs)):
            va = da.getncattr(att) if att in da.ncattrs() else None
            vb = db.getncattr(att) if att in db.ncattrs() else None
            if not np.array_equal(np.asarray(va), np.asarray(vb)):
                out.append(f"global attribute {att}: {va} vs {vb}")
        for name in sorted(set(da.variables) | set(db.variables)):
            if name not in da.variables or name not in db.variables:
                out.append(f"variable {name} only in {'b' if name in db.variables else 'a'}")
                continue
            va, vb = da[name], db[name]
            if va.dimensions != vb.dimensions:
                out.append(f"{name}: dimensions {va.dimensions} vs {vb.dimensions}")
                continue
            xa, xb = np.asarray(va[...]), np.asarray(vb[...])
            if xa.dtype != xb.dtype or xa.shape != xb.shape:
                out.append(f"{name}: {xa.dtype}{xa.shape} vs {xb.dtype}{xb.shape}")
            elif xa.tobytes() != xb.tobytes():
                out.append(f"{name}: {int(np.sum(xa != xb))} values differ")
            for att in sorted(set(va.ncattrs()) | set(vb.ncattrs())):
                aa = va.getncattr(att) if att in va.ncattrs() else None
                ab = vb.getncattr(att) if att in vb.ncattrs() else None
                if not np.array_equal(np.asarray(aa), np.asarray(ab)):
                    out.append(f"{name}:{att}: {aa} vs {ab}")
    return out
