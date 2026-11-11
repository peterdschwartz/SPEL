"""Synthetic subgrid domains for property-based tests.

The domain is declared as gridcells with ELM's full hierarchy (`hierarchy.py`: topounits,
landunits, columns, patches, weights) in `[[domain.gridcell]]` tables, laid out by
`hierarchy.Layout`. The builder writes what `elm_init` reads (`spel-inputs0001.nc`,
`spel-constants0001.nc`) for every variable of the inventory:

- topology (subgrid indices, types, flags, weights) from the layout (`topology_value`); a
  topology field without a rule stops the build, so a new routine never runs on a silently wrong
  hierarchy;
- parameters without a subgrid dimension (pftcon-like arrays, scalars) and the constants from a
  reference capture (`reference`), with `constants` overrides; the domain decides
  `max_topounits`/`has_topounit` (`DOMAIN_CONSTANTS`);
- with declared topounit elevations, `top_pp%elevation` is pinned to them (`Domain.pinned`):
  `downhill_ti` follows from them, so a scenario cannot draw them;
- everything else (inputs, outputs) as fill values: the runner sets inputs through the shim's
  views before each compute.

With `init = "direct"` the inputs file is not written: `Domain.direct` holds the bounds of every
variable to allocate and the values that are not fill values, and the worker allocates them
through the shim's descriptors (`spel_py_set_bounds`/`spel_py_allocate`, spel_varMod) and writes
the values through views (runtime.Case.init). The constants file is written either way: the case
reads it (`read_constants`), and it carries the namelist options.

Which variables exist, their types, ranks, dimensions and namelist guards all come from the
Inventory (the case's analysis). Dimensions are parsed bounds expressions, evaluated over the
domain sizes and constants; a non-subgrid dimension the reference capture also has takes the
reference's `lbounds`/`ubounds` (e.g. snow layers: the case never sets `nlevsno`).
"""

from __future__ import annotations

import contextlib
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from spel.proptest.errors import ProptestError
from spel.proptest.fexpr import ExprError
from spel.proptest.hierarchy import (TYPE_CODES, Gridcell, HierarchyError, LandunitKind, Layout,
                                     UrbanDensity, check_type_codes, ltype_of)
from spel.proptest.inventory import LEVELS, NC_DIM, Inventory, VarInfo

INPUTS_FILE = "spel-inputs0001.nc"
CONSTANTS_FILE = "spel-constants0001.nc"
REAL_FILL = 1.0e36
INT_FILL = -9999
INIT_MODES = ("files", "direct")

# time scalars (`spel_t_<name>`) a synthetic step uses unless `[domain.time]` says otherwise
TIME_DEFAULTS: dict[str, Any] = {
    "dtime_mod": 1800.0,
    "spel_step_size": 1800,
    "spel_rad_step_size": 1800,
    "spel_nstep": 1,
    "nstep_mod": 1,
    "year_curr": 2000,
    "year_prev": 2000,
    "mon_curr": 7,
    "mon_prev": 7,
    "day_curr": 1,
    "day_prev": 1,
    "jday_mod": 182,
    "secs_curr": 43200,
    "secs_prev": 41400,
    "dayspyr_mod": 365.0,
    "thiscalday_mod": 182.5,
    "nextsw_cday_mod": 182.52,
    "doalb": True,
}


class DomainError(ProptestError):
    pass


@dataclass
class DomainSpec:
    gridcells: list[Gridcell]
    nsets: int = 1
    snow_layers: int = 0
    reference: Optional[Path] = None  # input-data dir of a capture: constants, parameters, extents
    constants: dict[str, Any] = field(default_factory=dict)
    time: dict[str, Any] = field(default_factory=dict)
    dims: dict[str, int] = field(default_factory=dict)  # values for bound names (nlevsno = 5)
    # files: elm_init reads an inputs file | direct: see module doc | None: direct when the case
    # can (made with spel_varMod: shim.direct_init), else files (runner.Session decides)
    init: Optional[str] = None

    KEYS = ("gridcell", "nsets", "snow_layers", "reference", "constants", "time", "dims", "init")

    @classmethod
    def from_table(cls, table: dict, base: Path, input_data: Optional[Path] = None) -> "DomainSpec":
        extra = set(table) - set(cls.KEYS)
        if extra:
            raise DomainError(f"[domain]: unknown keys {sorted(extra)} (sites are [[domain.gridcell]] tables; "
                              "see SPEC.md)")
        cells = table.get("gridcell", [])
        if not isinstance(cells, list) or not cells:
            raise DomainError("[domain]: declare the gridcells as [[domain.gridcell]] tables")
        try:
            gridcells = [Gridcell.from_table(g, f"[[domain.gridcell]] {k}") for k, g in enumerate(cells)]
        except HierarchyError as exc:
            raise DomainError(str(exc)) from None
        spec = cls(
            gridcells=gridcells,
            nsets=int(table.get("nsets", 1)),
            snow_layers=int(table.get("snow_layers", 0)),
            reference=_reference_path(table.get("reference"), base, input_data),
            constants=dict(table.get("constants", {})),
            time={k.lower(): v for k, v in table.get("time", {}).items()},
            dims={k.lower(): int(v) for k, v in table.get("dims", {}).items()},
            init=table.get("init"),
        )
        if spec.init is not None and spec.init not in INIT_MODES:
            raise DomainError(f"[domain]: init must be one of {list(INIT_MODES)}, not {spec.init!r}")
        if spec.nsets < 1:
            raise DomainError("[domain]: nsets must be >= 1")
        return spec

    def to_table(self, input_data: Optional[Path] = None) -> dict:
        """The `[domain]` table (the inverse of from_table)."""
        out: dict[str, Any] = {"nsets": self.nsets}
        if self.reference is not None:
            ref = Path(self.reference)
            under = input_data is not None and ref.resolve().parent == Path(input_data).resolve()
            out["reference"] = ref.name if under else str(ref)
        for key in ("snow_layers",):
            if getattr(self, key):
                out[key] = getattr(self, key)
        for key in ("constants", "time", "dims"):
            if getattr(self, key):
                out[key] = dict(getattr(self, key))
        if self.init is not None:
            out["init"] = self.init
        out["gridcell"] = [g.to_table() for g in self.gridcells]
        return out

    def layout(self, params: Optional[dict[str, Any]] = None) -> Layout:
        """The layout of the declared gridcells (type codes checked against the case's `params`)."""
        try:
            if params:
                check_type_codes(params)
            return Layout(self.gridcells, self.nsets)
        except HierarchyError as exc:
            raise DomainError(str(exc)) from None

    def check(self, inv: Inventory) -> None:
        """The `time` and `dims` keys name something of the case (a misspelt key would otherwise
        leave the default in place without a word)."""
        names = inv.time_names()
        bad = sorted(k for k in self.time if names is not None and k not in names)
        if bad:
            raise DomainError(
                f"[domain.time]: {bad} are not per-step scalars of the case "
                f"(it has {sorted(names)})"
            )
        dims = inv.bound_names()
        bad = sorted(k for k in self.dims if k not in dims)
        if bad:
            raise DomainError(
                f"[domain.dims]: {bad} are not used by any dimension bound "
                f"(bounds use {sorted(dims)})"
            )


def _reference_path(ref: Any, base: Path, input_data: Optional[Path]) -> Optional[Path]:
    if not ref:
        return None
    path = Path(ref)
    if not path.is_absolute():
        cand = (input_data / ref) if input_data is not None else None
        path = cand if cand is not None and cand.is_dir() else (base / ref)
    if not path.is_dir():
        raise DomainError(f"[domain]: reference {ref!r} is not a directory")
    return path


# -- filters

# filterMod members that select the elements of one landunit kind, and the complements that only
# add the others (a routine taking `urbanl` and `nourbanl` works on urban landunits)
FILTER_KINDS = {
    "urbanl": LandunitKind.URBAN, "urbanc": LandunitKind.URBAN, "urbanp": LandunitKind.URBAN,
    "lakec": LandunitKind.LAKE, "lakep": LandunitKind.LAKE, "lakesnowc": LandunitKind.LAKE,
    "lakenosnowc": LandunitKind.LAKE,
}
FILTER_COMPLEMENTS = {"nourbanl": "urbanl", "nourbanc": "urbanc", "nourbanp": "urbanp",
                      "nolakec": "lakec", "nolakep": "lakep"}
KIND_PATCH_FILTER = {LandunitKind.URBAN: "urbanp", LandunitKind.LAKE: "lakep"}  # a scenario's `over`

_U, _L, _S, _C = LandunitKind.URBAN, LandunitKind.LAKE, LandunitKind.SOIL, LandunitKind.CROP


def _road_perv(lay: Layout, level: str) -> np.ndarray:
    c = lay.ancestor_of(level, "c")
    return lay.itype("c")[c] == TYPE_CODES["icol_road_perv"]


# The members setFilters builds from the topology alone (filterMod::setFiltersOneGroup, active
# elements on active topounits): member -> (level, layout, level -> mask). Members that depend on
# other data (pcropp: pftcon, snowc: the snow state) are not predicted.
EXPECTED_FILTERS: dict[str, tuple[str, Callable[[Layout, str], np.ndarray]]] = {
    "urbanl": ("l", lambda y, v: y.is_kind(v, _U)),
    "nourbanl": ("l", lambda y, v: ~y.is_kind(v, _U)),
    "urbanc": ("c", lambda y, v: y.is_kind(v, _U)),
    "nourbanc": ("c", lambda y, v: ~y.is_kind(v, _U)),
    "urbanp": ("p", lambda y, v: y.is_kind(v, _U)),
    "nourbanp": ("p", lambda y, v: ~y.is_kind(v, _U)),
    "lakec": ("c", lambda y, v: y.is_kind(v, _L)),
    "nolakec": ("c", lambda y, v: ~y.is_kind(v, _L)),
    "lakep": ("p", lambda y, v: y.is_kind(v, _L)),
    "nolakep": ("p", lambda y, v: ~y.is_kind(v, _L)),
    "nolakeurbanp": ("p", lambda y, v: ~y.is_kind(v, _L, _U)),
    "soilc": ("c", lambda y, v: y.is_kind(v, _S, _C)),
    "soilp": ("p", lambda y, v: y.is_kind(v, _S, _C)),
    "hydrologyc": ("c", lambda y, v: y.is_kind(v, _S, _C) | _road_perv(y, v)),
    "hydrononsoic": ("c", lambda y, v: _road_perv(y, v)),
}


def expected_filters(layout: Layout) -> dict[str, np.ndarray]:
    """The Fortran indices of every predicted filter member (EXPECTED_FILTERS), active elements only."""
    out = {}
    for member, (level, pred) in EXPECTED_FILTERS.items():
        mask = pred(layout, level) & layout.active(level)
        out[member] = np.flatnonzero(mask) + 1
    return out


def landunit_types(kind: LandunitKind) -> tuple[int, ...]:
    """The lun_pp%itype values of a landunit kind."""
    if kind is LandunitKind.URBAN:
        return tuple(ltype_of(kind, d) for d in UrbanDensity)
    return (ltype_of(kind, None),)


def density_of(ltype: int) -> Optional[UrbanDensity]:
    return next((d for d in UrbanDensity if TYPE_CODES[d.type_code] == ltype), None)


@dataclass
class Domain:
    spec: DomainSpec
    layout: Layout
    directory: Path
    # values the constants file holds (for guards and dimensions)
    constants: dict[str, Any]

    # inst%field -> Fortran bounds per dimension
    extents: dict[str, list[tuple[int, int]]]
    written: list[str]
    # inst%field -> reason (guard off)
    skipped: dict[str, str]

    # parameters the routines read that got fill values: inst%field -> reason. A run refuses
    # them unless the scenario rules them (runner.Session.preflight).
    filled: dict[str, str] = field(default_factory=dict)

    # time scalar -> the default it took
    time_defaulted: dict[str, Any] = field(default_factory=dict)

    # init = "direct": {"sizes": {level: elements}, "alloc": {name: (extents, type)}, "values": {name: array}}
    direct: Optional[dict[str, Any]] = None

    # inputs the domain fixes (declared topounit elevations): name -> values; no rule may set them
    pinned: dict[str, np.ndarray] = field(default_factory=dict)

    # the topology values written: name -> Fortran-shaped array (BuiltDomain compares the case's)
    topology: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def path_arg(self) -> str:
        """What spel_py_init takes: the directory with a trailing slash."""
        return str(self.directory) + "/"


# -- topology rules

BOUND_LEVEL = {"g": "g", "t": "t", "l": "l", "c": "c", "p": "p"}  # bounds_type begX/endX -> level
# Subgrid structure fields by what they hold. A field of a routine that is in none of these (and
# has no rule in `topology_value`) stops the build.
LEVEL_OF_FIELD = {"gridcell": "g", "topounit": "t", "landunit": "l", "column": "c", "patch": "p",
                  "grc_ind": "g"}  # the element's ancestor (or first descendant)
FIRST_CHILD = {"topi": "t", "lndi": "l", "coli": "c", "pfti": "p"}
LAST_CHILD = {"topf": "t", "lndf": "l", "colf": "c", "pftf": "p"}
TOPOLOGY_COUNT = {"npfts": "p", "ncolumns": "c", "nlandunits": "l", "ntopounits": "t"}
# weight fields: the ancestor level the weight is relative to
TOPOLOGY_WEIGHT = {"wtgcell": "g", "wtgcell_save": "g", "wttopounit": "t", "wtlunit": "l", "wtcol": "c"}
TOPOLOGY_FALSE = {"glcmecpoi", "is_fates"}
# logical fields that hold for the elements of landunits of these kinds
TOPOLOGY_KIND = {
    "lakpoi": (_L,), "is_lake": (_L,), "urbpoi": (_U,), "ifspecial": (_U, _L),
    "is_soil": (_S,), "is_on_soil_col": (_S,), "is_crop": (_C,), "is_on_crop_col": (_C,),
}
# constants the domain decides (topounit_varcon): name -> layout -> value
DOMAIN_CONSTANTS: dict[str, Callable[[Layout], Any]] = {
    "max_topounits": lambda y: y.max_topounits,
    "has_topounit": lambda y: y.max_topounits > 1,
}


def topology_value(var: VarInfo, layout: Layout, spec: DomainSpec, ext: list[tuple[int, int]]) -> np.ndarray:
    """Values of a subgrid structure field in its Fortran shape."""
    lev = var.level
    f = var.field
    if f == "landunit_indices" and lev in ("g", "t") and var.rank == 2:
        nl = ext[0][1] - ext[0][0] + 1
        return layout.landunit_indices(lev, nl).astype(np.int32)
    if var.rank != 1:
        raise DomainError(f"{var.name}: topology field of rank {var.rank} (no rule)")
    n = layout.count(lev)
    if f == "topo_grc_ind" and lev == "t":
        return layout.topo_index().astype(np.int32)
    if f == "downhill_ti" and lev == "t":
        try:
            return layout.downhill().astype(np.int32)
        except HierarchyError as exc:
            raise DomainError(f"{var.name}: {exc}") from None
    if f in LEVEL_OF_FIELD:
        target = LEVEL_OF_FIELD[f]
        idx = layout.parent(lev, target) if LEVELS.index(target) <= LEVELS.index(lev) else layout.first_child(lev, target)
        return idx.astype(np.int32)
    if f in FIRST_CHILD:
        return layout.first_child(lev, FIRST_CHILD[f]).astype(np.int32)
    if f in LAST_CHILD:
        return layout.last_child(lev, LAST_CHILD[f]).astype(np.int32)
    if f in TOPOLOGY_COUNT:
        return layout.children(lev, TOPOLOGY_COUNT[f]).astype(np.int32)
    if f in TOPOLOGY_WEIGHT:
        rel = TOPOLOGY_WEIGHT[f]
        return layout.weight(lev, rel) if LEVELS.index(rel) < LEVELS.index(lev) else np.ones(n)
    if f == "active":
        return layout.active(lev)
    if f in TOPOLOGY_FALSE:
        return np.zeros(n, dtype=bool)
    if f in TOPOLOGY_KIND and lev in ("l", "c", "p"):
        return layout.is_kind(lev, *TOPOLOGY_KIND[f])
    if f == "hydrologically_active" and lev == "c":
        # column_varcon::is_hydrologically_active: soil and crop landunits, the pervious road
        return layout.is_kind("c", _S, _C) | _road_perv(layout, "c")
    if f == "snl":
        return np.full(n, -spec.snow_layers, dtype=np.int32)
    if f == "itype" and lev in ("l", "c", "p"):
        return layout.itype(lev).astype(np.int32)
    raise DomainError(
        f"no synthetic-domain rule for topology field {var.name} (level {lev}); "
        "add one to spel/proptest/domain.py (the TOPOLOGY_* tables or topology_value)"
    )


# -- netCDF writing


def read_constants(path: Optional[Path]) -> dict[str, Any]:
    """The scalars and arrays of a constants file (`spel-constants0001.nc`), names in lower case."""
    if path is None or not path.exists():
        return {}
    import netCDF4

    out = {}
    with netCDF4.Dataset(path) as ds:
        for name, v in ds.variables.items():
            out[name.lower()] = (
                np.asarray(v[...]) if v.ndim else np.asarray(v[...]).item()
            )
    return out


_read_constants = read_constants  # the former name


def _dim_name(lo: int, hi: int) -> str:
    return f"x{lo}_{hi}".replace("-", "m")


def _nc_type(ftype: str) -> str:
    return "f8" if ftype == "real" else "i4"


def _fill(ftype: str) -> Any:
    return REAL_FILL if ftype == "real" else INT_FILL


def var_extents(
    var: VarInfo, layout: Layout, env: dict[str, int], ref
) -> list[tuple[int, int]]:
    """Fortran (lb, ub) per dimension: subgrid -> 1..(elements of the level), others from the
    reference or evaluated."""
    ref_lb = ref_ub = None
    if ref is not None and var.nc_name in ref.variables:
        rv = ref.variables[var.nc_name]
        if "lbounds" in rv.ncattrs():
            ref_lb = np.atleast_1d(rv.getncattr("lbounds"))
            ref_ub = np.atleast_1d(rv.getncattr("ubounds"))
    out = []
    for k, d in enumerate(var.dims):
        if d.level:
            out.append((1, layout.count(d.level)))
        elif ref_lb is not None and len(ref_lb) == len(var.dims):
            out.append((int(ref_lb[k]), int(ref_ub[k])))
        else:
            try:
                out.append(d.extent(env))
            except (ExprError, KeyError) as exc:
                raise DomainError(
                    f"{var.name}: dimension {d.text}: {exc} (give it in [domain.dims])"
                ) from None
    return out


def _reference_value(var: VarInfo, ref, extents) -> tuple[Optional[np.ndarray], str]:
    """Step 0 of a non-subgrid parameter in the reference capture, in Fortran dim order; without
    a value, the reason."""
    if ref is None:
        return None, "no reference capture"
    if var.nc_name not in ref.variables:
        return None, "not in the reference capture"
    rv = ref.variables[var.nc_name]
    data = np.asarray(rv[...])
    if "time" in rv.dimensions:
        data = (
            data[rv.dimensions.index("time")] if rv.dimensions[0] != "time" else data[0]
        )
    data = data.T  # file dims are reversed Fortran dims
    shape = tuple(hi - lo + 1 for lo, hi in extents)
    if data.shape != shape:
        return None, f"shape {data.shape} in the reference capture, {shape} here"
    return data, ""


def build_domain(inv: Inventory, spec: DomainSpec, directory: Path) -> Domain:
    import netCDF4

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    layout = spec.layout(inv.parameters)
    decided = {k: f(layout) for k, f in DOMAIN_CONSTANTS.items() if k in inv.constants}
    clash = sorted(k for k in decided if k.lower() in {c.lower() for c in spec.constants})
    if clash:
        raise DomainError(f"{clash} follow from the declared gridcells (the most topounits a gridcell has); "
                          "remove them from [domain.constants]/[namelist]")

    # Both files are written under temporary names and renamed at the end, so a worker of
    # another run on the same domain never reads a half-written file.
    tmp = f".tmp-{os.getpid()}"
    final_const, final_inputs = directory / CONSTANTS_FILE, directory / INPUTS_FILE
    # constants: the reference file, then the overrides
    const_out = directory / (CONSTANTS_FILE + tmp)
    ref_const = spec.reference / CONSTANTS_FILE if spec.reference else None
    if ref_const is not None and ref_const.exists():
        shutil.copyfile(ref_const, const_out)
    else:
        with netCDF4.Dataset(const_out, "w"):
            pass
    _apply_constant_overrides(inv, const_out, {**spec.constants, **decided})
    missing = _complete_constants(inv, const_out)
    constants = _read_constants(const_out)

    env = inv.bound_env(layout.sizes(), constants)
    env.update(spec.dims)
    ref_inputs = spec.reference / INPUTS_FILE if spec.reference else None
    ref = (
        netCDF4.Dataset(ref_inputs)
        if ref_inputs is not None and ref_inputs.exists()
        else None
    )

    extents: dict[str, list[tuple[int, int]]] = {}
    written: list[str] = []
    skipped: dict[str, str] = {}
    filled: dict[str, str] = {}
    time_defaulted: dict[str, Any] = {}
    topology: dict[str, np.ndarray] = {}
    pinned = _pinned(inv, layout)
    direct = spec.init == "direct"
    alloc: dict[str, tuple[list[tuple[int, int]], str]] = {}
    values: dict[str, np.ndarray] = {}

    def put(ds, name: str, ftype: str, ext, data, levels=None, allocate: bool = False) -> None:
        """Into the inputs file, or (direct) into `alloc` and `values`: a value that is all
        fill is left to the descriptor's fill."""
        if not direct:
            _write(ds, name, ftype, ext, data, levels)
            return
        if allocate:
            alloc[name] = (list(ext), ftype)
        arr = np.asarray(data)
        if not allocate or not np.all(arr == (False if ftype == "logical" else _fill(ftype))):
            values[name] = arr

    try:
        with (contextlib.nullcontext() if direct else netCDF4.Dataset(directory / (INPUTS_FILE + tmp), "w")) as ds:
            if not direct:
                ds.createDimension("time", 1)
                for lev in LEVELS:
                    ds.createDimension(NC_DIM[lev], layout.count(lev))
            for var in inv.vars.values():
                if not inv.guard_active(var, constants):
                    skipped[var.name] = f"guard {var.guard} is false"
                    # not written (the case does not read it under these options), but its shape is
                    # known when its dimensions are: a rule for it can then be drawn here and used
                    # by a run under other options
                    try:
                        extents[var.name] = (
                            [] if var.driver else var_extents(var, layout, env, ref)
                        )
                    except DomainError:
                        pass
                    continue
                if var.instance == "bounds":
                    if not direct:  # direct: spel_py_setup sets the process bounds
                        lev = BOUND_LEVEL.get(var.field[3:].removesuffix("_all")) if var.field[:3] in ("beg", "end") else None
                        if lev is None:
                            raise DomainError(f"{var.name}: not a subgrid bound the domain knows")
                        _write(
                            ds,
                            var.nc_name,
                            "integer",
                            [],
                            np.int32(1 if var.field.startswith("beg") else layout.count(lev)),
                        )
                    written.append(var.name)
                    continue
                if var.driver:
                    # ReadWriteMod's spel_drv_<name>(1); a scalar input, so the scenario must rule it
                    # (spec.missing_rules) and the runner sets it before every call
                    extents[var.name] = []
                    if direct:
                        values[var.name] = np.zeros(())
                    else:
                        _write(
                            ds,
                            var.nc_name,
                            "real" if var.type == "real" else "integer",
                            [(1, 1)],
                            np.zeros(1),
                        )
                    written.append(var.name)
                    continue
                ext = var_extents(var, layout, env, ref)
                extents[var.name] = ext
                data, why = _data_for(var, inv, spec, layout, ref, ext)
                if why and var.kind == "parameter":
                    filled[var.name] = why
                if var.name in pinned:
                    data = pinned[var.name]
                if _is_topology(var, inv) or var.name in pinned:
                    topology[var.name] = np.asarray(data)

                put(
                    ds,
                    var.name if direct else var.nc_name,
                    var.type,
                    ext,
                    data,
                    [d.level for d in var.dims],
                    allocate=True,
                )
                written.append(var.name)
            tvalues, time_defaulted = time_values(inv, spec, constants)
            for s in getattr(inv.unit_test, "time_scalars", []) or []:
                io_real = s.io_type == "real"
                value = tvalues[s.name.lower()]
                if direct:  # the shim's entry of ReadWriteMod's spel_t_<name>(1)
                    values[s.array] = np.asarray(float(value) if io_real else int(value))
                else:
                    _write(
                        ds,
                        s.array,
                        "real" if io_real else "integer",
                        [(1, 1)],
                        np.array([value]),
                    )
    finally:
        if ref is not None:
            ref.close()
    os.replace(const_out, final_const)
    if direct:
        final_inputs.unlink(missing_ok=True)  # a files-mode run of the same directory left it
    else:
        os.replace(directory / (INPUTS_FILE + tmp), final_inputs)
    for name in missing:
        skipped[name] = (
            "constant not in the reference: written as 0 (the routines do not read it)"
        )
    return Domain(
        spec=spec,
        layout=layout,
        directory=directory,
        constants=constants,
        extents=extents,
        written=written,
        skipped=skipped,
        filled=filled,
        time_defaulted=time_defaulted,
        direct={"sizes": layout.sizes(), "alloc": alloc, "values": values} if direct else None,
        pinned=pinned,
        topology=topology,
    )


def _pinned(inv: Inventory, layout: Layout) -> dict[str, np.ndarray]:
    """Inputs the declaration fixes: `top_pp%elevation` when topounit elevations are declared
    (`downhill_ti` follows from them)."""
    out = {}
    elev = layout.elevation
    var = inv.vars.get("top_pp%elevation")
    if var is not None and any(e is not None for e in elev):
        if any(e is None for e in elev):
            raise DomainError("declare the elevation of every topounit, or of none")
        out[var.name] = np.asarray(elev, dtype=float)
    return out


def _is_topology(var: VarInfo, inv: Inventory) -> bool:
    return var.kind == "topology" or (var.instance in _physical_instances(inv) and var.type in ("integer", "logical"))


def _complete_constants(inv: Inventory, path: Path) -> list[str]:
    """FUTConstantsMod reads every scalar constant unconditionally: add the ones the file lacks.
    Missing ones the routines read must come from [domain.constants]."""
    import netCDF4

    added = []
    with netCDF4.Dataset(path, "a") as ds:
        have = {k.lower() for k in ds.variables}
        needed = [
            c for c in inv.constants.values() if c.rank == 0 and c.name not in have
        ]
        read = sorted(c.name for c in needed if c.status)
        if read:
            raise DomainError(
                f"constants read by the routines but missing from the reference: {read}; "
                "set them in [domain.constants]"
            )
        for c in needed:
            ds.createVariable(c.name, _nc_type(c.type), ()).assignValue(0)
            added.append(c.name)
    return added


def _data_for(
    var: VarInfo, inv: Inventory, spec: DomainSpec, layout: Layout, ref, ext
) -> tuple[np.ndarray, str]:
    """The values the domain file gets, and for a variable without a subgrid dimension that got
    fill values, why the reference had none."""
    shape = tuple(hi - lo + 1 for lo, hi in ext)
    if _is_topology(var, inv):
        return topology_value(var, layout, spec, ext), ""
    why = ""
    if var.level is None:
        val, why = _reference_value(var, ref, ext)
        if val is not None:
            return val, ""
    if var.type == "logical":
        return np.zeros(shape, dtype=bool), why
    return (
        np.full(
            shape, _fill(var.type), dtype=np.float64 if var.type == "real" else np.int32
        ),
        why,
    )


def _physical_instances(inv: Inventory) -> set[str]:
    from spel.scripts.io.netcdf_io import SUBGRID_INDEX_FIELDS

    types = set(SUBGRID_INDEX_FIELDS)
    return {inst for inst, typ in inv.instances.items() if typ in types}


def _write(
    ds,
    name: str,
    ftype: str,
    ext: list[tuple[int, int]],
    data,
    levels: Optional[list] = None,
) -> None:
    levels = levels or [None] * len(ext)
    dims = []
    for (lo, hi), lev in zip(ext, levels):
        dname = NC_DIM[lev] if lev else _dim_name(lo, hi)
        if dname not in ds.dimensions:
            ds.createDimension(dname, hi - lo + 1)
        dims.append(dname)
    v = ds.createVariable(
        name, _nc_type(ftype), tuple(reversed(dims)), fill_value=_fill(ftype)
    )
    arr = np.asarray(data)
    if ftype == "logical":
        arr = arr.astype(np.int32)
    if ext:
        v.setncattr("lbounds", np.array([lo for lo, _ in ext], dtype=np.int32))
        v.setncattr("ubounds", np.array([hi for _, hi in ext], dtype=np.int32))
        v[...] = arr.T
    else:
        v.assignValue(arr)


def time_values(
    inv: Inventory, spec: DomainSpec, constants: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The value of every per-step scalar of the case (`[domain.time]`, else TIME_DEFAULTS, the
    constants file, 0), and the ones `[domain.time]` did not set."""
    values: dict[str, Any] = {}
    defaulted: dict[str, Any] = {}
    for s in getattr(inv.unit_test, "time_scalars", []) or []:
        name = s.name.lower()
        if name in spec.time:
            val = spec.time[name]
        else:
            val = TIME_DEFAULTS.get(name, constants.get(name, 0))
        val = float(val) if s.io_type == "real" else int(val)
        values[name] = val
        if name not in spec.time:
            defaulted[name] = val
    return values, defaulted


def _apply_constant_overrides(
    inv: Inventory, path: Path, overrides: dict[str, Any]
) -> None:
    if not overrides:
        return
    import netCDF4

    with netCDF4.Dataset(path, "a") as ds:
        for name, value in overrides.items():
            key = name.lower()
            c = inv.constants.get(key)
            if c is None:
                raise DomainError(
                    f"[domain.constants]: {name} is not a constant of the case"
                )
            if c.rank:
                raise DomainError(
                    f"[domain.constants]: {name} is an array; only scalars can be overridden"
                )
            if key in ds.variables:
                ds.variables[key].assignValue(
                    int(value) if c.type == "logical" else value
                )
            else:
                v = ds.createVariable(key, _nc_type(c.type), ())
                v.assignValue(int(value) if c.type == "logical" else value)
