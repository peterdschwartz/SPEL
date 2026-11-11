"""Synthetic subgrid domains for property-based tests.

A domain is a set of independent sites, each one gridcell with one topounit and one landunit
(`Layout`): a vegetated site (soil or crop landunit) has one column and one patch of its pft, an
urban site (`urban = ["md", ...]`) an urban landunit of that density with the five urban columns
(roof, sunwall, shadewall, impervious and pervious road) and one bare patch each, a lake site
(`lake = true`) a deep-lake landunit with one column and one bare patch. Every site appears
`per_pft` times per set, laid out set by set (set k of every level is a contiguous range, as
duplicateMod lays out `nsets` copies), so the gridcell, topounit and landunit levels have one
element per site and the column and patch levels one per column. The builder writes the two files `elm_init`
reads (`spel-inputs0001.nc`, `spel-constants0001.nc`) for every variable of the inventory:

- topology (subgrid indices, types, flags, weights) from the rules below; a topology field
  without a rule stops the build, so a new routine never runs on a silently wrong hierarchy;
- parameters without a subgrid dimension (pftcon-like arrays, scalars) and the constants from a
  reference capture (`reference`), with `constants` overrides;
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
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np

from spel.proptest.errors import ProptestError
from spel.proptest.fexpr import ExprError
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


class SiteKind(Enum):
    """What a site's landunit is (`[domain]`: `pfts` on `landunit`, `urban`, `lake`)."""

    SOIL = "soil"
    CROP = "crop"
    URBAN = "urban"
    LAKE = "lake"

    @property
    def vegetated(self) -> bool:
        return self in (SiteKind.SOIL, SiteKind.CROP)


class UrbanDensity(Enum):
    """ELM's urban classes (landunit_varcon): tall building district, high and medium density."""

    TBD = "tbd"
    HD = "hd"
    MD = "md"

    @property
    def type_code(self) -> str:
        """The landunit_varcon parameter of its landunit type (a TYPE_CODES key)."""
        return f"isturb_{self.value}"


@dataclass
class DomainSpec:
    pfts: list[int]
    per_pft: int = 1
    nsets: int = 1
    landunit: SiteKind = SiteKind.SOIL  # SOIL | CROP: the landunit of the vegetated sites (`pfts`)
    urban: list[UrbanDensity] = field(default_factory=list)  # urban sites, one per density
    lake: bool = False  # a deep-lake site
    snow_layers: int = 0
    reference: Optional[Path] = (
        None  # input-data dir of a capture: constants, parameters, extents
    )
    constants: dict[str, Any] = field(default_factory=dict)
    time: dict[str, Any] = field(default_factory=dict)
    dims: dict[str, int] = field(
        default_factory=dict
    )  # values for bound names (nlevsno = 5)
    # files: elm_init reads an inputs file | direct: see module doc | None: direct when the case
    # can (made with spel_varMod: shim.direct_init), else files (runner.Session decides)
    init: Optional[str] = None

    @classmethod
    def from_table(
        cls, table: dict, base: Path, input_data: Optional[Path] = None
    ) -> "DomainSpec":
        known = {
            "pfts",
            "per_pft",
            "nsets",
            "landunit",
            "urban",
            "lake",
            "snow_layers",
            "reference",
            "constants",
            "time",
            "dims",
            "init",
        }
        extra = set(table) - known
        if extra:
            raise DomainError(f"[domain]: unknown keys {sorted(extra)}")
        if "pfts" not in table and not table.get("urban") and not table.get("lake"):
            raise DomainError("[domain]: give the sites: `pfts` (vegetated), `urban` and/or `lake`")
        densities = [d.value for d in UrbanDensity]
        urban = table.get("urban", [])
        if isinstance(urban, str) or not isinstance(urban, list):
            raise DomainError(f"[domain]: urban must be a list of densities {densities}, not {urban!r}")
        bad = [u for u in urban if str(u).lower() not in densities]
        if bad or len({str(u).lower() for u in urban}) != len(urban):
            raise DomainError(f"[domain]: urban takes distinct densities of {densities}, not {urban}")
        landunit = str(table.get("landunit", SiteKind.SOIL.value)).lower()
        if landunit not in (SiteKind.SOIL.value, SiteKind.CROP.value):
            raise DomainError(f"[domain]: landunit must be soil or crop, not {landunit!r}")
        ref = table.get("reference")
        ref_path = None
        if ref:
            ref_path = Path(ref)
            if not ref_path.is_absolute():
                cand = (input_data / ref) if input_data is not None else None
                ref_path = cand if cand is not None and cand.is_dir() else (base / ref)
            if not ref_path.is_dir():
                raise DomainError(f"[domain]: reference {ref!r} is not a directory")
        spec = cls(
            pfts=[int(p) for p in table.get("pfts", [])],
            per_pft=int(table.get("per_pft", 1)),
            nsets=int(table.get("nsets", 1)),
            landunit=SiteKind(landunit),
            urban=[UrbanDensity(str(u).lower()) for u in urban],
            lake=bool(table.get("lake", False)),
            snow_layers=int(table.get("snow_layers", 0)),
            reference=ref_path,
            constants=dict(table.get("constants", {})),
            time={k.lower(): v for k, v in table.get("time", {}).items()},
            dims={k.lower(): int(v) for k, v in table.get("dims", {}).items()},
            init=table.get("init"),
        )
        if spec.init is not None and spec.init not in INIT_MODES:
            raise DomainError(f"[domain]: init must be one of {list(INIT_MODES)}, not {spec.init!r}")
        if not isinstance(table.get("lake", False), bool):
            raise DomainError("[domain]: lake is true or false")
        if not (spec.pfts or spec.urban or spec.lake) or spec.per_pft < 1 or spec.nsets < 1:
            raise DomainError(
                "[domain]: need at least one site (pfts, urban, lake), per_pft >= 1 and nsets >= 1"
            )
        return spec

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


# ELM's subgrid type codes (landunit_varcon, column_varcon, pftvarcon). The case's own module
# parameters, when the analysis has them, must agree (`check_type_codes`).
TYPE_CODES = {
    "istsoil": 1, "istcrop": 2, "istdlak": 5, "isturb_tbd": 7, "isturb_hd": 8, "isturb_md": 9,
    "icol_roof": 71, "icol_sunwall": 72, "icol_shadewall": 73, "icol_road_imperv": 74, "icol_road_perv": 75,
    "noveg": 0,
}
# the columns of an urban landunit, in initGridCellsMod's order (set_landunit_urban)
URBAN_COLUMNS = ("icol_roof", "icol_sunwall", "icol_shadewall", "icol_road_imperv", "icol_road_perv")
# The urban column weights in their landunit follow set_landunit_urban for this geometry. The
# geometry itself (`lun_pp%wtlunit_roof`, `wtroad_perv`) is an input a scenario draws, but the
# weights are topology, fixed when the domain is built.
URBAN_WEIGHT_GEOMETRY = {"wtlunit_roof": 0.5, "wtroad_perv": 0.5}
# filterMod members that select the elements of one site kind, and the complements that only add
# the other sites (a routine taking `urbanl` and `nourbanl` works on urban sites)
FILTER_SITES = {
    "urbanl": SiteKind.URBAN, "urbanc": SiteKind.URBAN, "urbanp": SiteKind.URBAN,
    "lakec": SiteKind.LAKE, "lakep": SiteKind.LAKE, "lakesnowc": SiteKind.LAKE, "lakenosnowc": SiteKind.LAKE,
}
FILTER_COMPLEMENTS = {"nourbanl": "urbanl", "nourbanc": "urbanc", "nourbanp": "urbanp",
                      "nolakec": "lakec", "nolakep": "lakep"}
SITE_PATCH_FILTER = {SiteKind.URBAN: "urbanp", SiteKind.LAKE: "lakep"}  # a scenario's `over` for the kind


def landunit_types(kind: SiteKind, codes: Optional[dict[str, int]] = None) -> tuple[int, ...]:
    """The lun_pp%itype values of a site kind."""
    c = {**TYPE_CODES, **(codes or {})}
    if kind is SiteKind.URBAN:
        return tuple(c[d.type_code] for d in UrbanDensity)
    return (c[{SiteKind.SOIL: "istsoil", SiteKind.CROP: "istcrop", SiteKind.LAKE: "istdlak"}[kind]],)


def density_of(ltype: int, codes: Optional[dict[str, int]] = None) -> Optional[UrbanDensity]:
    c = {**TYPE_CODES, **(codes or {})}
    return next((d for d in UrbanDensity if c[d.type_code] == ltype), None)


BOUND_LEVEL = {"g": "g", "t": "t", "l": "l", "c": "c", "p": "p"}  # bounds_type begX/endX -> level
SITE_LEVELS = ("g", "t", "l")  # one element per site
COLUMN_LEVELS = ("c", "p")  # one element per column (every column has one patch)


def _urban_weights(wtlunit_roof: float, wtroad_perv: float) -> tuple[float, ...]:
    canyon = (1.0 - wtlunit_roof) / 3.0
    return (wtlunit_roof, canyon, canyon, canyon * (1.0 - wtroad_perv), canyon * wtroad_perv)


@dataclass(frozen=True)
class Site:
    """One site of a set: a gridcell with one topounit and one landunit, and its columns."""

    kind: SiteKind
    pft: int  # the patches' itype (noveg for urban and lake sites)
    ltype: int  # lun_pp%itype
    ctypes: tuple[int, ...]  # col_pp%itype of each column
    weights: tuple[float, ...]  # each column's weight in the landunit
    label: str  # "pft 1", "urban md", "lake"

    @property
    def vegetated(self) -> bool:
        return self.kind.vegetated


def make_sites(pfts: list[int], per_pft: int = 1, landunit: SiteKind | str = SiteKind.SOIL,
               urban: tuple[UrbanDensity | str, ...] = (), lake: bool = False,
               codes: Optional[dict[str, int]] = None) -> list[Site]:
    """The sites of one set: each pft, urban density and the lake `per_pft` times, in that order.
    landunit and urban also take the enums' values ("crop", "md")."""
    c = {**TYPE_CODES, **(codes or {})}
    out: list[Site] = []
    landunit = SiteKind(landunit)
    if not landunit.vegetated:
        raise DomainError(f"vegetated sites are on a soil or crop landunit, not {landunit.value}")
    crop = landunit is SiteKind.CROP
    for pft in pfts:
        # initGridCellsMod: soil columns are type 1, crop columns istcrop*100 + cft
        site = Site(landunit, int(pft), c["istcrop" if crop else "istsoil"],
                    (c["istcrop"] * 100 + int(pft),) if crop else (1,), (1.0,), f"pft {pft}")
        out += [site] * per_pft
    weights = _urban_weights(**URBAN_WEIGHT_GEOMETRY)
    for density in map(UrbanDensity, urban):
        site = Site(SiteKind.URBAN, c["noveg"], c[density.type_code], tuple(c[k] for k in URBAN_COLUMNS),
                    weights, f"urban {density.value}")
        out += [site] * per_pft
    if lake:
        # set_landunit_wet_ice_lake: the lake column's type is the landunit's
        out += [Site(SiteKind.LAKE, c["noveg"], c["istdlak"], (c["istdlak"],), (1.0,), "lake")] * per_pft
    return out


class Layout:
    """Sites laid out set by set. Gridcell, topounit and landunit element k (0-based) is site k;
    column and patch element k is column k, the columns of a site consecutive. Arrays are per
    element of a level, 0-based; `parent`/`first_child`/`last_child` give Fortran indices."""

    def __init__(self, pfts: list[int] = (), per_pft: int = 1, nsets: int = 1,
                 landunit: SiteKind | str = SiteKind.SOIL, urban: tuple[UrbanDensity | str, ...] = (), lake: bool = False, codes: Optional[dict[str, int]] = None,
                 sites: Optional[list[Site]] = None):
        self.pfts = list(pfts)
        self.per_pft = per_pft
        self.nsets = nsets
        self.set_sites: list[Site] = sites if sites is not None else make_sites(
            list(pfts), per_pft, landunit, tuple(urban), lake, codes)
        if not self.set_sites:
            raise DomainError("the domain has no sites")
        self.sites: list[Site] = self.set_sites * nsets
        ncols = [len(s.ctypes) for s in self.sites]
        self.col_site = np.repeat(np.arange(len(self.sites)), ncols)  # site of each column
        self.site_first = np.concatenate([[0], np.cumsum(ncols)[:-1]]).astype(np.int64)  # its first column
        self.site_ncols = np.asarray(ncols, dtype=np.int64)
        self.col_pos = np.arange(len(self.col_site)) - self.site_first[self.col_site]  # column within its site

    @classmethod
    def of(cls, spec: "DomainSpec", codes: Optional[dict[str, int]] = None) -> "Layout":
        return cls(spec.pfts, spec.per_pft, spec.nsets, spec.landunit, tuple(spec.urban), spec.lake, codes)

    @property
    def per_set_sites(self) -> int:
        return len(self.set_sites)

    def per_set(self, level: str = "p") -> int:
        """Elements of one set at this level."""
        return self.count(level) // self.nsets

    def count(self, level: str) -> int:
        return len(self.sites) if level in SITE_LEVELS else len(self.col_site)

    def sizes(self) -> dict[str, int]:
        return {lev: self.count(lev) for lev in LEVELS}

    def site_of(self, level: str = "p") -> np.ndarray:
        """The site (0-based, over every set) of each element of the level."""
        return np.arange(len(self.sites)) if level in SITE_LEVELS else self.col_site

    def set_of(self, level: str = "p") -> np.ndarray:
        return self.site_of(level) // self.per_set_sites

    def _site_attr(self, level: str, attr) -> np.ndarray:
        return np.asarray([attr(self.sites[k]) for k in range(len(self.sites))])[self.site_of(level)]

    def pft_of(self, level: str = "p") -> np.ndarray:
        """The pft of each element's site; -1 for urban and lake sites (no per-pft rule applies)."""
        return self._site_attr(level, lambda s: s.pft if s.vegetated else -1)

    def kind_of(self, level: str = "p") -> list[SiteKind]:
        site = self.site_of(level)
        return [self.sites[k].kind for k in site]

    def is_kind(self, level: str, *kinds: SiteKind) -> np.ndarray:
        """Whether each element of the level belongs to a site of one of these kinds."""
        return np.asarray([kind in kinds for kind in self.kind_of(level)], dtype=bool)

    def one_to_one(self, a: str, b: str) -> bool:
        """Whether the two levels have the same elements (element k of one is element k of the other)."""
        return (a in SITE_LEVELS) == (b in SITE_LEVELS) or bool(np.all(self.site_ncols == 1))

    def parent(self, level: str, coarser: str) -> np.ndarray:
        """Fortran index of each element's ancestor at a coarser (or the same) level."""
        if coarser in SITE_LEVELS:
            return self.site_of(level) + 1
        return np.arange(1, self.count(level) + 1)

    def first_child(self, level: str, finer: str) -> np.ndarray:
        if level in SITE_LEVELS and finer in COLUMN_LEVELS:
            return self.site_first + 1
        return np.arange(1, self.count(level) + 1)

    def last_child(self, level: str, finer: str) -> np.ndarray:
        if level in SITE_LEVELS and finer in COLUMN_LEVELS:
            return self.site_first + self.site_ncols
        return np.arange(1, self.count(level) + 1)

    def children(self, level: str, finer: str) -> np.ndarray:
        if level in SITE_LEVELS and finer in COLUMN_LEVELS:
            return self.site_ncols.copy()
        return np.ones(self.count(level), dtype=np.int64)

    def row_permutation(self, level: str, site_perm: np.ndarray) -> np.ndarray:
        """A permutation of the first `len(site_perm)` sites as rows of a level: row k of the
        result holds what row `perm[k]` held. Sites are only exchanged with sites of the same
        kind, so their column blocks have the same length."""
        site_perm = np.asarray(site_perm, dtype=np.int64)
        if level in SITE_LEVELS:
            return site_perm
        return np.concatenate([np.arange(self.site_first[s], self.site_first[s] + self.site_ncols[s])
                               for s in site_perm]).astype(np.int64)

    def describe(self, level: str, k: int) -> str:
        """Element k (Fortran index) of a level, for reports."""
        site = int(self.site_of(level)[k - 1])
        s = self.sites[site]
        text = f"set {site // self.per_set_sites}, {s.label}"
        if level in COLUMN_LEVELS and len(s.ctypes) > 1:
            text += f", column {int(self.col_pos[k - 1]) + 1} of {len(s.ctypes)} (itype {s.ctypes[self.col_pos[k - 1]]})"
        return text


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

    @property
    def path_arg(self) -> str:
        """What spel_py_init takes: the directory with a trailing slash."""
        return str(self.directory) + "/"


# -- topology rules


# Subgrid structure fields by what they hold. A field of a routine that is in none of these (and
# has no rule in `topology_value`) stops the build.
LEVEL_OF_FIELD = {"gridcell": "g", "topounit": "t", "landunit": "l", "column": "c", "patch": "p",
                  "topo_grc_ind": "g", "grc_ind": "g"}  # the element's ancestor (or first descendant)
FIRST_CHILD = {"topi": "t", "lndi": "l", "coli": "c", "pfti": "p"}
LAST_CHILD = {"topf": "t", "lndf": "l", "colf": "c", "pftf": "p"}
TOPOLOGY_INDEX = set(LEVEL_OF_FIELD) | set(FIRST_CHILD) | set(LAST_CHILD)
TOPOLOGY_COUNT = {"npfts": "p", "ncolumns": "c", "nlandunits": "l", "ntopounits": "t"}
TOPOLOGY_WEIGHT = {"wtgcell", "wtlunit", "wtcol", "wttopounit", "wtgcell_save"}
TOPOLOGY_TRUE = {"active"}
TOPOLOGY_FALSE = {"glcmecpoi", "is_fates"}
# logical fields that hold for the elements of sites of these kinds
TOPOLOGY_KIND = {
    "lakpoi": (SiteKind.LAKE,), "is_lake": (SiteKind.LAKE,), "urbpoi": (SiteKind.URBAN,),
    "ifspecial": (SiteKind.URBAN, SiteKind.LAKE),
    "is_soil": (SiteKind.SOIL,), "is_on_soil_col": (SiteKind.SOIL,),
    "is_crop": (SiteKind.CROP,), "is_on_crop_col": (SiteKind.CROP,),
}


def check_type_codes(params: dict[str, Any]) -> dict[str, int]:
    """TYPE_CODES, checked against the case's module parameters where the analysis has them."""
    for name, value in TYPE_CODES.items():
        have = params.get(name)
        if isinstance(have, int) and have != value:
            raise DomainError(f"the case has {name} = {have}; the synthetic domain assumes {value} "
                              "(update TYPE_CODES in spel/proptest/domain.py)")
    return dict(TYPE_CODES)


def _ancestor_or_descendant(layout: "Layout", lev: str, target: str) -> np.ndarray:
    if LEVELS.index(target) <= LEVELS.index(lev):
        return layout.parent(lev, target)
    return layout.first_child(lev, target)


def topology_value(
    var: VarInfo, layout: Layout, spec: DomainSpec, params: dict[str, Any]
) -> np.ndarray:
    """Values of a subgrid structure field, per element of its level."""
    lev = var.level
    f = var.field
    n = layout.count(lev)
    if f in LEVEL_OF_FIELD:
        return _ancestor_or_descendant(layout, lev, LEVEL_OF_FIELD[f]).astype(np.int32)
    if f in FIRST_CHILD:
        return layout.first_child(lev, FIRST_CHILD[f]).astype(np.int32)
    if f in LAST_CHILD:
        return layout.last_child(lev, LAST_CHILD[f]).astype(np.int32)
    if f in TOPOLOGY_COUNT:
        return layout.children(lev, TOPOLOGY_COUNT[f]).astype(np.int32)
    if f in TOPOLOGY_WEIGHT:
        if f == "wtcol" or lev in SITE_LEVELS:  # one patch per column, one landunit per gridcell
            return np.ones(n)
        w = np.asarray([wt for s in layout.sites for wt in s.weights])
        return w if lev in COLUMN_LEVELS else np.ones(n)
    if f in TOPOLOGY_TRUE:
        return np.ones(n, dtype=bool)
    if f in TOPOLOGY_FALSE:
        return np.zeros(n, dtype=bool)
    if f in TOPOLOGY_KIND:
        return layout.is_kind(lev, *TOPOLOGY_KIND[f])
    if f == "hydrologically_active" and lev in COLUMN_LEVELS:
        # column_varcon::is_hydrologically_active: soil and crop landunits, the pervious road
        ctype = np.asarray([ct for s in layout.sites for ct in s.ctypes])[np.arange(n)]
        return layout.is_kind(lev, SiteKind.SOIL, SiteKind.CROP) | (ctype == TYPE_CODES["icol_road_perv"])
    if f == "snl":
        return np.full(n, -spec.snow_layers, dtype=np.int32)
    if f == "itype" and lev in ("l", "c", "p"):
        if lev == "l":
            return np.asarray([s.ltype for s in layout.sites], dtype=np.int32)
        if lev == "c":
            return np.asarray([ct for s in layout.sites for ct in s.ctypes], dtype=np.int32)
        return np.asarray([s.pft for s in layout.sites for _ in s.ctypes], dtype=np.int32)
    raise DomainError(
        f"no synthetic-domain rule for topology field {var.name} (level {lev}); "
        "add one to spel/proptest/domain.py (the TOPOLOGY_* sets or topology_value)"
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
    layout = Layout.of(spec, check_type_codes(inv.parameters))

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
    _apply_constant_overrides(inv, const_out, spec.constants)
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
    )


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
    if var.kind == "topology" or (
        var.instance in _physical_instances(inv) and var.type in ("integer", "logical")
    ):
        val = topology_value(var, layout, spec, inv.parameters)
        if var.rank != 1:
            raise DomainError(f"{var.name}: topology field of rank {var.rank}")
        return val, ""
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
