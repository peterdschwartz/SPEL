"""ELM's subgrid hierarchy for synthetic domains: declarations and their layout.

A domain is declared as gridcells, each with topounits, landunits, columns and patches, every
element with a weight in its parent (`Gridcell`, `Topounit`, `Landunit`, `Column`, `Patch`). The
constructors follow initGridCellsMod:

- `Soil(1, Patch(13, 0.4))`: a natural-vegetation landunit (istsoil) with ONE soil column holding
  one patch per pft;
- `Crop(17, 19)`: a crop landunit (istcrop) with one column per CFT (itype istcrop*100 + cft),
  one patch each; a CFT's weight is its column's weight in the landunit;
- `Urban("md")`: an urban landunit of that density with the five urban columns (roof, sunwall,
  shadewall, impervious and pervious road; set_landunit_urban's weights) and one bare patch each;
- `Lake()`: a deep-lake landunit with one column and one bare patch.

Weights left out share what the given ones leave, equally; the weights of every element's
children must sum to 1. An element is active when its topounit is active (weight > 0, as
set_topounit decides with several topounits) and its own weight is > 0 (subgridWeightsMod
without its special cases).

`Layout` lays the declared gridcells out as ELM does: depth first, every element's children
contiguous, the `copies` of a gridcell next to each other, and the whole set repeated `nsets`
times (set k of every level is a contiguous range, as duplicateMod copies sets).

In TOML a gridcell is `[[domain.gridcell]]` with `landunits = [...]` (one topounit) or
`topounits = [{ weight, elevation, landunits = [...] }, ...]`; `to_table`/`from_table` convert.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Sequence

import numpy as np

from spel.proptest.errors import ProptestError

LEVELS = ("g", "t", "l", "c", "p")
LEVEL_NAMES = {"g": "gridcell", "t": "topounit", "l": "landunit", "c": "column", "p": "patch"}
WEIGHT_TOL = 1e-9
ISPVAL = -9999  # elm_varcon::ispval
MAX_LUNIT = 12  # landunit_varcon::max_lunit


class HierarchyError(ProptestError):
    pass


class LandunitKind(Enum):
    SOIL = "soil"
    CROP = "crop"
    URBAN = "urban"
    LAKE = "lake"

    @property
    def vegetated(self) -> bool:
        return self in (LandunitKind.SOIL, LandunitKind.CROP)


class UrbanDensity(Enum):
    """ELM's urban classes (landunit_varcon): tall building district, high and medium density."""

    TBD = "tbd"
    HD = "hd"
    MD = "md"

    @property
    def type_code(self) -> str:
        """The landunit_varcon parameter of its landunit type (a TYPE_CODES key)."""
        return f"isturb_{self.value}"


# ELM's subgrid type codes (landunit_varcon, column_varcon, pftvarcon). The case's own module
# parameters, when the analysis has them, must agree (`check_type_codes`).
TYPE_CODES = {
    "istsoil": 1, "istcrop": 2, "istdlak": 5, "isturb_tbd": 7, "isturb_hd": 8, "isturb_md": 9,
    "icol_roof": 71, "icol_sunwall": 72, "icol_shadewall": 73, "icol_road_imperv": 74, "icol_road_perv": 75,
    "noveg": 0,
}
# the columns of an urban landunit, in initGridCellsMod's order (set_landunit_urban)
URBAN_COLUMNS = ("icol_roof", "icol_sunwall", "icol_shadewall", "icol_road_imperv", "icol_road_perv")
SOIL_COLUMN = 1  # initGridCellsMod: the natural-vegetation column's type


def check_type_codes(params: dict[str, Any]) -> None:
    """TYPE_CODES agree with the case's module parameters where the analysis has them."""
    for name, value in TYPE_CODES.items():
        have = params.get(name)
        if isinstance(have, int) and have != value:
            raise HierarchyError(f"the case has {name} = {have}; the synthetic domain assumes {value} "
                                 "(update TYPE_CODES in spel/proptest/hierarchy.py)")


def ltype_of(kind: LandunitKind, density: Optional[UrbanDensity]) -> int:
    if kind is LandunitKind.URBAN:
        assert density is not None
        return TYPE_CODES[density.type_code]
    return TYPE_CODES[{LandunitKind.SOIL: "istsoil", LandunitKind.CROP: "istcrop", LandunitKind.LAKE: "istdlak"}[kind]]


def column_role(itype: int) -> str:
    """A column type's name: soil, crop_<cft>, lake, roof, sunwall, ... ."""
    for key in URBAN_COLUMNS:
        if TYPE_CODES[key] == itype:
            return key.removeprefix("icol_")
    if itype == SOIL_COLUMN:
        return "soil"
    if itype == TYPE_CODES["istdlak"]:
        return "lake"
    if itype > TYPE_CODES["istcrop"] * 100:
        return f"crop_{itype - TYPE_CODES['istcrop'] * 100}"
    return str(itype)


# -- declarations


@dataclass(frozen=True)
class Patch:
    pft: int
    weight: Optional[float] = None  # in its column

    def to_table(self) -> Any:
        return self.pft if self.weight is None else {"pft": self.pft, "weight": self.weight}

    @classmethod
    def from_table(cls, value: Any, where: str) -> "Patch":
        if isinstance(value, bool) or not isinstance(value, (int, dict)):
            raise HierarchyError(f"{where}: a patch is a pft number or {{ pft, weight }}, not {value!r}")
        if isinstance(value, int):
            return cls(value)
        _keys(value, {"pft", "weight"}, where)
        if "pft" not in value:
            raise HierarchyError(f"{where}: `pft` is required")
        return cls(int(value["pft"]), _weight(value, where))


@dataclass(frozen=True)
class Column:
    itype: int
    patches: tuple[Patch, ...]
    weight: Optional[float] = None  # in its landunit

    def to_table(self) -> dict:
        out: dict[str, Any] = {"itype": self.itype, "patches": [p.to_table() for p in self.patches]}
        if self.weight is not None:
            out["weight"] = self.weight
        return out

    @classmethod
    def from_table(cls, value: Any, where: str, itype: int) -> "Column":
        if not isinstance(value, dict):
            raise HierarchyError(f"{where}: a column is a table {{ patches = [...], weight }}")
        _keys(value, {"itype", "patches", "weight"}, where)
        patches = tuple(Patch.from_table(p, f"{where}.patches[{k}]") for k, p in enumerate(value.get("patches", [])))
        return cls(int(value.get("itype", itype)), patches, _weight(value, where))


@dataclass(frozen=True)
class Landunit:
    kind: LandunitKind
    columns: tuple[Column, ...]
    weight: Optional[float] = None  # in its topounit
    density: Optional[UrbanDensity] = None  # urban

    @property
    def ltype(self) -> int:
        return ltype_of(self.kind, self.density)

    def to_table(self) -> dict:
        out: dict[str, Any] = {"kind": self.kind.value}
        if self.kind is LandunitKind.SOIL:
            if len(self.columns) == 1 and self.columns[0].weight is None and self.columns[0].itype == SOIL_COLUMN:
                out["patches"] = [p.to_table() for p in self.columns[0].patches]
            else:
                out["columns"] = [c.to_table() for c in self.columns]
        elif self.kind is LandunitKind.CROP:
            out["cfts"] = [Patch(c.patches[0].pft, c.weight).to_table() for c in self.columns]
        elif self.kind is LandunitKind.URBAN:
            assert self.density is not None
            out["density"] = self.density.value
            roof, road_perv = _urban_geometry(self.columns)
            if (roof, road_perv) != (0.5, 0.5):
                out.update(roof=roof, road_perv=road_perv)
        if self.weight is not None:
            out["weight"] = self.weight
        return out

    @classmethod
    def from_table(cls, value: Any, where: str) -> "Landunit":
        if not isinstance(value, dict) or "kind" not in value:
            raise HierarchyError(f"{where}: a landunit is a table with `kind` ({[k.value for k in LandunitKind]})")
        try:
            kind = LandunitKind(str(value["kind"]).lower())
        except ValueError:
            raise HierarchyError(f"{where}: kind must be one of {[k.value for k in LandunitKind]}, "
                                 f"not {value['kind']!r}") from None
        weight = _weight(value, where)
        if kind is LandunitKind.SOIL:
            _keys(value, {"kind", "weight", "patches", "columns"}, where)
            if ("patches" in value) == ("columns" in value):
                raise HierarchyError(f"{where}: a soil landunit takes `patches` (one column) or `columns`")
            if "patches" in value:
                return Soil(*(Patch.from_table(p, f"{where}.patches[{k}]") for k, p in enumerate(value["patches"])),
                            weight=weight)
            cols = tuple(Column.from_table(c, f"{where}.columns[{k}]", SOIL_COLUMN) for k, c in enumerate(value["columns"]))
            return _checked(cls(kind, cols, weight), where)
        if kind is LandunitKind.CROP:
            _keys(value, {"kind", "weight", "cfts"}, where)
            return Crop(*(Patch.from_table(p, f"{where}.cfts[{k}]") for k, p in enumerate(value.get("cfts", []))),
                        weight=weight)
        if kind is LandunitKind.URBAN:
            _keys(value, {"kind", "weight", "density", "roof", "road_perv"}, where)
            try:
                density = UrbanDensity(str(value.get("density", "")).lower())
            except ValueError:
                raise HierarchyError(f"{where}: density must be one of {[d.value for d in UrbanDensity]}") from None
            return Urban(density, weight=weight, roof=float(value.get("roof", 0.5)),
                         road_perv=float(value.get("road_perv", 0.5)))
        _keys(value, {"kind", "weight"}, where)
        return Lake(weight=weight)


@dataclass(frozen=True)
class Topounit:
    landunits: tuple[Landunit, ...]
    weight: Optional[float] = None  # in its gridcell; 0: inactive
    elevation: Optional[float] = None  # use_IM2_hillslope_hydrology: downhill_ti from these

    def to_table(self) -> dict:
        out: dict[str, Any] = {}
        if self.weight is not None:
            out["weight"] = self.weight
        if self.elevation is not None:
            out["elevation"] = self.elevation
        out["landunits"] = [lu.to_table() for lu in self.landunits]
        return out

    @classmethod
    def from_table(cls, value: Any, where: str) -> "Topounit":
        if not isinstance(value, dict):
            raise HierarchyError(f"{where}: a topounit is a table {{ weight, elevation, landunits = [...] }}")
        _keys(value, {"weight", "elevation", "landunits"}, where)
        lus = tuple(Landunit.from_table(x, f"{where}.landunits[{k}]") for k, x in enumerate(value.get("landunits", [])))
        elev = value.get("elevation")
        return cls(lus, _weight(value, where), None if elev is None else float(elev))


@dataclass(frozen=True)
class Gridcell:
    topounits: tuple[Topounit, ...]
    copies: int = 1
    name: str = ""  # reports, queries; relations permute gridcells of the same name

    @classmethod
    def of(cls, *landunits: Landunit, copies: int = 1, name: str = "") -> "Gridcell":
        """A gridcell with one topounit."""
        return cls((Topounit(tuple(landunits)),), copies, name)

    @property
    def label(self) -> str:
        if self.name:
            return self.name
        return "+".join(lu.kind.value if lu.kind is not LandunitKind.URBAN else f"urban {lu.density.value}"
                        for t in self.topounits for lu in t.landunits)

    def to_table(self) -> dict:
        out: dict[str, Any] = {}
        if self.name:
            out["name"] = self.name
        if self.copies != 1:
            out["copies"] = self.copies
        if len(self.topounits) == 1 and self.topounits[0].weight is None and self.topounits[0].elevation is None:
            out["landunits"] = [lu.to_table() for lu in self.topounits[0].landunits]
        else:
            out["topounits"] = [t.to_table() for t in self.topounits]
        return out

    @classmethod
    def from_table(cls, value: Any, where: str) -> "Gridcell":
        if not isinstance(value, dict):
            raise HierarchyError(f"{where}: a gridcell is a table")
        _keys(value, {"name", "copies", "landunits", "topounits"}, where)
        if ("landunits" in value) == ("topounits" in value):
            raise HierarchyError(f"{where}: give `landunits` (one topounit) or `topounits`")
        if "landunits" in value:
            tops = (Topounit(tuple(Landunit.from_table(x, f"{where}.landunits[{k}]")
                                   for k, x in enumerate(value["landunits"]))),)
        else:
            tops = tuple(Topounit.from_table(t, f"{where}.topounits[{k}]") for k, t in enumerate(value["topounits"]))
        g = cls(tops, int(value.get("copies", 1)), str(value.get("name", "")))
        return g.checked(where)

    def checked(self, where: str = "gridcell") -> "Gridcell":
        """The gridcell, after checking its structure and that the weights can sum to 1."""
        if self.copies < 1:
            raise HierarchyError(f"{where}: copies must be >= 1")
        if not self.topounits:
            raise HierarchyError(f"{where}: no topounit")
        resolve_weights([t.weight for t in self.topounits], f"{where}.topounits")
        for k, t in enumerate(self.topounits):
            tw = f"{where}.topounit[{k}]"
            if not t.landunits:
                raise HierarchyError(f"{tw}: no landunit")
            resolve_weights([lu.weight for lu in t.landunits], f"{tw}.landunits")
            for j, lu in enumerate(t.landunits):
                _checked(lu, f"{tw}.landunit[{j}]")
        return self


# -- constructors (initGridCellsMod)


def _patch(p: Patch | int) -> Patch:
    return p if isinstance(p, Patch) else Patch(int(p))


def Soil(*patches: Patch | int, weight: Optional[float] = None) -> Landunit:
    """istsoil: one column (type 1) with one patch per pft."""
    return Landunit(LandunitKind.SOIL, (Column(SOIL_COLUMN, tuple(_patch(p) for p in patches)),), weight)


def Crop(*cfts: Patch | int, weight: Optional[float] = None) -> Landunit:
    """istcrop: one column per CFT (itype istcrop*100 + cft) with one patch; a CFT's weight is its
    column's weight in the landunit."""
    cols = []
    for c in map(_patch, cfts):
        cols.append(Column(TYPE_CODES["istcrop"] * 100 + c.pft, (Patch(c.pft),), c.weight))
    return Landunit(LandunitKind.CROP, tuple(cols), weight)


def Urban(density: UrbanDensity | str, weight: Optional[float] = None, roof: float = 0.5,
          road_perv: float = 0.5) -> Landunit:
    """An urban landunit: the five columns with set_landunit_urban's weights for this roof
    fraction (wtlunit_roof) and pervious-road fraction (wtroad_perv), one bare patch each."""
    density = UrbanDensity(density.lower() if isinstance(density, str) else density)
    weights = _urban_weights(roof, road_perv)
    cols = tuple(Column(TYPE_CODES[k], (Patch(TYPE_CODES["noveg"]),), w) for k, w in zip(URBAN_COLUMNS, weights))
    return Landunit(LandunitKind.URBAN, cols, weight, density)


def Lake(weight: Optional[float] = None) -> Landunit:
    """istdlak: one column of the landunit's type (set_landunit_wet_ice_lake), one bare patch."""
    return Landunit(LandunitKind.LAKE, (Column(TYPE_CODES["istdlak"], (Patch(TYPE_CODES["noveg"]),)),), weight)


def _urban_weights(roof: float, road_perv: float) -> tuple[float, ...]:
    if not (0.0 <= roof <= 1.0 and 0.0 <= road_perv <= 1.0):
        raise HierarchyError(f"urban: roof ({roof}) and road_perv ({road_perv}) are fractions in [0, 1]")
    canyon = (1.0 - roof) / 3.0
    return (roof, canyon, canyon, canyon * (1.0 - road_perv), canyon * road_perv)


def _urban_geometry(columns: Sequence[Column]) -> tuple[float, float]:
    """(roof, road_perv) back from the column weights."""
    w = [c.weight or 0.0 for c in columns]
    road = w[3] + w[4]
    return round(w[0], 12), round(w[4] / road, 12) if road > 0 else 0.5


# -- weights


def resolve_weights(weights: Sequence[Optional[float]], where: str) -> list[float]:
    """Given weights, the missing ones sharing the rest equally; the result sums to 1."""
    given = [w for w in weights if w is not None]
    if any(w < 0 for w in given):
        raise HierarchyError(f"{where}: negative weight in {list(weights)}")
    rest = 1.0 - sum(given)
    missing = len(weights) - len(given)
    if rest < -WEIGHT_TOL or (missing == 0 and abs(rest) > WEIGHT_TOL) or (missing and rest <= WEIGHT_TOL):
        raise HierarchyError(f"{where}: weights {list(weights)} must sum to 1 (missing ones share the rest)")
    return [w if w is not None else rest / missing for w in weights]


def _checked(lu: Landunit, where: str) -> Landunit:
    if not lu.columns:
        raise HierarchyError(f"{where}: no column")
    resolve_weights([c.weight for c in lu.columns], f"{where}.columns")
    for k, c in enumerate(lu.columns):
        if not c.patches:
            raise HierarchyError(f"{where}.column[{k}]: no patch")
        resolve_weights([p.weight for p in c.patches], f"{where}.column[{k}].patches")
    return lu


def _weight(value: dict, where: str) -> Optional[float]:
    w = value.get("weight")
    if w is None:
        return None
    if isinstance(w, bool) or not isinstance(w, (int, float)):
        raise HierarchyError(f"{where}: weight must be a number, not {w!r}")
    return float(w)


def _keys(value: dict, known: set[str], where: str) -> None:
    extra = set(value) - known
    if extra:
        raise HierarchyError(f"{where}: unknown keys {sorted(extra)} (known: {sorted(known)})")


# -- layout


@dataclass
class _Level:
    """The elements of one level, 0-based: parent (index in the next coarser level) and the
    weight in it; the declaration each element comes from."""

    parent: list[int] = field(default_factory=list)
    weight: list[float] = field(default_factory=list)
    decl: list[Any] = field(default_factory=list)


class Layout:
    """The declared gridcells laid out as ELM lays out a clump. All arrays are per element of a
    level; indices given to and returned by the methods are Fortran (1-based) unless named
    `*_of` (0-based positions)."""

    def __init__(self, gridcells: Sequence[Gridcell], nsets: int = 1):
        if not gridcells:
            raise HierarchyError("the domain has no gridcell")
        if nsets < 1:
            raise HierarchyError("nsets must be >= 1")
        self.declared: list[Gridcell] = [g.checked(f"gridcell[{k}]") for k, g in enumerate(gridcells)]
        self.nsets = nsets
        # one set: every declared gridcell `copies` times, in order
        self.set_gridcells: list[Gridcell] = [g for g in self.declared for _ in range(g.copies)]
        self.unit_of_gridcell: list[int] = []  # position of each gridcell within its set
        self.levels: dict[str, _Level] = {lev: _Level() for lev in LEVELS}
        self.topo_number: list[int] = []  # topounit number within its gridcell (1..n)
        self.elevation: list[Optional[float]] = []
        for _ in range(nsets):
            for unit, g in enumerate(self.set_gridcells):
                self._add_gridcell(g, unit)
        self._arrays()

    def _add_gridcell(self, g: Gridcell, unit: int) -> None:
        L = self.levels
        gi = len(L["g"].parent)
        L["g"].parent.append(-1)
        L["g"].weight.append(1.0)
        L["g"].decl.append(g)
        self.unit_of_gridcell.append(unit)
        for k, (t, tw) in enumerate(zip(g.topounits, resolve_weights([t.weight for t in g.topounits], "topounits"))):
            ti = len(L["t"].parent)
            L["t"].parent.append(gi)
            L["t"].weight.append(tw)
            L["t"].decl.append(t)
            self.topo_number.append(k + 1)
            self.elevation.append(t.elevation)
            for lu, lw in zip(t.landunits, resolve_weights([lu.weight for lu in t.landunits], "landunits")):
                li = len(L["l"].parent)
                L["l"].parent.append(ti)
                L["l"].weight.append(lw)
                L["l"].decl.append(lu)
                for c, cw in zip(lu.columns, resolve_weights([c.weight for c in lu.columns], "columns")):
                    ci = len(L["c"].parent)
                    L["c"].parent.append(li)
                    L["c"].weight.append(cw)
                    L["c"].decl.append(c)
                    for p, pw in zip(c.patches, resolve_weights([p.weight for p in c.patches], "patches")):
                        L["p"].parent.append(ci)
                        L["p"].weight.append(pw)
                        L["p"].decl.append(p)

    def _arrays(self) -> None:
        self._parent = {lev: np.asarray(self.levels[lev].parent, dtype=np.int64) for lev in LEVELS}
        self._wparent = {lev: np.asarray(self.levels[lev].weight, dtype=float) for lev in LEVELS}
        # first/last child (0-based) of each element in the next finer level
        self._first: dict[str, np.ndarray] = {}
        self._last: dict[str, np.ndarray] = {}
        for coarse, fine in zip(LEVELS[:-1], LEVELS[1:]):
            n = self.count(coarse)
            par = self._parent[fine]
            counts = np.bincount(par, minlength=n)
            first = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
            self._first[coarse], self._last[coarse] = first, first + counts - 1
        self._ltype = np.asarray([lu.ltype for lu in self.levels["l"].decl], dtype=np.int64)
        self._kind_l = [lu.kind for lu in self.levels["l"].decl]

    # -- sizes and positions
    @property
    def per_set_gridcells(self) -> int:
        return len(self.set_gridcells)

    def count(self, level: str) -> int:
        return len(self.levels[level].parent)

    def sizes(self) -> dict[str, int]:
        return {lev: self.count(lev) for lev in LEVELS}

    def per_set(self, level: str) -> int:
        return self.count(level) // self.nsets

    @property
    def max_topounits(self) -> int:
        return max(len(g.topounits) for g in self.declared)

    # -- hierarchy
    def ancestor_of(self, level: str, coarser: str) -> np.ndarray:
        """0-based index of each element's ancestor at a coarser (or the same) level."""
        idx = np.arange(self.count(level))
        lev = level
        while LEVELS.index(lev) > LEVELS.index(coarser):
            idx = self._parent[lev][idx]
            lev = LEVELS[LEVELS.index(lev) - 1]
        return idx

    def parent(self, level: str, coarser: str) -> np.ndarray:
        """Fortran index of each element's ancestor at a coarser (or the same) level."""
        return self.ancestor_of(level, coarser) + 1

    def first_child(self, level: str, finer: str) -> np.ndarray:
        """Fortran index of each element's first descendant at a finer (or the same) level."""
        idx = np.arange(self.count(level))
        lev = level
        while LEVELS.index(lev) < LEVELS.index(finer):
            idx = self._first[lev][idx]
            lev = LEVELS[LEVELS.index(lev) + 1]
        return idx + 1

    def last_child(self, level: str, finer: str) -> np.ndarray:
        idx = np.arange(self.count(level))
        lev = level
        while LEVELS.index(lev) < LEVELS.index(finer):
            idx = self._last[lev][idx]
            lev = LEVELS[LEVELS.index(lev) + 1]
        return idx + 1

    def children(self, level: str, finer: str) -> np.ndarray:
        return self.last_child(level, finer) - self.first_child(level, finer) + 1

    def one_to_one(self, a: str, b: str) -> bool:
        """Whether two levels have the same elements (a tree: equal counts)."""
        return self.count(a) == self.count(b)

    def descendants(self, level: str, k: int, finer: str) -> np.ndarray:
        """0-based rows of `finer` under element k (Fortran) of `level`: a contiguous range."""
        return np.arange(self.first_child(level, finer)[k - 1] - 1, self.last_child(level, finer)[k - 1])

    # -- gridcells and sets
    def gridcell_of(self, level: str) -> np.ndarray:
        """0-based gridcell of each element (the unit relations permute)."""
        return self.ancestor_of(level, "g")

    def set_of(self, level: str) -> np.ndarray:
        return self.gridcell_of(level) // self.per_set_gridcells

    def name_of(self, level: str) -> list[str]:
        """The label of each element's declaring gridcell (its name, or its landunits)."""
        g = self.gridcell_of(level)
        labels = [gc.label for gc in self.levels["g"].decl]
        return [labels[k] for k in g]

    def row_permutation(self, level: str, gridcell_perm: Sequence[int]) -> np.ndarray:
        """A permutation of the first `len(gridcell_perm)` gridcells as rows of a level: row k of
        the result holds what row `perm[k]` held. Gridcells are only exchanged with gridcells of
        the same declaration, so their blocks have the same length."""
        perm = np.asarray(gridcell_perm, dtype=np.int64)
        first = self.first_child("g", level) - 1
        last = self.last_child("g", level) - 1
        return np.concatenate([np.arange(first[s], last[s] + 1) for s in perm]).astype(np.int64)

    # -- element attributes
    def weight(self, level: str, relative_to: str) -> np.ndarray:
        """Each element's weight in its ancestor at `relative_to` (wtcol, wtlunit, wttopounit, wtgcell)."""
        w = np.ones(self.count(level))
        idx = np.arange(self.count(level))
        lev = level
        while LEVELS.index(lev) > LEVELS.index(relative_to):
            w = w * self._wparent[lev][idx]
            idx = self._parent[lev][idx]
            lev = LEVELS[LEVELS.index(lev) - 1]
        return w

    def active(self, level: str) -> np.ndarray:
        """Active: the topounit's weight > 0 and every weight from the element up to it > 0."""
        if level == "g":
            return np.ones(self.count("g"), dtype=bool)
        top = self.weight(level, "g") > 0.0
        return top

    def kind_of(self, level: str) -> list[Optional[LandunitKind]]:
        """The landunit kind of each element (None for gridcells and topounits)."""
        if level in ("g", "t"):
            return [None] * self.count(level)
        lu = self.ancestor_of(level, "l")
        return [self._kind_l[k] for k in lu]

    def is_kind(self, level: str, *kinds: LandunitKind) -> np.ndarray:
        return np.asarray([k in kinds for k in self.kind_of(level)], dtype=bool)

    def ltype(self, level: str) -> np.ndarray:
        return self._ltype[self.ancestor_of(level, "l")]

    def itype(self, level: str) -> np.ndarray:
        """lun_pp/col_pp/veg_pp%itype."""
        if level == "l":
            return self._ltype.copy()
        if level == "c":
            return np.asarray([c.itype for c in self.levels["c"].decl], dtype=np.int64)
        if level == "p":
            return np.asarray([p.pft for p in self.levels["p"].decl], dtype=np.int64)
        raise HierarchyError(f"no itype at the {LEVEL_NAMES[level]} level")

    def pft_of(self, level: str) -> Optional[np.ndarray]:
        """The pft of each patch (per-pft rules apply to patch-level variables only)."""
        return self.itype("p") if level == "p" else None

    def topo_index(self) -> np.ndarray:
        """top_pp%topo_grc_ind: each topounit's number within its gridcell."""
        return np.asarray(self.topo_number, dtype=np.int64)

    def downhill(self) -> np.ndarray:
        """top_pp%downhill_ti (Fortran topounit indices, -1: none) as set_topounit computes it
        under use_IM2_hillslope_hydrology from the topounits' elevations; -1 where a gridcell has
        one topounit (initSubgridMod's add_topounit sets -1 for every topounit, set_topounit only
        overwrites it for several). ELM's loop keeps `dn_index` from the previous topounit when a
        topounit has no lower neighbour; so does this (bit-for-bit with ELM, quirk included)."""
        out = np.full(self.count("t"), -1, dtype=np.int64)
        for g in range(self.count("g")):
            ts = list(range(self._first["g"][g], self._last["g"][g] + 1))
            if len(ts) < 2:
                continue
            elev = [self.elevation[t] for t in ts]
            if any(e is None for e in elev):
                raise HierarchyError(f"gridcell {g + 1}: declare every topounit's elevation (downhill_ti)")
            min_elev = min(elev)
            dn_index = -1
            for t1 in ts:
                dn_elev = min_elev
                for t2 in ts:
                    if self.elevation[t2] < self.elevation[t1] and self.elevation[t2] >= dn_elev:
                        dn_elev = self.elevation[t2]
                        dn_index = t2 + 1
                out[t1] = dn_index
        return out

    def landunit_indices(self, level: str, max_lunit: int = MAX_LUNIT) -> np.ndarray:
        """grc_pp/top_pp%landunit_indices(ltype, element): the first landunit of each type under
        the element (elm_ptrs_compdown), ispval where there is none. Shape (max_lunit, n)."""
        n = self.count(level)
        out = np.full((max_lunit, n), ISPVAL, dtype=np.int64)
        owner = self.ancestor_of("l", level)
        for li in range(self.count("l") - 1, -1, -1):  # backwards: the first landunit of a type wins
            t = int(self._ltype[li])
            if 1 <= t <= max_lunit:
                out[t - 1, owner[li]] = li + 1
        return out

    def describe(self, level: str, k: int) -> str:
        """Element k (Fortran index) of a level, for reports."""
        j = k - 1
        g = int(self.gridcell_of(level)[j])
        parts = [f"set {g // self.per_set_gridcells}", self.levels["g"].decl[g].label]
        if level != "g":
            t = int(self.ancestor_of(level, "t")[j])
            ntop = int(self.children("g", "t")[g])
            if ntop > 1:
                parts.append(f"topounit {self.topo_number[t]}/{ntop}")
        if level in ("l", "c", "p"):
            lu = self.levels["l"].decl[int(self.ancestor_of(level, "l")[j])]
            parts.append(lu.kind.value if lu.density is None else f"urban {lu.density.value}")
        if level in ("c", "p"):
            c = int(self.ancestor_of(level, "c")[j])
            parts.append(f"column {column_role(self.levels['c'].decl[c].itype)}")
        if level == "p":
            parts.append(f"pft {self.levels['p'].decl[j].pft}")
        return ", ".join(parts)

    def table(self, level: str):
        """A pandas DataFrame with one row per element of the level, indexed by its Fortran index:
        set, gridcell (index and name), topounit (index and number), and where they exist
        landunit, kind, density, column, role, pft, weights (in the parent and the gridcell) and
        active."""
        import pandas as pd

        n = self.count(level)
        rows: dict[str, Any] = {"set": self.set_of(level), "gridcell": self.parent(level, "g"),
                                "name": self.name_of(level)}
        if level != "g":
            t = self.ancestor_of(level, "t")
            rows["topounit"] = t + 1
            rows["topo_number"] = np.asarray(self.topo_number)[t]
        if level in ("l", "c", "p"):
            lu = self.ancestor_of(level, "l")
            decl = self.levels["l"].decl
            rows["landunit"] = lu + 1
            rows["kind"] = [decl[k].kind.value for k in lu]
            rows["density"] = [decl[k].density.value if decl[k].density else "" for k in lu]
        if level in ("c", "p"):
            c = self.ancestor_of(level, "c")
            ctype = self.itype("c")
            rows["column"] = c + 1
            rows["role"] = [column_role(int(ctype[k])) for k in c]
        if level == "p":
            rows["pft"] = self.itype("p")
        if level != "g":
            rows["weight"] = self._wparent[level]
            rows["wtgcell"] = self.weight(level, "g")
        rows["active"] = self.active(level)
        df = pd.DataFrame(rows, index=pd.Index(np.arange(1, n + 1), name=LEVEL_NAMES[level]))
        return df
