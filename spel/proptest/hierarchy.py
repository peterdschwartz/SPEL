"""ELM's subgrid hierarchy for synthetic domains: declarations and their layout.

A domain is declared as gridcells, each with topounits, landunits, columns and patches, every
element with a weight in its parent (`Gridcell`, `Topounit`, `Landunit`, `Column`, `Patch`). The
constructors follow initGridCellsMod's set_landunit_*:

- `Soil(1, Patch(13, 0.4))`: a natural-vegetation landunit (istsoil) with ONE soil column holding
  one patch per pft;
- `Polygon("low", 12)`: a polygonal-tundra landunit (use_polygonal_tundra): itype istsoil,
  `ispolygon`, `polygontype` 1/2/3 (low/flat/high centred), one soil column like `Soil`;
- `Crop(17, 19)`: a crop landunit (istcrop) with one column per CFT (itype istcrop*100 + cft),
  one patch each; a CFT's weight is its column's weight in the landunit;
- `Urban("md")`: an urban landunit of that density with the five urban columns (roof, sunwall,
  shadewall, impervious and pervious road; set_landunit_urban's weights) and one bare patch each;
- `Lake()`, `Wetland()`, `Glacier()`: deep lake, wetland, plain glacier: one column of the
  landunit's type and one bare patch (set_landunit_wet_ice_lake);
- `GlacierMec({1: 0.3, 2: (0.7, 1500.0)})`: glacier with elevation classes (istice_mec): one column
  per class m (itype istice_mec*100 + m) with its weight and, optionally, its `glc_topo`.

A topounit holds at most one landunit of each type (one polygon of each kind): ELM creates one
per type and topounit. Gridcells have `glcmask` (glacier-mec columns of every class exist and are
active there: the ice-sheet coupling's mask) and `icemask` (do_smb_c).

Weights left out share what the given ones leave, equally; the weights of every element's
children must sum to 1. Activity follows subgridWeightsMod (is_active_l/c/p, `all_active` off):

- a topounit is active when its weight is > 0 (set_topounit);
- a landunit when its topounit is and: its weight is > 0, or it is glacier-mec in a glcmask
  gridcell, or it is istsoil (soil, polygon) and the topounit is not all plain glacier (ELM's
  virtual natural-vegetation landunit);
- a column when its topounit is and: its landunit is active and its weight is > 0, or it is
  glacier-mec in a glcmask gridcell, or its landunit is an active urban landunit;
- a patch when its column is active and its weight is > 0.

ELM's implicit elements (`Subgrid`, `[domain] subgrid = {...}`): ELM always creates a soil
landunit on every topounit with every natural pft as a patch, a crop landunit with every CFT
column when create_crop_landunit, and every glacier-mec class in glcmask gridcells; most of them
have weight 0 (inactive, or active: the virtual soil landunit). With a `Subgrid` the layout adds
what the declaration leaves out, with weight 0 (an added landunit's first pft/CFT/class gets
weight 1 within it, as the surface dataset's weights must sum to 1). Without it, the domain holds
only what is declared: smaller, but not what ELM would build from the same surface data.

`Layout` lays the gridcells out as ELM lays out a clump (initGridCellsMod, `order="elm"`):
gridcells and their topounits in order, then the landunits in type passes over all topounits
(soil with its polygons, crop, urban tbd/hd/md, lake, wetland, glacier, glacier-mec), the columns
and patches of a landunit right after the previous landunit's. The landunits of a topounit are
therefore not contiguous (ELM leaves top_pp%lndi... unset). `order="depth"` keeps every element's
descendants contiguous instead (the layout before 2026-11-12). A set is one clump: the `copies` of
a gridcell are next to each other, and the whole set is repeated `nsets` times (set k of every
level is a contiguous range, as duplicateMod copies sets).

In TOML a gridcell is `[[domain.gridcell]]` with `landunits = [...]` (one topounit) or
`topounits = [{ weight, elevation, landunits = [...] }, ...]`; `to_table`/`from_table` convert.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from spel.proptest.errors import ProptestError

LEVELS = ("g", "t", "l", "c", "p")
LEVEL_NAMES = {"g": "gridcell", "t": "topounit", "l": "landunit", "c": "column", "p": "patch"}
WEIGHT_TOL = 1e-9
ISPVAL = -9999  # elm_varcon::ispval
MAX_LUNIT = 12  # landunit_varcon::max_lunit
MAX_NON_POLY_LUNIT = 9  # landunit_varcon::max_non_poly_lunit
ALL_GLACIER_TOL = 1e-13  # subgridWeightsMod::is_topo_all_gla


class HierarchyError(ProptestError):
    pass


class LandunitKind(Enum):
    SOIL = "soil"
    CROP = "crop"
    URBAN = "urban"
    LAKE = "lake"
    WETLAND = "wetland"
    GLACIER = "glacier"
    GLACIER_MEC = "glacier_mec"
    POLYGON = "polygon"

    @property
    def vegetated(self) -> bool:
        return self in (LandunitKind.SOIL, LandunitKind.CROP, LandunitKind.POLYGON)

    @property
    def special(self) -> bool:
        """landunit_is_special: every landunit but istsoil (soil, polygons) and istcrop."""
        return not self.vegetated


class UrbanDensity(Enum):
    """ELM's urban classes (landunit_varcon): tall building district, high and medium density."""

    TBD = "tbd"
    HD = "hd"
    MD = "md"

    @property
    def type_code(self) -> str:
        """The landunit_varcon parameter of its landunit type (a TYPE_CODES key)."""
        return f"isturb_{self.value}"


class PolygonType(Enum):
    """Polygonal tundra (landunit_varcon ilowcenpoly/iflatcenpoly/ihighcenpoly)."""

    LOW = "low"
    FLAT = "flat"
    HIGH = "high"

    @property
    def code(self) -> int:
        """lun_pp%polygontype."""
        return {"low": 1, "flat": 2, "high": 3}[self.value]

    @property
    def type_code(self) -> str:
        """Its key in landunit_indices (polygontype + max_non_poly_lunit)."""
        return f"ist{self.value}cenpoly"


# ELM's subgrid type codes (landunit_varcon, column_varcon, pftvarcon). The case's own module
# parameters, when the analysis has them, must agree (`check_type_codes`).
TYPE_CODES = {
    "istsoil": 1, "istcrop": 2, "istice": 3, "istice_mec": 4, "istdlak": 5, "istwet": 6,
    "isturb_tbd": 7, "isturb_hd": 8, "isturb_md": 9,
    "istlowcenpoly": 10, "istflatcenpoly": 11, "isthighcenpoly": 12,
    "icol_roof": 71, "icol_sunwall": 72, "icol_shadewall": 73, "icol_road_imperv": 74, "icol_road_perv": 75,
    "noveg": 0,
}
# the columns of an urban landunit, in initGridCellsMod's order (set_landunit_urban)
URBAN_COLUMNS = ("icol_roof", "icol_sunwall", "icol_shadewall", "icol_road_imperv", "icol_road_perv")
SOIL_COLUMN = 1  # initGridCellsMod: the natural-vegetation column's type
_LTYPE_KEY = {
    LandunitKind.SOIL: "istsoil", LandunitKind.POLYGON: "istsoil", LandunitKind.CROP: "istcrop",
    LandunitKind.LAKE: "istdlak", LandunitKind.WETLAND: "istwet", LandunitKind.GLACIER: "istice",
    LandunitKind.GLACIER_MEC: "istice_mec",
}
# initGridCellsMod's landunit passes (the ELM order), urban by density
_PASS = {LandunitKind.SOIL: 0, LandunitKind.POLYGON: 0, LandunitKind.CROP: 1, LandunitKind.LAKE: 5,
         LandunitKind.WETLAND: 6, LandunitKind.GLACIER: 7, LandunitKind.GLACIER_MEC: 8}
_URBAN_PASS = {UrbanDensity.TBD: 2, UrbanDensity.HD: 3, UrbanDensity.MD: 4}


def check_type_codes(params: dict[str, Any]) -> None:
    """TYPE_CODES agree with the case's module parameters where the analysis has them."""
    for name, value in TYPE_CODES.items():
        have = params.get(name)
        if isinstance(have, int) and have != value:
            raise HierarchyError(f"the case has {name} = {have}; the synthetic domain assumes {value} "
                                 "(update TYPE_CODES in spel/proptest/hierarchy.py)")


def ltype_of(kind: LandunitKind, density: Optional[UrbanDensity]) -> int:
    """lun_pp%itype (polygons are istsoil)."""
    if kind is LandunitKind.URBAN:
        assert density is not None
        return TYPE_CODES[density.type_code]
    return TYPE_CODES[_LTYPE_KEY[kind]]


def mec_column(m: int) -> int:
    """column_varcon::icemec_class_to_col_itype."""
    return TYPE_CODES["istice_mec"] * 100 + m


def column_role(itype: int) -> str:
    """A column type's name: soil, crop_<cft>, glacier_mec_<class>, lake, wetland, glacier, roof, ... ."""
    for key in URBAN_COLUMNS:
        if TYPE_CODES[key] == itype:
            return key.removeprefix("icol_")
    if itype == SOIL_COLUMN:
        return "soil"
    named = {TYPE_CODES["istdlak"]: "lake", TYPE_CODES["istwet"]: "wetland", TYPE_CODES["istice"]: "glacier"}
    if itype in named:
        return named[itype]
    if itype > TYPE_CODES["istice_mec"] * 100:
        return f"glacier_mec_{itype - TYPE_CODES['istice_mec'] * 100}"
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
    topo: Optional[float] = None  # glacier-mec: col_pp%glc_topo (m)

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
    polygon: Optional[PolygonType] = None  # polygonal tundra

    @property
    def ltype(self) -> int:
        return ltype_of(self.kind, self.density)

    @property
    def index_type(self) -> int:
        """Its key in landunit_indices: the ltype, polygontype + max_non_poly_lunit for polygons."""
        if self.kind is LandunitKind.POLYGON:
            assert self.polygon is not None
            return self.polygon.code + MAX_NON_POLY_LUNIT
        return self.ltype

    @property
    def label(self) -> str:
        if self.kind is LandunitKind.URBAN:
            return f"urban {self.density.value}"
        if self.kind is LandunitKind.POLYGON:
            return f"polygon {self.polygon.value}"
        return self.kind.value

    def to_table(self) -> dict:
        out: dict[str, Any] = {"kind": self.kind.value}
        if self.kind in (LandunitKind.SOIL, LandunitKind.POLYGON):
            if self.kind is LandunitKind.POLYGON:
                out["type"] = self.polygon.value
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
        elif self.kind is LandunitKind.GLACIER_MEC:
            out["classes"] = []
            for c in self.columns:
                row: dict[str, Any] = {"class": c.itype - mec_column(0)}
                if c.weight is not None:
                    row["weight"] = c.weight
                if c.topo is not None:
                    row["topo"] = c.topo
                out["classes"].append(row)
        if self.weight is not None:
            out["weight"] = self.weight
        return out

    @classmethod
    def from_table(cls, value: Any, where: str) -> "Landunit":
        kinds = [k.value for k in LandunitKind]
        if not isinstance(value, dict) or "kind" not in value:
            raise HierarchyError(f"{where}: a landunit is a table with `kind` ({kinds})")
        try:
            kind = LandunitKind(str(value["kind"]).lower())
        except ValueError:
            raise HierarchyError(f"{where}: kind must be one of {kinds}, not {value['kind']!r}") from None
        weight = _weight(value, where)
        if kind in (LandunitKind.SOIL, LandunitKind.POLYGON):
            poly = None
            known = {"kind", "weight", "patches", "columns"}
            if kind is LandunitKind.POLYGON:
                known.add("type")
                try:
                    poly = PolygonType(str(value.get("type", "")).lower())
                except ValueError:
                    raise HierarchyError(f"{where}: a polygon's type is one of "
                                         f"{[t.value for t in PolygonType]}") from None
            _keys(value, known, where)
            if ("patches" in value) == ("columns" in value):
                raise HierarchyError(f"{where}: a {kind.value} landunit takes `patches` (one column) or `columns`")
            if "patches" in value:
                patches = tuple(Patch.from_table(p, f"{where}.patches[{k}]") for k, p in enumerate(value["patches"]))
                return Soil(*patches, weight=weight) if poly is None else Polygon(poly, *patches, weight=weight)
            cols = tuple(Column.from_table(c, f"{where}.columns[{k}]", SOIL_COLUMN) for k, c in enumerate(value["columns"]))
            return _checked(cls(kind, cols, weight, polygon=poly), where)
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
        if kind is LandunitKind.GLACIER_MEC:
            _keys(value, {"kind", "weight", "classes"}, where)
            classes: dict[int, tuple[Optional[float], Optional[float]]] = {}
            for k, row in enumerate(value.get("classes", [])):
                w = f"{where}.classes[{k}]"
                if not isinstance(row, dict) or "class" not in row:
                    raise HierarchyError(f"{w}: a class is a table {{ class, weight, topo }}")
                _keys(row, {"class", "weight", "topo"}, w)
                topo = row.get("topo")
                classes[int(row["class"])] = (_weight(row, w), None if topo is None else float(topo))
            return GlacierMec(classes, weight=weight)
        _keys(value, {"kind", "weight"}, where)
        return {LandunitKind.LAKE: Lake, LandunitKind.WETLAND: Wetland, LandunitKind.GLACIER: Glacier}[kind](weight=weight)


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

    @property
    def all_glacier(self) -> bool:
        """subgridWeightsMod::is_topo_all_gla: plain glacier covers the topounit."""
        w = resolve_weights([lu.weight for lu in self.landunits], "landunits")
        return sum(x for lu, x in zip(self.landunits, w) if lu.kind is LandunitKind.GLACIER) >= 1.0 - ALL_GLACIER_TOL


@dataclass(frozen=True)
class Gridcell:
    topounits: tuple[Topounit, ...]
    copies: int = 1
    name: str = ""  # reports, queries; relations permute gridcells of the same name
    glcmask: bool = False  # ldomain%glcmask: glacier-mec columns of every class exist and are active
    icemask: float = 0.0  # icemask_grc (do_smb_c)

    @classmethod
    def of(cls, *landunits: Landunit, copies: int = 1, name: str = "", glcmask: bool = False,
           icemask: float = 0.0) -> "Gridcell":
        """A gridcell with one topounit."""
        return cls((Topounit(tuple(landunits)),), copies, name, glcmask, icemask)

    @property
    def label(self) -> str:
        if self.name:
            return self.name
        return "+".join(lu.label for t in self.topounits for lu in t.landunits)

    def to_table(self) -> dict:
        out: dict[str, Any] = {}
        if self.name:
            out["name"] = self.name
        if self.copies != 1:
            out["copies"] = self.copies
        if self.glcmask:
            out["glcmask"] = True
        if self.icemask:
            out["icemask"] = self.icemask
        if len(self.topounits) == 1 and self.topounits[0].weight is None and self.topounits[0].elevation is None:
            out["landunits"] = [lu.to_table() for lu in self.topounits[0].landunits]
        else:
            out["topounits"] = [t.to_table() for t in self.topounits]
        return out

    @classmethod
    def from_table(cls, value: Any, where: str) -> "Gridcell":
        if not isinstance(value, dict):
            raise HierarchyError(f"{where}: a gridcell is a table")
        _keys(value, {"name", "copies", "landunits", "topounits", "glcmask", "icemask"}, where)
        if ("landunits" in value) == ("topounits" in value):
            raise HierarchyError(f"{where}: give `landunits` (one topounit) or `topounits`")
        if "landunits" in value:
            tops = (Topounit(tuple(Landunit.from_table(x, f"{where}.landunits[{k}]")
                                   for k, x in enumerate(value["landunits"]))),)
        else:
            tops = tuple(Topounit.from_table(t, f"{where}.topounits[{k}]") for k, t in enumerate(value["topounits"]))
        g = cls(tops, int(value.get("copies", 1)), str(value.get("name", "")), bool(value.get("glcmask", False)),
                float(value.get("icemask", 0.0)))
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
            seen: dict[int, int] = {}
            for j, lu in enumerate(t.landunits):
                _checked(lu, f"{tw}.landunit[{j}]")
                if lu.index_type in seen:
                    raise HierarchyError(f"{tw}: landunits {seen[lu.index_type]} and {j} are both {lu.label} "
                                         "(ELM creates one landunit of each type per topounit)")
                seen[lu.index_type] = j
        return self


@dataclass(frozen=True)
class Subgrid:
    """What ELM creates on every topounit besides the declared landunits (subgridMod): the
    natural pfts `natpft` = (natpft_lb, natpft_ub), the CFTs `cft` = (cft_lb, cft_ub) when
    create_crop_landunit (None: no crop landunit), and `maxpatch_glcmec` glacier elevation classes
    (glcmask gridcells). `Subgrid.elm(crop=...)` gives elm_varpar's values (use_crop: 0-14 and
    15-50; otherwise 0-16 and no crop landunit)."""

    natpft: tuple[int, int]
    cft: Optional[tuple[int, int]] = None
    maxpatch_glcmec: int = 0

    KEYS = ("natpft", "cft", "maxpatch_glcmec")

    @classmethod
    def elm(cls, crop: bool = False, maxpatch_glcmec: int = 0) -> "Subgrid":
        if crop:  # numpft = mxpft = 50, numcft = 36: natpft_size = 51 - 36
            return cls((0, 14), (15, 50), maxpatch_glcmec)
        return cls((0, 16), None, maxpatch_glcmec)  # numpft = numveg = 16, no crop landunit

    def to_table(self) -> dict:
        out: dict[str, Any] = {"natpft": list(self.natpft)}
        if self.cft is not None:
            out["cft"] = list(self.cft)
        if self.maxpatch_glcmec:
            out["maxpatch_glcmec"] = self.maxpatch_glcmec
        return out

    @classmethod
    def from_table(cls, value: Any, where: str = "[domain] subgrid") -> "Subgrid":
        if not isinstance(value, dict) or "natpft" not in value:
            raise HierarchyError(f"{where}: a table {{ natpft = [lb, ub], cft = [lb, ub], maxpatch_glcmec = n }}")
        _keys(value, set(cls.KEYS), where)

        def pair(key: str) -> Optional[tuple[int, int]]:
            v = value.get(key)
            if v is None:
                return None
            if not (isinstance(v, (list, tuple)) and len(v) == 2 and int(v[0]) <= int(v[1])):
                raise HierarchyError(f"{where}: {key} is [lb, ub]")
            return int(v[0]), int(v[1])

        return cls(pair("natpft"), pair("cft"), int(value.get("maxpatch_glcmec", 0)))

    # -- the implicit elements
    def expand(self, g: Gridcell, where: str) -> list[Topounit]:
        """The gridcell's topounits with every landunit, column and patch ELM creates."""
        return [self._topounit(g, t, f"{where}.topounit[{k}]") for k, t in enumerate(g.topounits)]

    def _topounit(self, g: Gridcell, t: Topounit, where: str) -> Topounit:
        lus = list(t.landunits)
        kinds = {lu.kind for lu in lus}
        for j, lu in enumerate(lus):
            if lu.kind in (LandunitKind.LAKE, LandunitKind.WETLAND, LandunitKind.GLACIER) and lu.weight == 0.0:
                raise HierarchyError(f"{where}.landunit[{j}]: ELM creates no {lu.kind.value} landunit of weight 0 "
                                     "(subgridMod); leave it out")
            if lu.kind is LandunitKind.GLACIER_MEC and not g.glcmask and lu.weight == 0.0:
                raise HierarchyError(f"{where}.landunit[{j}]: a glacier-mec landunit of weight 0 exists only in "
                                     "a glcmask gridcell")
            if lu.kind is LandunitKind.CROP and self.cft is None:
                raise HierarchyError(f"{where}.landunit[{j}]: a crop landunit needs create_crop_landunit "
                                     "(give `cft` in the subgrid)")
        if LandunitKind.SOIL not in kinds:
            lus.insert(0, Soil(self.natpft[0], weight=0.0))
        if self.cft is not None and LandunitKind.CROP not in kinds:
            lus.append(Crop(self.cft[0], weight=0.0))
        if g.glcmask and self.maxpatch_glcmec and LandunitKind.GLACIER_MEC not in kinds:
            lus.append(GlacierMec({1: 1.0}, weight=0.0))
        out = []
        for j, lu in enumerate(lus):
            w = f"{where}.landunit[{j}]"
            if lu.kind in (LandunitKind.SOIL, LandunitKind.POLYGON):
                lu = replace(lu, columns=tuple(replace(c, patches=self._fill(c.patches, self.natpft, w))
                                               for c in lu.columns))
            elif lu.kind is LandunitKind.CROP:
                assert self.cft is not None
                have = {c.patches[0].pft: c for c in lu.columns}
                bad = sorted(set(have) - set(range(self.cft[0], self.cft[1] + 1)))
                if bad:
                    raise HierarchyError(f"{w}: CFTs {bad} are outside cft_lb:cft_ub = {self.cft}")
                cw = dict(zip(have, resolve_weights([c.weight for c in lu.columns], f"{w}.cfts")))
                lu = Crop(*(Patch(m, cw.get(m, 0.0)) for m in range(self.cft[0], self.cft[1] + 1)), weight=lu.weight)
            elif lu.kind is LandunitKind.GLACIER_MEC and g.glcmask:
                if not self.maxpatch_glcmec:
                    raise HierarchyError(f"{w}: a glcmask gridcell needs maxpatch_glcmec in the subgrid")
                have = {c.itype - mec_column(0): c for c in lu.columns}
                bad = sorted(m for m in have if not 1 <= m <= self.maxpatch_glcmec)
                if bad:
                    raise HierarchyError(f"{w}: classes {bad} are outside 1..maxpatch_glcmec = {self.maxpatch_glcmec}")
                cw = dict(zip(have, resolve_weights([c.weight for c in lu.columns], f"{w}.classes")))
                lu = GlacierMec({m: (cw.get(m, 0.0), have[m].topo if m in have else None)
                                 for m in range(1, self.maxpatch_glcmec + 1)}, weight=lu.weight)
            out.append(lu)
        return replace(t, landunits=tuple(out))

    @staticmethod
    def _fill(patches: Sequence[Patch], rng: tuple[int, int], where: str) -> tuple[Patch, ...]:
        have = [p.pft for p in patches]
        bad = sorted(set(have) - set(range(rng[0], rng[1] + 1)))
        if bad:
            raise HierarchyError(f"{where}: pfts {bad} are outside natpft_lb:natpft_ub = {rng}")
        w = dict(zip(have, resolve_weights([p.weight for p in patches], f"{where}.patches")))
        return tuple(Patch(m, w.get(m, 0.0)) for m in range(rng[0], rng[1] + 1))


# -- constructors (initGridCellsMod)


def _patch(p: Patch | int) -> Patch:
    return p if isinstance(p, Patch) else Patch(int(p))


def Soil(*patches: Patch | int, weight: Optional[float] = None) -> Landunit:
    """istsoil: one column (type 1) with one patch per pft."""
    return Landunit(LandunitKind.SOIL, (Column(SOIL_COLUMN, tuple(_patch(p) for p in patches)),), weight)


def Polygon(kind: PolygonType | str, *patches: Patch | int, weight: Optional[float] = None) -> Landunit:
    """A polygonal-tundra landunit (add_polygon_landunit): itype istsoil, one soil column with
    one patch per pft, `polygontype` low/flat/high."""
    kind = PolygonType(kind.lower() if isinstance(kind, str) else kind)
    return Landunit(LandunitKind.POLYGON, (Column(SOIL_COLUMN, tuple(_patch(p) for p in patches)),), weight,
                    polygon=kind)


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


def _wet_ice_lake(kind: LandunitKind, weight: Optional[float]) -> Landunit:
    ltype = ltype_of(kind, None)
    return Landunit(kind, (Column(ltype, (Patch(TYPE_CODES["noveg"]),)),), weight)


def Lake(weight: Optional[float] = None) -> Landunit:
    """istdlak: one column of the landunit's type (set_landunit_wet_ice_lake), one bare patch."""
    return _wet_ice_lake(LandunitKind.LAKE, weight)


def Wetland(weight: Optional[float] = None) -> Landunit:
    """istwet: one column of the landunit's type, one bare patch."""
    return _wet_ice_lake(LandunitKind.WETLAND, weight)


def Glacier(weight: Optional[float] = None) -> Landunit:
    """istice (plain glacier): one column of the landunit's type, one bare patch."""
    return _wet_ice_lake(LandunitKind.GLACIER, weight)


def GlacierMec(classes: Mapping[int, Optional[float] | tuple[Optional[float], Optional[float]]] | Sequence[int],
               weight: Optional[float] = None) -> Landunit:
    """istice_mec: one column per elevation class m (itype istice_mec*100 + m) in class order,
    one bare patch each. `classes` maps a class to its weight in the landunit, or to
    (weight, glc_topo); a list of classes shares the weight equally."""
    if not isinstance(classes, Mapping):
        classes = {int(m): None for m in classes}
    cols = []
    for m in sorted(classes):
        v = classes[m]
        w, topo = v if isinstance(v, tuple) else (v, None)
        if m < 1:
            raise HierarchyError(f"glacier_mec: classes start at 1, not {m}")
        cols.append(Column(mec_column(m), (Patch(TYPE_CODES["noveg"]),), w, topo))
    return Landunit(LandunitKind.GLACIER_MEC, tuple(cols), weight)


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
    if lu.kind is LandunitKind.GLACIER_MEC:
        classes = [c.itype for c in lu.columns]
        if len(set(classes)) != len(classes):
            raise HierarchyError(f"{where}: a glacier-mec class appears twice")
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

ORDERS = ("elm", "depth")
# landunit-level flags (initSubgridMod::add_landunit) and the kinds they hold for
LANDUNIT_FLAGS = {
    "ifspecial": tuple(k for k in LandunitKind if k.special),
    "lakpoi": (LandunitKind.LAKE,), "urbpoi": (LandunitKind.URBAN,),
    "glcmecpoi": (LandunitKind.GLACIER_MEC,), "ispolygon": (LandunitKind.POLYGON,),
}
# column flags (add_column) and patch flags (add_patch: those of its column)
COLUMN_FLAGS = {
    "is_soil": (LandunitKind.SOIL, LandunitKind.POLYGON), "is_crop": (LandunitKind.CROP,),
    "is_lake": (LandunitKind.LAKE,),
}
PATCH_FLAGS = {"is_on_soil_col": COLUMN_FLAGS["is_soil"], "is_on_crop_col": COLUMN_FLAGS["is_crop"]}
FLAG_LEVEL = {**{f: "l" for f in LANDUNIT_FLAGS}, **{f: "c" for f in COLUMN_FLAGS}, **{f: "p" for f in PATCH_FLAGS}}


@dataclass
class _Level:
    """The elements of one level, 0-based: parent (index in the next coarser level) and the
    weight in it; the declaration each element comes from."""

    parent: list[int] = field(default_factory=list)
    weight: list[float] = field(default_factory=list)
    decl: list[Any] = field(default_factory=list)


class Layout:
    """The declared gridcells laid out as ELM lays out a clump (see the module docstring). All
    arrays are per element of a level; indices given to and returned by the methods are Fortran
    (1-based) unless named `*_of` or documented as rows (0-based positions)."""

    def __init__(self, gridcells: Sequence[Gridcell], nsets: int = 1, subgrid: Optional[Subgrid] = None,
                 order: str = "elm"):
        if not gridcells:
            raise HierarchyError("the domain has no gridcell")
        if nsets < 1:
            raise HierarchyError("nsets must be >= 1")
        if order not in ORDERS:
            raise HierarchyError(f"order must be one of {ORDERS}, not {order!r}")
        self.declared: list[Gridcell] = [g.checked(f"gridcell[{k}]") for k, g in enumerate(gridcells)]
        self.nsets = nsets
        self.subgrid = subgrid
        self.order = order
        # the topounits as laid out: with ELM's implicit elements under a subgrid
        expanded = [subgrid.expand(g, f"gridcell[{k}]") if subgrid else list(g.topounits)
                    for k, g in enumerate(self.declared)]
        # one set: every declared gridcell `copies` times, in order
        self.set_gridcells: list[Gridcell] = [g for g in self.declared for _ in range(g.copies)]
        set_tops = [tops for g, tops in zip(self.declared, expanded) for _ in range(g.copies)]
        self.unit_of_gridcell: list[int] = []  # position of each gridcell within its set
        self.levels: dict[str, _Level] = {lev: _Level() for lev in LEVELS}
        self.topo_number: list[int] = []  # topounit number within its gridcell (1..n)
        self.elevation: list[Optional[float]] = []
        for _ in range(nsets):
            self._add_set(set_tops)
        self._arrays()

    def _add_set(self, set_tops: list[list[Topounit]]) -> None:
        L = self.levels
        entries = []  # (sort key, topounit row, landunit, its weight)
        for unit, (g, tops) in enumerate(zip(self.set_gridcells, set_tops)):
            gi = len(L["g"].parent)
            L["g"].parent.append(-1)
            L["g"].weight.append(1.0)
            L["g"].decl.append(g)
            self.unit_of_gridcell.append(unit)
            for k, (t, tw) in enumerate(zip(tops, resolve_weights([t.weight for t in tops], "topounits"))):
                ti = len(L["t"].parent)
                L["t"].parent.append(gi)
                L["t"].weight.append(tw)
                L["t"].decl.append(t)
                self.topo_number.append(k + 1)
                self.elevation.append(t.elevation)
                lws = resolve_weights([lu.weight for lu in t.landunits], "landunits")
                for j, (lu, lw) in enumerate(zip(t.landunits, lws)):
                    if self.order == "depth":
                        key: tuple = (ti, j)
                    else:  # initGridCellsMod: one pass per landunit type over the clump's topounits
                        kind_pass = _URBAN_PASS[lu.density] if lu.kind is LandunitKind.URBAN else _PASS[lu.kind]
                        sub = lu.polygon.code if lu.kind is LandunitKind.POLYGON else 0
                        key = (kind_pass, ti, sub)
                    entries.append((key, ti, lu, lw))
        for _, ti, lu, lw in sorted(entries, key=lambda e: e[0]):
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
        self._ltype = np.asarray([lu.ltype for lu in self.levels["l"].decl], dtype=np.int64)
        self._kind_l = [lu.kind for lu in self.levels["l"].decl]
        self._glcmask_g = np.asarray([g.glcmask for g in self.levels["g"].decl], dtype=bool)
        self._all_glacier_t = np.asarray([t.all_glacier for t in self.levels["t"].decl], dtype=bool)

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
        """Fortran index of each element's first descendant at a finer (or the same) level
        (ispval-free: 0 where it has none)."""
        return self._extreme_child(level, finer, first=True)

    def last_child(self, level: str, finer: str) -> np.ndarray:
        return self._extreme_child(level, finer, first=False)

    def _extreme_child(self, level: str, finer: str, first: bool) -> np.ndarray:
        owner = self.ancestor_of(finer, level)
        rows = np.arange(self.count(finer), dtype=np.int64)
        if first:
            big = np.iinfo(np.int64).max
            out = np.full(self.count(level), big, dtype=np.int64)
            np.minimum.at(out, owner, rows)
            out[out == big] = -1
        else:
            out = np.full(self.count(level), -1, dtype=np.int64)
            np.maximum.at(out, owner, rows)
        return out + 1

    def contiguous(self, level: str, finer: str) -> bool:
        """Whether every element's descendants at `finer` are one contiguous range (always for
        l -> c -> p; in the ELM order not for a topounit's or gridcell's landunits)."""
        return bool(np.all(self.last_child(level, finer) - self.first_child(level, finer) + 1
                           == self.children(level, finer)))

    def children(self, level: str, finer: str) -> np.ndarray:
        """The number of each element's descendants at a finer level."""
        return np.bincount(self.ancestor_of(finer, level), minlength=self.count(level)).astype(np.int64)

    def one_to_one(self, a: str, b: str) -> bool:
        """Whether two levels have the same elements (a tree: equal counts)."""
        return self.count(a) == self.count(b)

    def descendants(self, level: str, k: int, finer: str) -> np.ndarray:
        """0-based rows of `finer` under element k (Fortran) of `level`, in layout order."""
        return np.flatnonzero(self.ancestor_of(finer, level) == k - 1)

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
        """A permutation of the first `len(gridcell_perm)` gridcells as rows of a level: the
        rows of gridcell k receive what the rows of gridcell `perm[k]` held (as an index array:
        `new = old[result]`). Gridcells are only exchanged with gridcells of the same
        declaration, whose elements come in the same relative order."""
        perm = np.asarray(gridcell_perm, dtype=np.int64)
        owner = self.gridcell_of(level)
        n = int(np.count_nonzero(owner < len(perm)))
        out = np.arange(n, dtype=np.int64)
        for g, src in enumerate(perm):
            dst_rows, src_rows = np.flatnonzero(owner == g), np.flatnonzero(owner == src)
            if len(dst_rows) != len(src_rows):
                raise HierarchyError(f"gridcells {g + 1} and {src + 1} differ in their {LEVEL_NAMES[level]}s")
            out[dst_rows] = src_rows
        return out

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
        """col_pp/lun_pp/top_pp/veg_pp%active as subgridWeightsMod's is_active_* set them (see the
        module docstring; `all_active` off)."""
        if level == "g":
            return np.ones(self.count("g"), dtype=bool)
        own = self._wparent[level] > 0.0
        if level == "t":
            return own
        top = self.active("t")[self.ancestor_of(level, "t")]
        if level == "p":
            return self.active("c")[self._parent["p"]] & own
        lt = self.ltype(level)
        mec = (lt == TYPE_CODES["istice_mec"]) & self._glcmask_g[self.gridcell_of(level)]
        if level == "l":
            soil = (lt == TYPE_CODES["istsoil"]) & ~self._all_glacier_t[self.ancestor_of("l", "t")]
            return top & (own | mec | soil)
        lun = self.active("l")[self._parent["c"]]
        urban = self.is_kind("c", LandunitKind.URBAN)
        return top & ((lun & own) | mec | (lun & urban))

    def flag(self, level: str, name: str) -> np.ndarray:
        """A logical subgrid flag (LANDUNIT_FLAGS, COLUMN_FLAGS, PATCH_FLAGS) of each element of
        its level or a finer one (the ancestor's value)."""
        home = FLAG_LEVEL.get(name)
        if home is None:
            raise HierarchyError(f"no subgrid flag {name!r}")
        if LEVELS.index(level) < LEVELS.index(home):
            raise HierarchyError(f"{name} is a {LEVEL_NAMES[home]} flag, not of {LEVEL_NAMES[level]}s")
        kinds = {**LANDUNIT_FLAGS, **COLUMN_FLAGS, **PATCH_FLAGS}[name]
        return self.is_kind(level, *kinds)

    def polygon_type(self) -> np.ndarray:
        """lun_pp%polygontype: 1/2/3 on polygon landunits, ispval elsewhere."""
        return np.asarray([lu.polygon.code if lu.polygon else ISPVAL for lu in self.levels["l"].decl], dtype=np.int64)

    def glc_topo(self) -> np.ndarray:
        """col_pp%glc_topo at initialization: the class's topo on glacier-mec columns (NaN where
        the declaration has none), 0 elsewhere."""
        mec = self.is_kind("c", LandunitKind.GLACIER_MEC)
        topo = [c.topo if c.topo is not None else np.nan for c in self.levels["c"].decl]
        return np.where(mec, np.asarray(topo, dtype=float), 0.0)

    def mxy(self) -> np.ndarray:
        """veg_pp%mxy: pft + 1 - natpft_lb on soil and crop landunits, ispval elsewhere."""
        if self.subgrid is None:
            raise HierarchyError("veg_pp%mxy needs natpft_lb: give the domain a subgrid")
        veg = self.is_kind("p", LandunitKind.SOIL, LandunitKind.POLYGON, LandunitKind.CROP)
        return np.where(veg, self.itype("p") + 1 - self.subgrid.natpft[0], ISPVAL).astype(np.int64)

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
            ts = [int(t) for t in self.descendants("g", g + 1, "t")]
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
        decl = self.levels["l"].decl
        for li in range(self.count("l") - 1, -1, -1):  # backwards: the first landunit of a type wins
            t = int(decl[li].index_type)
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
            parts.append(lu.label)
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
            rows["polygon"] = [decl[k].polygon.value if decl[k].polygon else "" for k in lu]
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
