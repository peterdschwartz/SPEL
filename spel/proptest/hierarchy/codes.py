"""ELM's subgrid levels, landunit kinds, type codes and logical flags."""

from __future__ import annotations

from enum import Enum, StrEnum
from typing import Any, Optional

from spel.proptest.errors import ProptestError

WEIGHT_TOL = 1e-9
ISPVAL = -9999  # elm_varcon::ispval
MAX_LUNIT = 12  # landunit_varcon::max_lunit
MAX_NON_POLY_LUNIT = 9  # landunit_varcon::max_non_poly_lunit
ALL_GLACIER_TOL = 1e-13  # subgridWeightsMod::is_topo_all_gla


class HierarchyError(ProptestError):
    pass


class Level(StrEnum):
    """A level of the subgrid hierarchy, coarsest first. A StrEnum: `Level.P == "p"`, so the
    inventory's level strings and the Layout's arguments mix freely (`Level("p")`)."""

    G = "g"
    T = "t"
    L = "l"
    C = "c"
    P = "p"

    @property
    def label(self) -> str:
        """gridcell, topounit, landunit, column, patch."""
        return _LABELS[self]

    @property
    def depth(self) -> int:
        """0 for gridcells ... 4 for patches."""
        return LEVELS.index(self)

    @property
    def coarser(self) -> Optional["Level"]:
        """The parent level (None for gridcells)."""
        return LEVELS[self.depth - 1] if self.depth else None

    @property
    def finer(self) -> Optional["Level"]:
        """The child level (None for patches)."""
        return LEVELS[self.depth + 1] if self.depth + 1 < len(LEVELS) else None


LEVELS: tuple[Level, ...] = tuple(Level)
_LABELS = {
    Level.G: "gridcell",
    Level.T: "topounit",
    Level.L: "landunit",
    Level.C: "column",
    Level.P: "patch",
}
LEVEL_NAMES: dict[str, str] = {lev.value: lev.label for lev in LEVELS}


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
    "istsoil": 1,
    "istcrop": 2,
    "istice": 3,
    "istice_mec": 4,
    "istdlak": 5,
    "istwet": 6,
    "isturb_tbd": 7,
    "isturb_hd": 8,
    "isturb_md": 9,
    "istlowcenpoly": 10,
    "istflatcenpoly": 11,
    "isthighcenpoly": 12,
    "icol_roof": 71,
    "icol_sunwall": 72,
    "icol_shadewall": 73,
    "icol_road_imperv": 74,
    "icol_road_perv": 75,
    "noveg": 0,
}
# the columns of an urban landunit, in initGridCellsMod's order (set_landunit_urban)
URBAN_COLUMNS = (
    "icol_roof",
    "icol_sunwall",
    "icol_shadewall",
    "icol_road_imperv",
    "icol_road_perv",
)
SOIL_COLUMN = 1  # initGridCellsMod: the natural-vegetation column's type


def check_type_codes(params: dict[str, Any]) -> None:
    """TYPE_CODES agree with the case's module parameters where the analysis has them."""
    for name, value in TYPE_CODES.items():
        have = params.get(name)
        if isinstance(have, int) and have != value:
            raise HierarchyError(
                f"the case has {name} = {have}; the synthetic domain assumes {value} "
                "(update TYPE_CODES in spel/proptest/hierarchy/codes.py)"
            )


def mec_column(m: int) -> int:
    """column_varcon::icemec_class_to_col_itype."""
    return TYPE_CODES["istice_mec"] * 100 + m


def crop_column(cft: int) -> int:
    """initGridCellsMod: a CFT's column type (istcrop*100 + cft)."""
    return TYPE_CODES["istcrop"] * 100 + cft


def column_role(itype: int) -> str:
    """A column type's name: soil, crop_<cft>, glacier_mec_<class>, lake, wetland, glacier, roof, ... ."""
    for key in URBAN_COLUMNS:
        if TYPE_CODES[key] == itype:
            return key.removeprefix("icol_")
    if itype == SOIL_COLUMN:
        return "soil"
    named = {
        TYPE_CODES["istdlak"]: "lake",
        TYPE_CODES["istwet"]: "wetland",
        TYPE_CODES["istice"]: "glacier",
    }
    if itype in named:
        return named[itype]
    if itype > mec_column(0):
        return f"glacier_mec_{itype - mec_column(0)}"
    if itype > crop_column(0):
        return f"crop_{itype - crop_column(0)}"
    return str(itype)


class Flag(Enum):
    """A logical subgrid flag ELM sets from the landunit kind (initSubgridMod's add_landunit,
    add_column, add_patch). The value is its Fortran name; it lives at `level` and holds for
    elements of `kinds` (finer elements take their ancestor's value)."""

    IFSPECIAL = "ifspecial"
    LAKPOI = "lakpoi"
    URBPOI = "urbpoi"
    GLCMECPOI = "glcmecpoi"
    ISPOLYGON = "ispolygon"
    IS_SOIL = "is_soil"
    IS_CROP = "is_crop"
    IS_LAKE = "is_lake"
    IS_ON_SOIL_COL = "is_on_soil_col"
    IS_ON_CROP_COL = "is_on_crop_col"

    @property
    def level(self) -> Level:
        return _FLAGS[self][0]

    @property
    def kinds(self) -> tuple[LandunitKind, ...]:
        return _FLAGS[self][1]


_SOIL_KINDS = (LandunitKind.SOIL, LandunitKind.POLYGON)

_FLAGS: dict[Flag, tuple[Level, tuple[LandunitKind, ...]]] = {
    Flag.IFSPECIAL: (Level.L, tuple(k for k in LandunitKind if k.special)),
    Flag.LAKPOI: (Level.L, (LandunitKind.LAKE,)),
    Flag.URBPOI: (Level.L, (LandunitKind.URBAN,)),
    Flag.GLCMECPOI: (Level.L, (LandunitKind.GLACIER_MEC,)),
    Flag.ISPOLYGON: (Level.L, (LandunitKind.POLYGON,)),
    Flag.IS_SOIL: (Level.C, _SOIL_KINDS),
    Flag.IS_CROP: (Level.C, (LandunitKind.CROP,)),
    Flag.IS_LAKE: (Level.C, (LandunitKind.LAKE,)),
    Flag.IS_ON_SOIL_COL: (Level.P, _SOIL_KINDS),
    Flag.IS_ON_CROP_COL: (Level.P, (LandunitKind.CROP,)),
}
# the flags by level, as {name: kinds}
LANDUNIT_FLAGS = {f.value: f.kinds for f in Flag if f.level is Level.L}
COLUMN_FLAGS = {f.value: f.kinds for f in Flag if f.level is Level.C}
PATCH_FLAGS = {f.value: f.kinds for f in Flag if f.level is Level.P}
FLAG_LEVEL = {f.value: f.level.value for f in Flag}
