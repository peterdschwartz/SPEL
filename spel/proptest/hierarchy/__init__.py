"""ELM's subgrid hierarchy for synthetic domains: declarations and their layout.

A domain is declared as gridcells, each with topounits, landunits, columns and patches, every
element with a weight in its parent (`Gridcell`, `Topounit`, `Landunit`, `Column`, `Patch`). Each landunit kind is a class
(decl.py) following initGridCellsMod's set_landunit_*:

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

Modules: `codes` (`Level`, `LandunitKind`, `Flag`, ELM's type codes), `decl` (the declarations, a
`Landunit` subclass per kind), `subgrid` (`Subgrid`), `layout` (`Layout`, one `LevelTable` per
level), `views` (`Layout.tree()`'s element views). Everything is importable from this package.
"""

from spel.proptest.hierarchy._table import resolve_weights
from spel.proptest.hierarchy.codes import (
    ALL_GLACIER_TOL,
    COLUMN_FLAGS,
    FLAG_LEVEL,
    ISPVAL,
    LANDUNIT_FLAGS,
    LEVEL_NAMES,
    LEVELS,
    MAX_LUNIT,
    MAX_NON_POLY_LUNIT,
    PATCH_FLAGS,
    SOIL_COLUMN,
    TYPE_CODES,
    URBAN_COLUMNS,
    WEIGHT_TOL,
    Flag,
    HierarchyError,
    LandunitKind,
    Level,
    PolygonType,
    UrbanDensity,
    check_type_codes,
    column_role,
    crop_column,
    mec_column,
)
from spel.proptest.hierarchy.decl import (
    LANDUNITS,
    Column,
    Crop,
    Glacier,
    GlacierMec,
    Gridcell,
    Lake,
    Landunit,
    MecClass,
    Patch,
    Polygon,
    Soil,
    Topounit,
    Urban,
    Wetland,
    ltype_of,
)
from spel.proptest.hierarchy.layout import DOWNHILL_RULES, ORDERS, Layout, LevelTable
from spel.proptest.hierarchy.pfts import PftBase, pft_enum, pft_names
from spel.proptest.hierarchy.subgrid import Subgrid
from spel.proptest.hierarchy.views import ColumnView, GridcellView, LandunitView, PatchView, TopounitView

__all__ = [
    "ALL_GLACIER_TOL", "COLUMN_FLAGS", "FLAG_LEVEL", "ISPVAL", "LANDUNIT_FLAGS", "LEVEL_NAMES", "LEVELS",
    "MAX_LUNIT", "MAX_NON_POLY_LUNIT", "PATCH_FLAGS", "SOIL_COLUMN", "TYPE_CODES", "URBAN_COLUMNS", "WEIGHT_TOL",
    "Flag", "HierarchyError", "LandunitKind", "Level", "PolygonType", "UrbanDensity", "check_type_codes",
    "column_role", "crop_column", "mec_column", "LANDUNITS", "Column", "Crop", "Glacier", "GlacierMec", "Gridcell",
    "Lake", "Landunit", "MecClass", "Patch", "Polygon", "Soil", "Topounit", "Urban", "Wetland", "ltype_of",
    "DOWNHILL_RULES", "ORDERS", "Layout", "LevelTable", "Subgrid", "ColumnView", "GridcellView", "LandunitView", "PatchView",
    "TopounitView", "resolve_weights", "Pft", "PftBase", "pft_enum", "pft_names",
]


def __getattr__(name: str):
    # `Pft` is read from the configured srcroot's pftvarcon.F90 on first use, not at import
    if name == "Pft":
        return pft_enum()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
