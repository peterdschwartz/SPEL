"""Declarations of a domain: gridcells, topounits, landunits (one class per kind), columns, patches.

Each landunit kind is a class that stores what the user declares (a soil landunit its column of
pfts, an urban landunit its density and canyon geometry, a glacier-mec landunit its classes) and
derives its ELM columns (`columns`), type codes, label, ELM layout order, TOML table and the
elements a `Subgrid` completes it with. `Landunit.from_table` dispatches on `kind` through
`LANDUNITS`. Declarations are frozen; change them with `with_weight`, `with_patches`,
`with_landunit`, ... (not `dataclasses.replace`, which the vararg constructors don't take).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, ClassVar, Mapping, Optional, Sequence

from spel.proptest.hierarchy._table import check_keys, resolve_weights, weight_of
from spel.proptest.hierarchy.codes import (
    ALL_GLACIER_TOL,
    MAX_NON_POLY_LUNIT,
    SOIL_COLUMN,
    TYPE_CODES,
    URBAN_COLUMNS,
    HierarchyError,
    LandunitKind,
    PolygonType,
    UrbanDensity,
    crop_column,
    mec_column,
)

if TYPE_CHECKING:
    from spel.proptest.hierarchy.subgrid import Subgrid


@dataclass(frozen=True)
class Patch:
    pft: int
    weight: Optional[float] = None  # in its column

    def __post_init__(self) -> None:
        # a `Pft` member is stored as its index: Pft enums are built per srcroot and don't pickle
        if type(self.pft) is not int and not isinstance(self.pft, bool):
            object.__setattr__(self, "pft", int(self.pft))

    def to_table(self) -> Any:
        return (
            self.pft
            if self.weight is None
            else {"pft": self.pft, "weight": self.weight}
        )

    @classmethod
    def from_table(cls, value: Any, where: str) -> "Patch":
        if isinstance(value, bool) or not isinstance(value, (int, dict)):
            raise HierarchyError(
                f"{where}: a patch is a pft number or {{ pft, weight }}, not {value!r}"
            )
        if isinstance(value, int):
            return cls(value)
        check_keys(value, {"pft", "weight"}, where)
        if "pft" not in value:
            raise HierarchyError(f"{where}: `pft` is required")
        return cls(int(value["pft"]), weight_of(value, where))


@dataclass(frozen=True)
class Column:
    itype: int
    patches: tuple[Patch, ...]
    weight: Optional[float] = None  # in its landunit
    topo: Optional[float] = None  # glacier-mec: col_pp%glc_topo (m)

    def to_table(self) -> dict:
        out: dict[str, Any] = {
            "itype": self.itype,
            "patches": [p.to_table() for p in self.patches],
        }
        if self.weight is not None:
            out["weight"] = self.weight
        return out

    @classmethod
    def from_table(cls, value: Any, where: str, itype: int) -> "Column":
        if not isinstance(value, dict):
            raise HierarchyError(
                f"{where}: a column is a table {{ patches = [...], weight }}"
            )
        check_keys(value, {"itype", "patches", "weight"}, where)
        patches = tuple(
            Patch.from_table(p, f"{where}.patches[{k}]")
            for k, p in enumerate(value.get("patches", []))
        )
        return cls(int(value.get("itype", itype)), patches, weight_of(value, where))


@dataclass(frozen=True)
class MecClass:
    """A glacier elevation class m of a glacier-mec landunit: its column's weight in the landunit
    and, optionally, its col_pp%glc_topo."""

    index: int
    weight: Optional[float] = None
    topo: Optional[float] = None

    @property
    def itype(self) -> int:
        return mec_column(self.index)


def _patch(p: Patch | int) -> Patch:
    return p if isinstance(p, Patch) else Patch(int(p))


# -- landunits

LANDUNITS: dict[LandunitKind, type["Landunit"]] = {}


@dataclass(frozen=True, init=False)
class Landunit:
    """A landunit of one kind (a subclass per kind). Every kind has:

    - `weight` in its topounit (None: shares what its siblings leave);
    - `columns`: its ELM columns (stored for soil and polygons, derived for the others);
    - `ltype` (lun_pp%itype), `index_type` (its row of landunit_indices), `label`, `elm_order`
      (initGridCellsMod's pass, then the order within it);
    - `to_table`/`from_table` (TOML), `check`, and `require`/`complete` (ELM's implicit elements
      under a `Subgrid`).
    """

    kind: ClassVar[LandunitKind]
    ltype_key: ClassVar[str]  # TYPE_CODES key of lun_pp%itype
    elm_pass: ClassVar[int]  # initGridCellsMod's landunit pass

    weight: Optional[float]

    def __init_subclass__(cls, **kw: Any) -> None:
        super().__init_subclass__(**kw)
        if "kind" in cls.__dict__:
            LANDUNITS[cls.kind] = cls

    def _set(self, **values: Any) -> None:
        for key, value in values.items():
            object.__setattr__(self, key, value)

    # -- ELM attributes
    @property
    def ltype(self) -> int:
        """lun_pp%itype (polygons are istsoil)."""
        return TYPE_CODES[self.ltype_key]

    @property
    def index_type(self) -> int:
        """Its key in landunit_indices: the ltype, polygontype + max_non_poly_lunit for polygons."""
        return self.ltype

    @property
    def label(self) -> str:
        return self.kind.value

    @property
    def elm_order(self) -> tuple[int, int]:
        """(initGridCellsMod's pass over the clump's topounits, the order within a topounit)."""
        return self.elm_pass, 0

    # -- changes
    def with_weight(self, weight: Optional[float]) -> "Landunit":
        out = copy.copy(self)
        out._set(weight=weight)
        return out

    # -- checks
    def check(self, where: str) -> "Landunit":
        """The landunit, after checking that it has columns and patches whose weights can sum to 1."""
        cols = self.columns
        if not cols:
            raise HierarchyError(f"{where}: no column")
        resolve_weights([c.weight for c in cols], f"{where}.columns")
        for k, c in enumerate(cols):
            if not c.patches:
                raise HierarchyError(f"{where}.column[{k}]: no patch")
            resolve_weights(
                [p.weight for p in c.patches], f"{where}.column[{k}].patches"
            )
        return self

    # -- ELM's implicit elements (subgridMod)
    def require(self, subgrid: "Subgrid", glcmask: bool, where: str) -> None:
        """Whether ELM can create this landunit under the subgrid (raises if not)."""

    def complete(self, subgrid: "Subgrid", glcmask: bool, where: str) -> "Landunit":
        """The landunit with every column and patch ELM creates under the subgrid."""
        return self

    # -- TOML
    def to_table(self) -> dict:
        out: dict[str, Any] = {"kind": self.kind.value, **self._table()}
        if self.weight is not None:
            out["weight"] = self.weight
        return out

    def _table(self) -> dict:
        """The kind's own keys (between `kind` and `weight`)."""
        return {}

    @classmethod
    def from_table(cls, value: Any, where: str) -> "Landunit":
        """A landunit of any kind from its table (dispatched on `kind`)."""
        kinds = [k.value for k in LandunitKind]
        if not isinstance(value, dict) or "kind" not in value:
            raise HierarchyError(
                f"{where}: a landunit is a table with `kind` ({kinds})"
            )
        try:
            kind = LandunitKind(str(value["kind"]).lower())
        except ValueError:
            raise HierarchyError(
                f"{where}: kind must be one of {kinds}, not {value['kind']!r}"
            ) from None
        return LANDUNITS[kind]._from_table(value, where, weight_of(value, where))

    @classmethod
    def _from_table(
        cls, value: dict, where: str, weight: Optional[float]
    ) -> "Landunit":
        check_keys(value, {"kind", "weight"}, where)
        return cls(weight=weight)

    @staticmethod
    def from_columns(
        kind: LandunitKind,
        columns: Sequence[Column],
        weight: Optional[float],
        density: Optional[UrbanDensity] = None,
        polygon: Optional[PolygonType] = None,
    ) -> "Landunit":
        """A landunit from ELM's columns (a restart file's, experiments/tools/elm_layout_oracle.py):
        soil and polygon columns as given, a crop column's first patch as its CFT, glacier-mec
        classes from the column types, the urban geometry from the roof and road weights.
        """
        cols = tuple(columns)
        if kind is LandunitKind.SOIL:
            return Soil.of_columns(cols, weight)
        if kind is LandunitKind.POLYGON:
            assert polygon is not None
            return Polygon.of_columns(polygon, cols, weight)
        if kind is LandunitKind.CROP:
            return Crop(
                *(Patch(c.patches[0].pft, c.weight) for c in cols), weight=weight
            )
        if kind is LandunitKind.URBAN:
            assert density is not None
            w = [c.weight or 0.0 for c in cols]
            road = w[3] + w[4]
            return Urban(
                density, weight, roof=w[0], road_perv=w[4] / road if road > 0 else 0.5
            )
        if kind is LandunitKind.GLACIER_MEC:
            return GlacierMec(
                [MecClass(c.itype - mec_column(0), c.weight, c.topo) for c in cols],
                weight,
            )
        return LANDUNITS[kind](weight=weight)


@dataclass(frozen=True, init=False)
class _NaturalVegetation(Landunit):
    """istsoil landunits (soil, polygons): one column of type 1 with one patch per pft (or
    explicit columns, from TOML's `columns = [...]`)."""

    ltype_key: ClassVar[str] = "istsoil"
    elm_pass: ClassVar[int] = 0

    columns: tuple[Column, ...]

    def with_columns(self, columns: Sequence[Column]) -> "_NaturalVegetation":
        out = copy.copy(self)
        out._set(columns=tuple(columns))
        return out

    def with_patches(
        self, patches: Sequence[Patch], column: int = 0
    ) -> "_NaturalVegetation":
        """The landunit with column `column` holding these patches (the other columns unchanged)."""
        cols = list(self.columns)
        cols[column] = replace(cols[column], patches=tuple(patches))
        return self.with_columns(cols)

    def complete(self, subgrid: "Subgrid", glcmask: bool, where: str) -> "Landunit":
        return self.with_columns(
            tuple(
                replace(c, patches=subgrid.natural_patches(c.patches, where))
                for c in self.columns
            )
        )

    def _table(self) -> dict:
        cols = self.columns
        if len(cols) == 1 and cols[0].weight is None and cols[0].itype == SOIL_COLUMN:
            return {"patches": [p.to_table() for p in cols[0].patches]}
        return {"columns": [c.to_table() for c in cols]}

    @staticmethod
    def _read_columns(
        value: dict, where: str
    ) -> tuple[Optional[tuple[Patch, ...]], tuple[Column, ...]]:
        """(patches, None) or (None, columns) from a table with one of `patches`/`columns`."""
        if ("patches" in value) == ("columns" in value):
            raise HierarchyError(
                f"{where}: a {value['kind']} landunit takes `patches` (one column) or `columns`"
            )
        if "patches" in value:
            return (
                tuple(
                    Patch.from_table(p, f"{where}.patches[{k}]")
                    for k, p in enumerate(value["patches"])
                ),
                (),
            )
        return None, tuple(
            Column.from_table(c, f"{where}.columns[{k}]", SOIL_COLUMN)
            for k, c in enumerate(value["columns"])
        )


@dataclass(frozen=True, init=False)
class Soil(_NaturalVegetation):
    """istsoil: 1 SOIL_COLUMN column  with one patch per pft: `Soil(1, Patch(13, 0.4), weight=0.5)`."""

    kind: ClassVar[LandunitKind] = LandunitKind.SOIL

    def __init__(self, *patches: Patch | int, weight: Optional[float] = None):
        self._set(
            weight=weight,
            columns=(Column(SOIL_COLUMN, tuple(_patch(p) for p in patches)),),
        )

    @classmethod
    def of_columns(
        cls, columns: Sequence[Column], weight: Optional[float] = None
    ) -> "Soil":
        return cls(weight=weight).with_columns(columns)  # type: ignore[return-value]

    @classmethod
    def _from_table(
        cls, value: dict, where: str, weight: Optional[float]
    ) -> "Landunit":
        check_keys(value, {"kind", "weight", "patches", "columns"}, where)
        patches, cols = cls._read_columns(value, where)
        if patches is not None:
            return cls(*patches, weight=weight)
        return cls.of_columns(cols, weight).check(where)


@dataclass(frozen=True, init=False)
class Polygon(_NaturalVegetation):
    """A polygonal-tundra landunit (add_polygon_landunit): itype istsoil, one soil column with
    one patch per pft, `polygontype` low/flat/high: `Polygon("low", 12)`."""

    kind: ClassVar[LandunitKind] = LandunitKind.POLYGON

    polygon: PolygonType

    def __init__(
        self,
        polygon: PolygonType | str,
        *patches: Patch | int,
        weight: Optional[float] = None,
    ):
        poly = PolygonType(polygon.lower() if isinstance(polygon, str) else polygon)
        self._set(
            weight=weight,
            columns=(Column(SOIL_COLUMN, tuple(_patch(p) for p in patches)),),
            polygon=poly,
        )

    @classmethod
    def of_columns(
        cls,
        polygon: PolygonType | str,
        columns: Sequence[Column],
        weight: Optional[float] = None,
    ) -> "Polygon":
        return cls(polygon, weight=weight).with_columns(columns)  # type: ignore[return-value]

    @property
    def index_type(self) -> int:
        return self.polygon.code + MAX_NON_POLY_LUNIT

    @property
    def label(self) -> str:
        return f"polygon {self.polygon.value}"

    @property
    def elm_order(self) -> tuple[int, int]:
        return self.elm_pass, self.polygon.code

    def _table(self) -> dict:
        return {"type": self.polygon.value, **super()._table()}

    @classmethod
    def _from_table(
        cls, value: dict, where: str, weight: Optional[float]
    ) -> "Landunit":
        try:
            poly = PolygonType(str(value.get("type", "")).lower())
        except ValueError:
            raise HierarchyError(
                f"{where}: a polygon's type is one of {[t.value for t in PolygonType]}"
            ) from None
        check_keys(value, {"kind", "weight", "patches", "columns", "type"}, where)
        patches, cols = cls._read_columns(value, where)
        if patches is not None:
            return cls(poly, *patches, weight=weight)
        return cls.of_columns(poly, cols, weight).check(where)


@dataclass(frozen=True, init=False)
class Crop(Landunit):
    """istcrop: one column per CFT (itype istcrop*100 + cft) with one patch; a CFT's weight is its
    column's weight in the landunit: `Crop(17, Patch(19, 0.2))`."""

    kind: ClassVar[LandunitKind] = LandunitKind.CROP
    ltype_key: ClassVar[str] = "istcrop"
    elm_pass: ClassVar[int] = 1

    cfts: tuple[Patch, ...]

    def __init__(self, *cfts: Patch | int, weight: Optional[float] = None):
        self._set(weight=weight, cfts=tuple(_patch(c) for c in cfts))

    @property
    def columns(self) -> tuple[Column, ...]:
        return tuple(
            Column(crop_column(c.pft), (Patch(c.pft),), c.weight) for c in self.cfts
        )

    def require(self, subgrid: "Subgrid", glcmask: bool, where: str) -> None:
        if subgrid.cft is None:
            raise HierarchyError(
                f"{where}: a crop landunit needs create_crop_landunit (give `cft` in the subgrid)"
            )

    def complete(self, subgrid: "Subgrid", glcmask: bool, where: str) -> "Landunit":
        assert subgrid.cft is not None
        lb, ub = subgrid.cft
        have = {c.pft: c for c in self.cfts}
        bad = sorted(set(have) - set(range(lb, ub + 1)))
        if bad:
            raise HierarchyError(
                f"{where}: CFTs {bad} are outside cft_lb:cft_ub = {subgrid.cft}"
            )
        cw = dict(
            zip(have, resolve_weights([c.weight for c in self.cfts], f"{where}.cfts"))
        )
        return Crop(
            *(Patch(m, cw.get(m, 0.0)) for m in range(lb, ub + 1)), weight=self.weight
        )

    def _table(self) -> dict:
        return {"cfts": [c.to_table() for c in self.cfts]}

    @classmethod
    def _from_table(
        cls, value: dict, where: str, weight: Optional[float]
    ) -> "Landunit":
        check_keys(value, {"kind", "weight", "cfts"}, where)
        return cls(
            *(
                Patch.from_table(p, f"{where}.cfts[{k}]")
                for k, p in enumerate(value.get("cfts", []))
            ),
            weight=weight,
        )


# initGridCellsMod's urban passes, by density
_URBAN_PASS = {UrbanDensity.TBD: 2, UrbanDensity.HD: 3, UrbanDensity.MD: 4}


def _urban_weights(roof: float, road_perv: float) -> tuple[float, ...]:
    if not (0.0 <= roof <= 1.0 and 0.0 <= road_perv <= 1.0):
        raise HierarchyError(
            f"urban: roof ({roof}) and road_perv ({road_perv}) are fractions in [0, 1]"
        )
    canyon = (1.0 - roof) / 3.0
    return (roof, canyon, canyon, canyon * (1.0 - road_perv), canyon * road_perv)


@dataclass(frozen=True, init=False)
class Urban(Landunit):
    """An urban landunit of a density: the five columns with set_landunit_urban's weights for its
    roof fraction (wtlunit_roof) and pervious-road fraction (wtroad_perv), one bare patch each.
    """

    kind: ClassVar[LandunitKind] = LandunitKind.URBAN

    density: UrbanDensity
    roof: float
    road_perv: float

    def __init__(
        self,
        density: UrbanDensity | str,
        weight: Optional[float] = None,
        roof: float = 0.5,
        road_perv: float = 0.5,
    ):
        dens = UrbanDensity(density.lower() if isinstance(density, str) else density)
        _urban_weights(roof, road_perv)
        self._set(weight=weight, density=dens, roof=roof, road_perv=road_perv)

    @property
    def columns(self) -> tuple[Column, ...]:
        noveg = Patch(TYPE_CODES["noveg"])
        return tuple(
            Column(TYPE_CODES[k], (noveg,), w)
            for k, w in zip(URBAN_COLUMNS, _urban_weights(self.roof, self.road_perv))
        )

    @property
    def ltype(self) -> int:
        return TYPE_CODES[self.density.type_code]

    @property
    def label(self) -> str:
        return f"urban {self.density.value}"

    @property
    def elm_order(self) -> tuple[int, int]:
        return _URBAN_PASS[self.density], 0

    def _table(self) -> dict:
        out: dict[str, Any] = {"density": self.density.value}
        if (self.roof, self.road_perv) != (0.5, 0.5):
            out.update(roof=self.roof, road_perv=self.road_perv)
        return out

    @classmethod
    def _from_table(
        cls, value: dict, where: str, weight: Optional[float]
    ) -> "Landunit":
        check_keys(value, {"kind", "weight", "density", "roof", "road_perv"}, where)
        try:
            density = UrbanDensity(str(value.get("density", "")).lower())
        except ValueError:
            raise HierarchyError(
                f"{where}: density must be one of {[d.value for d in UrbanDensity]}"
            ) from None
        return cls(
            density,
            weight=weight,
            roof=float(value.get("roof", 0.5)),
            road_perv=float(value.get("road_perv", 0.5)),
        )


@dataclass(frozen=True, init=False)
class _SingleColumn(Landunit):
    """Lake, wetland, plain glacier (set_landunit_wet_ice_lake): one column of the landunit's type
    and one bare patch. ELM creates them only where their weight is > 0."""

    def __init__(self, weight: Optional[float] = None):
        self._set(weight=weight)

    @property
    def columns(self) -> tuple[Column, ...]:
        return (Column(self.ltype, (Patch(TYPE_CODES["noveg"]),)),)

    def require(self, subgrid: "Subgrid", glcmask: bool, where: str) -> None:
        if self.weight == 0.0:
            raise HierarchyError(
                f"{where}: ELM creates no {self.kind.value} landunit of weight 0 (subgridMod); "
                "leave it out"
            )


@dataclass(frozen=True, init=False)
class Lake(_SingleColumn):
    """istdlak: deep lake."""

    kind: ClassVar[LandunitKind] = LandunitKind.LAKE
    ltype_key: ClassVar[str] = "istdlak"
    elm_pass: ClassVar[int] = 5


@dataclass(frozen=True, init=False)
class Wetland(_SingleColumn):
    """istwet."""

    kind: ClassVar[LandunitKind] = LandunitKind.WETLAND
    ltype_key: ClassVar[str] = "istwet"
    elm_pass: ClassVar[int] = 6


@dataclass(frozen=True, init=False)
class Glacier(_SingleColumn):
    """istice: plain glacier."""

    kind: ClassVar[LandunitKind] = LandunitKind.GLACIER
    ltype_key: ClassVar[str] = "istice"
    elm_pass: ClassVar[int] = 7


MecClasses = (
    Mapping[int, Optional[float] | tuple[Optional[float], Optional[float]]]
    | Sequence[int | MecClass]
)


@dataclass(frozen=True, init=False)
class GlacierMec(Landunit):
    """istice_mec: one column per elevation class m (itype istice_mec*100 + m) in class order, one
    bare patch each. `classes` maps a class to its weight in the landunit or to (weight, glc_topo),
    or lists classes (ints share the weight equally, or `MecClass`es)."""

    kind: ClassVar[LandunitKind] = LandunitKind.GLACIER_MEC
    ltype_key: ClassVar[str] = "istice_mec"
    elm_pass: ClassVar[int] = 8

    classes: tuple[MecClass, ...]

    def __init__(self, classes: MecClasses, weight: Optional[float] = None):
        if isinstance(classes, Mapping):
            items = []
            for m, v in classes.items():
                w, topo = v if isinstance(v, tuple) else (v, None)
                items.append(MecClass(int(m), w, topo))
        else:
            items = [
                c if isinstance(c, MecClass) else MecClass(int(c)) for c in classes
            ]
        for c in items:
            if c.index < 1:
                raise HierarchyError(f"glacier_mec: classes start at 1, not {c.index}")
        self._set(weight=weight, classes=tuple(sorted(items, key=lambda c: c.index)))

    @property
    def columns(self) -> tuple[Column, ...]:
        noveg = Patch(TYPE_CODES["noveg"])
        return tuple(Column(c.itype, (noveg,), c.weight, c.topo) for c in self.classes)

    def check(self, where: str) -> "Landunit":
        super().check(where)
        idx = [c.index for c in self.classes]
        if len(set(idx)) != len(idx):
            raise HierarchyError(f"{where}: a glacier-mec class appears twice")
        return self

    def require(self, subgrid: "Subgrid", glcmask: bool, where: str) -> None:
        if not glcmask and self.weight == 0.0:
            raise HierarchyError(
                f"{where}: a glacier-mec landunit of weight 0 exists only in a glcmask gridcell"
            )

    def complete(self, subgrid: "Subgrid", glcmask: bool, where: str) -> "Landunit":
        """In a glcmask gridcell, every class 1..maxpatch_glcmec exists."""
        if not glcmask:
            return self
        n = subgrid.maxpatch_glcmec
        if not n:
            raise HierarchyError(
                f"{where}: a glcmask gridcell needs maxpatch_glcmec in the subgrid"
            )
        have = {c.index: c for c in self.classes}
        bad = sorted(m for m in have if not 1 <= m <= n)
        if bad:
            raise HierarchyError(
                f"{where}: classes {bad} are outside 1..maxpatch_glcmec = {n}"
            )
        cw = dict(
            zip(
                have,
                resolve_weights([c.weight for c in self.classes], f"{where}.classes"),
            )
        )
        return GlacierMec(
            [
                MecClass(m, cw.get(m, 0.0), have[m].topo if m in have else None)
                for m in range(1, n + 1)
            ],
            weight=self.weight,
        )

    def _table(self) -> dict:
        rows = []
        for c in self.classes:
            row: dict[str, Any] = {"class": c.index}
            if c.weight is not None:
                row["weight"] = c.weight
            if c.topo is not None:
                row["topo"] = c.topo
            rows.append(row)
        return {"classes": rows}

    @classmethod
    def _from_table(
        cls, value: dict, where: str, weight: Optional[float]
    ) -> "Landunit":
        check_keys(value, {"kind", "weight", "classes"}, where)
        classes: dict[int, MecClass] = {}
        for k, row in enumerate(value.get("classes", [])):
            w = f"{where}.classes[{k}]"
            if not isinstance(row, dict) or "class" not in row:
                raise HierarchyError(
                    f"{w}: a class is a table {{ class, weight, topo }}"
                )
            check_keys(row, {"class", "weight", "topo"}, w)
            topo = row.get("topo")
            m = int(row["class"])
            classes[m] = MecClass(
                m, weight_of(row, w), None if topo is None else float(topo)
            )
        return cls(list(classes.values()), weight=weight)


# -- topounits and gridcells


@dataclass(frozen=True)
class Topounit:
    landunits: tuple[Landunit, ...]
    weight: Optional[float] = None  # in its gridcell; 0: inactive
    elevation: Optional[float] = (
        None  # use_IM2_hillslope_hydrology: downhill_ti from these
    )

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
            raise HierarchyError(
                f"{where}: a topounit is a table {{ weight, elevation, landunits = [...] }}"
            )
        check_keys(value, {"weight", "elevation", "landunits"}, where)
        lus = tuple(
            Landunit.from_table(x, f"{where}.landunits[{k}]")
            for k, x in enumerate(value.get("landunits", []))
        )
        elev = value.get("elevation")
        return cls(lus, weight_of(value, where), None if elev is None else float(elev))

    @property
    def all_glacier(self) -> bool:
        """subgridWeightsMod::is_topo_all_gla: plain glacier covers the topounit."""
        w = resolve_weights([lu.weight for lu in self.landunits], "landunits")
        return (
            sum(x for lu, x in zip(self.landunits, w) if isinstance(lu, Glacier))
            >= 1.0 - ALL_GLACIER_TOL
        )

    def with_landunit(self, m: int, landunit: Landunit) -> "Topounit":
        return replace(
            self, landunits=self.landunits[:m] + (landunit,) + self.landunits[m + 1 :]
        )

    def without_landunit(self, m: int) -> "Topounit":
        return replace(self, landunits=self.landunits[:m] + self.landunits[m + 1 :])

    @property
    def size(self) -> int:
        """Its elements over all levels."""
        return 1 + sum(
            1 + sum(1 + len(c.patches) for c in lu.columns) for lu in self.landunits
        )


@dataclass(frozen=True)
class Gridcell:
    topounits: tuple[Topounit, ...]
    copies: int = 1
    name: str = ""  # reports, queries; relations permute gridcells of the same name
    glcmask: bool = (
        False  # ldomain%glcmask: glacier-mec columns of every class exist and are active
    )
    icemask: float = 0.0  # icemask_grc (do_smb_c)

    @classmethod
    def of(
        cls,
        *landunits: Landunit,
        copies: int = 1,
        name: str = "",
        glcmask: bool = False,
        icemask: float = 0.0,
    ) -> "Gridcell":
        """A gridcell with one topounit."""
        return cls((Topounit(tuple(landunits)),), copies, name, glcmask, icemask)

    @classmethod
    def hillslope(
        cls,
        profile: Sequence[tuple[float, float]],
        *landunits: Landunit,
        copies: int = 1,
        name: str = "",
        glcmask: bool = False,
        icemask: float = 0.0,
    ) -> "Gridcell":
        """A gridcell with one topounit per (elevation, weight) of `profile`, in that order, each
        holding `landunits`: `Gridcell.hillslope([(50, 0.3), (150, 0.7)], Soil(13), name="hill")`.
        """
        if not profile:
            raise HierarchyError(
                f"Gridcell.hillslope {name!r}: the profile has no topounit"
            )
        if not landunits:
            raise HierarchyError(
                f"Gridcell.hillslope {name!r}: give the landunits of each topounit"
            )
        tops = tuple(
            Topounit(tuple(landunits), weight=float(w), elevation=float(e))
            for e, w in profile
        )
        return cls(tops, copies, name, glcmask, icemask)

    @property
    def label(self) -> str:
        if self.name:
            return self.name
        return "+".join(lu.label for t in self.topounits for lu in t.landunits)

    @property
    def size(self) -> int:
        """Its elements over all levels (one copy)."""
        return 1 + sum(t.size for t in self.topounits)

    def with_topounit(self, k: int, topounit: Topounit) -> "Gridcell":
        return replace(
            self, topounits=self.topounits[:k] + (topounit,) + self.topounits[k + 1 :]
        )

    def without_topounit(self, k: int) -> "Gridcell":
        return replace(self, topounits=self.topounits[:k] + self.topounits[k + 1 :])

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
        if (
            len(self.topounits) == 1
            and self.topounits[0].weight is None
            and self.topounits[0].elevation is None
        ):
            out["landunits"] = [lu.to_table() for lu in self.topounits[0].landunits]
        else:
            out["topounits"] = [t.to_table() for t in self.topounits]
        return out

    @classmethod
    def from_table(cls, value: Any, where: str) -> "Gridcell":
        if not isinstance(value, dict):
            raise HierarchyError(f"{where}: a gridcell is a table")
        check_keys(
            value,
            {"name", "copies", "landunits", "topounits", "glcmask", "icemask"},
            where,
        )
        if ("landunits" in value) == ("topounits" in value):
            raise HierarchyError(
                f"{where}: give `landunits` (one topounit) or `topounits`"
            )
        if "landunits" in value:
            tops = (
                Topounit(
                    tuple(
                        Landunit.from_table(x, f"{where}.landunits[{k}]")
                        for k, x in enumerate(value["landunits"])
                    )
                ),
            )
        else:
            tops = tuple(
                Topounit.from_table(t, f"{where}.topounits[{k}]")
                for k, t in enumerate(value["topounits"])
            )
        g = cls(
            tops,
            int(value.get("copies", 1)),
            str(value.get("name", "")),
            bool(value.get("glcmask", False)),
            float(value.get("icemask", 0.0)),
        )
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
                lu.check(f"{tw}.landunit[{j}]")
                if lu.index_type in seen:
                    raise HierarchyError(
                        f"{tw}: landunits {seen[lu.index_type]} and {j} are both {lu.label} "
                        "(ELM creates one landunit of each type per topounit)"
                    )
                seen[lu.index_type] = j
        return self


def ltype_of(kind: LandunitKind, density: Optional[UrbanDensity] = None) -> int:
    """lun_pp%itype of a landunit kind (urban: of a density; polygons are istsoil)."""
    if kind is LandunitKind.URBAN:
        assert density is not None
        return TYPE_CODES[density.type_code]
    return TYPE_CODES[LANDUNITS[kind].ltype_key]
