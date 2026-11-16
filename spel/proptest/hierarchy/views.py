"""A laid-out domain as a tree of its elements, as ELM holds them (`Layout.tree()`): Fortran
indices, weights in the parent, activity, and the declaration each element comes from."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from spel.proptest.hierarchy.codes import LandunitKind, UrbanDensity
from spel.proptest.hierarchy.decl import Landunit


@dataclass(frozen=True)
class PatchView:
    index: int
    pft: int
    weight: float
    wtgcell: float
    active: bool


@dataclass(frozen=True)
class ColumnView:
    index: int
    itype: int
    role: str
    weight: float
    active: bool
    patches: tuple[PatchView, ...]


@dataclass(frozen=True)
class LandunitView:
    index: int
    kind: LandunitKind
    density: Optional[UrbanDensity]
    itype: int
    weight: float
    active: bool
    columns: tuple[ColumnView, ...]
    decl: Optional[Landunit] = field(default=None, repr=False, compare=False)  # its declaration


@dataclass(frozen=True)
class TopounitView:
    index: int
    number: int  # within its gridcell (topo_grc_ind)
    weight: float
    active: bool
    elevation: Optional[float]
    landunits: tuple[LandunitView, ...]


@dataclass(frozen=True)
class GridcellView:
    index: int
    set: int
    name: str
    topounits: tuple[TopounitView, ...]

    @property
    def landunits(self) -> list[LandunitView]:
        return [lu for t in self.topounits for lu in t.landunits]

    @property
    def columns(self) -> list[ColumnView]:
        return [c for lu in self.landunits for c in lu.columns]

    @property
    def patches(self) -> list[PatchView]:
        return [p for c in self.columns for p in c.patches]
