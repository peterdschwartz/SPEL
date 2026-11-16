"""The declared gridcells laid out as ELM lays out a clump (see the package docstring)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Optional, Sequence, TypeVar

import numpy as np
import pandas as pd

from spel.proptest.hierarchy._table import resolve_weights
from spel.proptest.hierarchy.codes import (
    ISPVAL,
    LEVELS,
    MAX_LUNIT,
    TYPE_CODES,
    Flag,
    HierarchyError,
    LandunitKind,
    Level,
    column_role,
)
from spel.proptest.hierarchy.decl import (
    Column,
    Gridcell,
    Landunit,
    Patch,
    Polygon,
    Topounit,
    Urban,
)
from spel.proptest.hierarchy.subgrid import Subgrid
from spel.proptest.hierarchy.views import (
    ColumnView,
    GridcellView,
    LandunitView,
    PatchView,
    TopounitView,
)

ORDERS = ("elm", "depth")
# top_pp%downhill_ti: "elm" = set_topounit as in E3SM (the dn_index quirk included), "intended" =
# the nearest lower topounit of the gridcell, -1 when none is lower (see Layout.downhill)
DOWNHILL_RULES = ("elm", "intended")

D = TypeVar("D")


@dataclass(frozen=True)
class LevelTable(Generic[D]):
    """The elements of one level, in layout order (0-based rows): the row of each element's parent
    in the next coarser level (-1 for gridcells), its weight in the parent, and the declaration
    it comes from."""

    level: Level
    parent: np.ndarray
    weight: np.ndarray
    decl: tuple[D, ...]

    def __len__(self) -> int:
        return len(self.decl)


class _Rows(Generic[D]):
    """A LevelTable while it is built."""

    def __init__(self) -> None:
        self.parent: list[int] = []
        self.weight: list[float] = []
        self.decl: list[D] = []

    def add(self, parent: int, weight: float, decl: D) -> int:
        self.parent.append(parent)
        self.weight.append(weight)
        self.decl.append(decl)
        return len(self.decl) - 1

    def table(self, level: Level) -> LevelTable[D]:
        return LevelTable(
            level,
            np.asarray(self.parent, dtype=np.int64),
            np.asarray(self.weight, dtype=float),
            tuple(self.decl),
        )


class Layout:
    """The declared gridcells laid out as ELM lays out a clump (see the package docstring).
    Arrays are per element of a level; indices given to and returned by the methods are Fortran
    (1-based) unless named `*_of` or documented as rows (0-based positions). Levels are `Level`s
    or their letters ("g", "t", "l", "c", "p")."""

    def __init__(
        self,
        gridcells: Sequence[Gridcell],
        nsets: int = 1,
        subgrid: Optional[Subgrid] = None,
        order: str = "elm",
        downhill: str = "elm",
    ):
        if not gridcells:
            raise HierarchyError("the domain has no gridcell")
        if nsets < 1:
            raise HierarchyError("nsets must be >= 1")
        if order not in ORDERS:
            raise HierarchyError(f"order must be one of {ORDERS}, not {order!r}")
        if downhill not in DOWNHILL_RULES:
            raise HierarchyError(f"downhill must be one of {DOWNHILL_RULES}, not {downhill!r}")
        self.declared: list[Gridcell] = [
            g.checked(f"gridcell[{k}]") for k, g in enumerate(gridcells)
        ]
        self.nsets = nsets
        self.subgrid = subgrid
        self.order = order
        self.downhill_rule = downhill  # the rule of downhill() and of the domain's top_pp%downhill_ti
        # the topounits as laid out: with ELM's implicit elements under a subgrid
        expanded = [
            subgrid.expand(g, f"gridcell[{k}]") if subgrid else list(g.topounits)
            for k, g in enumerate(self.declared)
        ]
        # one set: every declared gridcell `copies` times, in order
        self.set_gridcells: list[Gridcell] = [
            g for g in self.declared for _ in range(g.copies)
        ]
        set_tops = [
            tops for g, tops in zip(self.declared, expanded) for _ in range(g.copies)
        ]
        rows: dict[Level, _Rows[Any]] = {lev: _Rows() for lev in LEVELS}
        unit_of_gridcell: list[int] = []
        topo_number: list[int] = []
        for _ in range(nsets):
            self._add_set(rows, set_tops, unit_of_gridcell, topo_number)
        self.tables: dict[Level, LevelTable[Any]] = {
            lev: rows[lev].table(lev) for lev in LEVELS
        }
        self.unit_of_gridcell: list[int] = (
            unit_of_gridcell  # position of each gridcell within its set
        )
        self.topo_number = np.asarray(
            topo_number, dtype=np.int64
        )  # topounit number within its gridcell (1..n)
        self.elevation: tuple[Optional[float], ...] = tuple(
            t.elevation for t in self.topounits
        )
        self._ltype = np.asarray([lu.ltype for lu in self.landunits], dtype=np.int64)
        self._glcmask_g = np.asarray([g.glcmask for g in self.gridcells], dtype=bool)
        self._all_glacier_t = np.asarray(
            [t.all_glacier for t in self.topounits], dtype=bool
        )

    def _add_set(
        self,
        rows: dict[Level, _Rows[Any]],
        set_tops: list[list[Topounit]],
        unit_of_gridcell: list[int],
        topo_number: list[int],
    ) -> None:
        entries = []  # (sort key, topounit row, landunit, its weight)
        for unit, (g, tops) in enumerate(zip(self.set_gridcells, set_tops)):
            gi = rows[Level.G].add(-1, 1.0, g)
            unit_of_gridcell.append(unit)
            for k, (t, tw) in enumerate(
                zip(tops, resolve_weights([t.weight for t in tops], "topounits"))
            ):
                ti = rows[Level.T].add(gi, tw, t)
                topo_number.append(k + 1)
                lws = resolve_weights([lu.weight for lu in t.landunits], "landunits")
                for j, (lu, lw) in enumerate(zip(t.landunits, lws)):
                    if self.order == "depth":
                        key: tuple = (ti, j)
                    else:  # initGridCellsMod: one pass per landunit type over the clump's topounits
                        kind_pass, within = lu.elm_order
                        key = (kind_pass, ti, within)
                    entries.append((key, ti, lu, lw))
        for _, ti, lu, lw in sorted(entries, key=lambda e: e[0]):
            li = rows[Level.L].add(ti, lw, lu)
            cols = lu.columns
            for c, cw in zip(
                cols, resolve_weights([c.weight for c in cols], "columns")
            ):
                ci = rows[Level.C].add(li, cw, c)
                for p, pw in zip(
                    c.patches, resolve_weights([p.weight for p in c.patches], "patches")
                ):
                    rows[Level.P].add(ci, pw, p)

    def table_of(self, level: Level | str) -> LevelTable[Any]:
        return self.tables[Level(level)]

    @property
    def gridcells(self) -> tuple[Gridcell, ...]:
        """The declaration of every laid-out gridcell (all sets, copies included)."""
        return self.tables[Level.G].decl

    @property
    def topounits(self) -> tuple[Topounit, ...]:
        return self.tables[Level.T].decl

    @property
    def landunits(self) -> tuple[Landunit, ...]:
        return self.tables[Level.L].decl

    @property
    def columns(self) -> tuple[Column, ...]:
        return self.tables[Level.C].decl

    @property
    def patches(self) -> tuple[Patch, ...]:
        return self.tables[Level.P].decl

    # -- sizes and positions
    @property
    def per_set_gridcells(self) -> int:
        return len(self.set_gridcells)

    def count(self, level: Level | str) -> int:
        return len(self.tables[Level(level)])

    def sizes(self) -> dict[str, int]:
        return {lev.value: self.count(lev) for lev in LEVELS}

    def per_set(self, level: Level | str) -> int:
        return self.count(level) // self.nsets

    @property
    def max_topounits(self) -> int:
        return max(len(g.topounits) for g in self.declared)

    def ancestor_of(self, level: Level | str, coarser: Level | str) -> np.ndarray:
        """0-based index of each element's ancestor at a coarser (or the same) level."""
        lev, top = Level(level), Level(coarser)
        idx = np.arange(self.count(lev))
        while lev.depth > top.depth:
            idx = self.tables[lev].parent[idx]
            lev = lev.coarser
        return idx

    def parent(self, level: Level | str, coarser: Level | str) -> np.ndarray:
        """Fortran index of each element's ancestor at a coarser (or the same) level."""
        return self.ancestor_of(level, coarser) + 1

    def first_child(self, level: Level | str, finer: Level | str) -> np.ndarray:
        """Fortran index of each element's first descendant at a finer (or the same) level
        (ispval-free: 0 where it has none)."""
        return self._extreme_child(level, finer, first=True)

    def last_child(self, level: Level | str, finer: Level | str) -> np.ndarray:
        return self._extreme_child(level, finer, first=False)

    def _extreme_child(
        self, level: Level | str, finer: Level | str, first: bool
    ) -> np.ndarray:
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

    def contiguous(self, level: Level | str, finer: Level | str) -> bool:
        """Whether every element's descendants at `finer` are one contiguous range (always for
        l -> c -> p; in the ELM order not for a topounit's or gridcell's landunits)."""
        return bool(
            np.all(
                self.last_child(level, finer) - self.first_child(level, finer) + 1
                == self.children(level, finer)
            )
        )

    def children(self, level: Level | str, finer: Level | str) -> np.ndarray:
        """The number of each element's descendants at a finer level."""
        return np.bincount(
            self.ancestor_of(finer, level), minlength=self.count(level)
        ).astype(np.int64)

    def one_to_one(self, a: Level | str, b: Level | str) -> bool:
        """Whether two levels have the same elements (a tree: equal counts)."""
        return self.count(a) == self.count(b)

    def descendants(self, level: Level | str, k: int, finer: Level | str) -> np.ndarray:
        """0-based rows of `finer` under element k (Fortran) of `level`, in layout order."""
        return np.flatnonzero(self.ancestor_of(finer, level) == k - 1)

    def gridcell_of(self, level: Level | str) -> np.ndarray:
        """0-based gridcell of each element (the unit relations permute)."""
        return self.ancestor_of(level, Level.G)

    def set_of(self, level: Level | str) -> np.ndarray:
        return self.gridcell_of(level) // self.per_set_gridcells

    def name_of(self, level: Level | str) -> list[str]:
        """The label of each element's declaring gridcell (its name, or its landunits)."""
        labels = [gc.label for gc in self.gridcells]
        return [labels[k] for k in self.gridcell_of(level)]

    def row_permutation(
        self, level: Level | str, gridcell_perm: Sequence[int]
    ) -> np.ndarray:
        """A permutation of the first `len(gridcell_perm)` gridcells as rows of a level: the
        rows of gridcell k receive what the rows of gridcell `perm[k]` held (as an index array:
        `new = old[result]`). Gridcells are only exchanged with gridcells of the same
        declaration, whose elements come in the same relative order."""
        perm = np.asarray(gridcell_perm, dtype=np.int64)
        owner = self.gridcell_of(level)
        n = int(np.count_nonzero(owner < len(perm)))
        out = np.arange(n, dtype=np.int64)
        for g, src in enumerate(perm):
            dst_rows, src_rows = np.flatnonzero(owner == g), np.flatnonzero(
                owner == src
            )
            if len(dst_rows) != len(src_rows):
                raise HierarchyError(
                    f"gridcells {g + 1} and {src + 1} differ in their {Level(level).label}s"
                )
            out[dst_rows] = src_rows
        return out

    # -- element attributes
    def weight(self, level: Level | str, relative_to: Level | str) -> np.ndarray:
        """Each element's weight in its ancestor at `relative_to` (wtcol, wtlunit, wttopounit, wtgcell)."""
        lev, top = Level(level), Level(relative_to)
        w = np.ones(self.count(lev))
        idx = np.arange(self.count(lev))
        while lev.depth > top.depth:
            w = w * self.tables[lev].weight[idx]
            idx = self.tables[lev].parent[idx]
            lev = lev.coarser
        return w

    def active(self, level: Level | str) -> np.ndarray:
        """col_pp/lun_pp/top_pp/veg_pp%active as subgridWeightsMod's is_active_* set them (see the
        package docstring; `all_active` off)."""
        lev = Level(level)
        if lev is Level.G:
            return np.ones(self.count(Level.G), dtype=bool)
        own = self.tables[lev].weight > 0.0
        if lev is Level.T:
            return own
        top = self.active(Level.T)[self.ancestor_of(lev, Level.T)]
        if lev is Level.P:
            return self.active(Level.C)[self.tables[Level.P].parent] & own
        lt = self.ltype(lev)
        mec = (lt == TYPE_CODES["istice_mec"]) & self._glcmask_g[self.gridcell_of(lev)]
        if lev is Level.L:
            soil = (lt == TYPE_CODES["istsoil"]) & ~self._all_glacier_t[
                self.ancestor_of(Level.L, Level.T)
            ]
            return top & (own | mec | soil)
        lun = self.active(Level.L)[self.tables[Level.C].parent]
        urban = self.is_kind(Level.C, LandunitKind.URBAN)
        return top & ((lun & own) | mec | (lun & urban))

    def flag(self, level: Level | str, name: Flag | str) -> np.ndarray:
        """A logical subgrid flag (`Flag`, or its Fortran name) of each element of its level or a
        finer one (the ancestor's value)."""
        try:
            flag = Flag(name)
        except ValueError:
            raise HierarchyError(f"no subgrid flag {name!r}") from None
        lev = Level(level)
        if lev.depth < flag.level.depth:
            raise HierarchyError(
                f"{flag.value} is a {flag.level.label} flag, not of {lev.label}s"
            )
        return self.is_kind(lev, *flag.kinds)

    def polygon_type(self) -> np.ndarray:
        """lun_pp%polygontype: 1/2/3 on polygon landunits, ispval elsewhere."""
        return np.asarray(
            [
                lu.polygon.code if isinstance(lu, Polygon) else ISPVAL
                for lu in self.landunits
            ],
            dtype=np.int64,
        )

    def glc_topo(self) -> np.ndarray:
        """col_pp%glc_topo at initialization: the class's topo on glacier-mec columns (NaN where
        the declaration has none), 0 elsewhere."""
        mec = self.is_kind(Level.C, LandunitKind.GLACIER_MEC)
        topo = [c.topo if c.topo is not None else np.nan for c in self.columns]
        return np.where(mec, np.asarray(topo, dtype=float), 0.0)

    def mxy(self) -> np.ndarray:
        """veg_pp%mxy: pft + 1 - natpft_lb on soil and crop landunits, ispval elsewhere."""
        if self.subgrid is None:
            raise HierarchyError(
                "veg_pp%mxy needs natpft_lb: give the domain a subgrid"
            )
        veg = self.is_kind(
            Level.P, LandunitKind.SOIL, LandunitKind.POLYGON, LandunitKind.CROP
        )
        return np.where(
            veg, self.itype(Level.P) + 1 - self.subgrid.natpft[0], ISPVAL
        ).astype(np.int64)

    def landunit_of(self, level: Level | str) -> list[Optional[Landunit]]:
        """The landunit declaration of each element (None for gridcells and topounits)."""
        lev = Level(level)
        if lev.depth < Level.L.depth:
            return [None] * self.count(lev)
        lus = self.landunits
        return [lus[k] for k in self.ancestor_of(lev, Level.L)]

    def kind_of(self, level: Level | str) -> list[Optional[LandunitKind]]:
        """The landunit kind of each element (None for gridcells and topounits)."""
        return [None if lu is None else lu.kind for lu in self.landunit_of(level)]

    def is_kind(self, level: Level | str, *kinds: LandunitKind) -> np.ndarray:
        return np.asarray([k in kinds for k in self.kind_of(level)], dtype=bool)

    def ltype(self, level: Level | str) -> np.ndarray:
        return self._ltype[self.ancestor_of(level, Level.L)]

    def itype(self, level: Level | str) -> np.ndarray:
        """lun_pp/col_pp/veg_pp%itype."""
        lev = Level(level)
        if lev is Level.L:
            return self._ltype.copy()
        if lev is Level.C:
            return np.asarray([c.itype for c in self.columns], dtype=np.int64)
        if lev is Level.P:
            return np.asarray([p.pft for p in self.patches], dtype=np.int64)
        raise HierarchyError(f"no itype at the {lev.label} level")

    def pft_of(self, level: Level | str) -> Optional[np.ndarray]:
        """The pft of each patch (per-pft rules apply to patch-level variables only)."""
        return self.itype(Level.P) if Level(level) is Level.P else None

    def topo_index(self) -> np.ndarray:
        """top_pp%topo_grc_ind: each topounit's number within its gridcell."""
        return self.topo_number.copy()

    def downhill(self, rule: Optional[str] = None, local: bool = False) -> np.ndarray:
        """top_pp%downhill_ti from the topounits' elevations (use_IM2_hillslope_hydrology):
        Fortran topounit indices, or with `local` the neighbour's number within its gridcell
        (topo_grc_ind); -1: none. -1 too where a gridcell has one topounit (initSubgridMod's
        add_topounit sets -1 for every topounit; set_topounit only overwrites it for several).

        `rule` (default: the layout's `downhill_rule`, which the domain's topology uses):
        - "elm": set_topounit as E3SM has it, bit-for-bit. Its `dn_index` is set to -1 once per
          gridcell, not per topounit, so a topounit with no lower neighbour keeps the previous
          topounit's answer: the lowest can point at a higher topounit or at itself.
        - "intended": the highest of the lower topounits (the last one of equal elevation, as
          ELM's `>=` picks), -1 when none is lower.
        """
        rule = self.downhill_rule if rule is None else rule
        if rule not in DOWNHILL_RULES:
            raise HierarchyError(f"downhill rule must be one of {DOWNHILL_RULES}, not {rule!r}")
        out = np.full(self.count(Level.T), -1, dtype=np.int64)
        elevation = self.elevation
        for g in range(self.count(Level.G)):
            ts = [int(t) for t in self.descendants(Level.G, g + 1, Level.T)]
            if len(ts) < 2:
                continue
            elev = [elevation[t] for t in ts]
            if any(e is None for e in elev):
                raise HierarchyError(
                    f"gridcell {g + 1}: declare every topounit's elevation (downhill_ti)"
                )
            min_elev = min(elev)
            dn_index = -1  # "elm": once per gridcell (the quirk)
            for t1 in ts:
                if rule == "intended":
                    dn_index = -1
                dn_elev = min_elev
                for t2 in ts:
                    if elevation[t2] < elevation[t1] and elevation[t2] >= dn_elev:
                        dn_elev = elevation[t2]
                        dn_index = t2 + 1
                out[t1] = dn_index
        if local:
            out = np.where(out > 0, self.topo_number[np.maximum(out - 1, 0)], -1)
        return out

    def downhill_problems(self, rule: Optional[str] = None) -> list[str]:
        """Per topounit, what is wrong with its downhill neighbour under `rule` (default: the
        layout's): "-> itself", "-> k (level)" or "-> k (uphill)" with k its number in the
        gridcell; "" when it is lower or there is none."""
        dn = self.downhill(rule)
        out = []
        for t, d in enumerate(dn):
            if d <= 0:
                out.append("")
            elif d - 1 == t:
                out.append("-> itself")
            elif self.elevation[d - 1] >= self.elevation[t]:
                kind = "level" if self.elevation[d - 1] == self.elevation[t] else "uphill"
                out.append(f"-> {self.topo_number[d - 1]} ({kind})")
            else:
                out.append("")
        return out

    def landunit_indices(
        self, level: Level | str, max_lunit: int = MAX_LUNIT
    ) -> np.ndarray:
        """grc_pp/top_pp%landunit_indices(ltype, element): the first landunit of each type under
        the element (elm_ptrs_compdown), ispval where there is none. Shape (max_lunit, n).
        """
        n = self.count(level)
        out = np.full((max_lunit, n), ISPVAL, dtype=np.int64)
        owner = self.ancestor_of(Level.L, level)
        lus = self.landunits
        for li in range(
            len(lus) - 1, -1, -1
        ):  # backwards: the first landunit of a type wins
            t = int(lus[li].index_type)
            if 1 <= t <= max_lunit:
                out[t - 1, owner[li]] = li + 1
        return out

    # -- reports
    def describe(self, level: Level | str, k: int) -> str:
        """Element k (Fortran index) of a level, for reports."""
        lev = Level(level)
        j = k - 1
        g = int(self.gridcell_of(lev)[j])
        parts = [f"set {g // self.per_set_gridcells}", self.gridcells[g].label]
        if lev is not Level.G:
            t = int(self.ancestor_of(lev, Level.T)[j])
            ntop = int(self.children(Level.G, Level.T)[g])
            if ntop > 1:
                parts.append(f"topounit {self.topo_number[t]}/{ntop}")
        if lev.depth >= Level.L.depth:
            parts.append(self.landunits[int(self.ancestor_of(lev, Level.L)[j])].label)
        if lev.depth >= Level.C.depth:
            parts.append(
                f"column {column_role(self.columns[int(self.ancestor_of(lev, Level.C)[j])].itype)}"
            )
        if lev is Level.P:
            parts.append(f"pft {self.patches[j].pft}")
        return ", ".join(parts)

    def table(self, level: Level | str)->pd.Dataframe:
        """A pandas DataFrame with one row per element of the level, indexed by its Fortran index:
        set, gridcell (index and name), topounit (index and number), and where they exist
        the topounit's elevation (when the domain declares any; NaN where one is not declared),
        landunit, kind, density, column, role, pft, weights (in the parent and the gridcell) and
        active."""

        lev = Level(level)
        n = self.count(lev)
        rows: dict[str, Any] = {
            "set": self.set_of(lev),
            "gridcell": self.parent(lev, Level.G),
            "name": self.name_of(lev),
        }
        if lev is not Level.G:
            t = self.ancestor_of(lev, Level.T)
            rows["topounit"] = t + 1
            rows["topo_number"] = self.topo_number[t]
            if any(e is not None for e in self.elevation):
                elev = np.asarray(
                    [np.nan if e is None else e for e in self.elevation], dtype=float
                )
                rows["elevation"] = elev[t]
        if lev.depth >= Level.L.depth:
            lu = self.ancestor_of(lev, Level.L)
            decl = self.landunits
            rows["landunit"] = lu + 1
            rows["kind"] = [decl[k].kind.value for k in lu]
            rows["density"] = [
                decl[k].density.value if isinstance(decl[k], Urban) else "" for k in lu
            ]
            rows["polygon"] = [
                decl[k].polygon.value if isinstance(decl[k], Polygon) else ""
                for k in lu
            ]
        if lev.depth >= Level.C.depth:
            c = self.ancestor_of(lev, Level.C)
            ctype = self.itype(Level.C)
            rows["column"] = c + 1
            rows["role"] = [column_role(int(ctype[k])) for k in c]
        if lev is Level.P:
            rows["pft"] = self.itype(Level.P)
        if lev is not Level.G:
            rows["weight"] = self.tables[lev].weight
            rows["wtgcell"] = self.weight(lev, Level.G)
        rows["active"] = self.active(lev)
        return pd.DataFrame(rows, index=pd.Index(np.arange(1, n + 1), name=lev.label))

    def tree(self) -> list[GridcellView]:
        """Every gridcell as a tree of views (Fortran indices, weights in the parent, activity)."""
        act = {lev: self.active(lev) for lev in LEVELS[1:]}
        wt = {lev: self.tables[lev].weight for lev in LEVELS[1:]}
        wg = self.weight(Level.P, Level.G)
        ctype, pft, ltype = (
            self.itype(Level.C),
            self.itype(Level.P),
            self.itype(Level.L),
        )
        kids = {lev: [[] for _ in range(self.count(lev))] for lev in LEVELS[:-1]}
        for lev in LEVELS[1:]:
            for row, parent in enumerate(self.tables[lev].parent):
                kids[lev.coarser][int(parent)].append(row)

        def patch(p: int) -> PatchView:
            return PatchView(
                p + 1,
                int(pft[p]),
                float(wt[Level.P][p]),
                float(wg[p]),
                bool(act[Level.P][p]),
            )

        def column(c: int) -> ColumnView:
            return ColumnView(
                c + 1,
                int(ctype[c]),
                column_role(int(ctype[c])),
                float(wt[Level.C][c]),
                bool(act[Level.C][c]),
                tuple(patch(p) for p in kids[Level.C][c]),
            )

        def landunit(li: int) -> LandunitView:
            decl = self.landunits[li]
            density = decl.density if isinstance(decl, Urban) else None
            return LandunitView(
                li + 1,
                decl.kind,
                density,
                int(ltype[li]),
                float(wt[Level.L][li]),
                bool(act[Level.L][li]),
                tuple(column(c) for c in kids[Level.L][li]),
                decl,
            )

        def topounit(t: int) -> TopounitView:
            return TopounitView(
                t + 1,
                int(self.topo_number[t]),
                float(wt[Level.T][t]),
                bool(act[Level.T][t]),
                self.elevation[t],
                tuple(landunit(li) for li in kids[Level.T][t]),
            )

        return [
            GridcellView(
                g + 1,
                g // self.per_set_gridcells,
                self.gridcells[g].label,
                tuple(topounit(t) for t in kids[Level.G][g]),
            )
            for g in range(self.count(Level.G))
        ]
