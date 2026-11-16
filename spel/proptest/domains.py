"""Drawn domains: random subgrid hierarchies for a spec (`[domain.draw]`).

A `DomainDraw` is a space of domains: how many gridcells and topounits, which landunit kinds with
which probabilities, how often a weight is exactly 0, the pfts, CFTs, urban densities, glacier
classes, glcmask and elevations to draw from. `draw(rng)` returns gridcells (hierarchy.Gridcell)
that ELM could build:

- every topounit has a soil landunit (weight 0 when the draw did not pick soil: ELM's virtual
  one), as subgridMod always creates it;
- lake, wetland and plain glacier landunits are never of weight 0 (ELM creates them only where
  their weight is > 0); soil, crop, urban and polygon landunits may be, and glacier-mec in a
  glcmask gridcell;
- a topounit or landunit of weight 0 has siblings whose weights are > 0.

The runner (`runner.run`) runs the spec on `count` drawn domains, one Session (worker) each:
the inputs are Hypothesis' as before, the domains are drawn here from the run's seed. Domains
stay outside Hypothesis because each needs its own Fortran allocation (a new worker process).
On a failure, `shrink_candidates` proposes smaller domains (fewer gridcells, topounits,
landunits, patches; equal weights where they were > 0; the zeros kept); the runner keeps a
candidate while the check still fails and saves the smallest as `domain.toml` with the failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterator, Mapping, Optional, Sequence

import numpy as np

from spel.proptest.hierarchy import (
    Crop,
    Glacier,
    GlacierMec,
    Gridcell,
    HierarchyError,
    Lake,
    Landunit,
    LandunitKind,
    MecClass,
    Patch,
    Polygon,
    PolygonType,
    Soil,
    Topounit,
    Urban,
    UrbanDensity,
    Wetland,
)

KINDS = {k.value: k for k in LandunitKind}
# kinds ELM creates at weight 0 (subgridMod): glacier-mec only in glcmask gridcells
ZERO_OK = {
    LandunitKind.SOIL,
    LandunitKind.CROP,
    LandunitKind.URBAN,
    LandunitKind.POLYGON,
}


class DrawError(HierarchyError):
    pass


def _pair(value: Any, key: str, cast=int) -> tuple:
    if not (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and cast(value[0]) <= cast(value[1])
    ):
        raise DrawError(f"[domain.draw] {key}: a range [lo, hi]")
    return cast(value[0]), cast(value[1])


def _prob(value: Any, key: str) -> float:
    v = float(value)
    if not 0.0 <= v <= 1.0:
        raise DrawError(f"[domain.draw] {key}: a probability in [0, 1], not {value!r}")
    return v


@dataclass(frozen=True)
class DomainDraw:
    count: int = 5  # domains per run
    gridcells: tuple[int, int] = (1, 3)
    topounits: tuple[int, int] = (1, 3)
    # the probability that a topounit holds a landunit of the kind (soil: else it is virtual)
    # (TOML and the constructor also take the kinds' names: {"soil": 1.0})
    kinds: Mapping[LandunitKind, float] = field(
        default_factory=lambda: {LandunitKind.SOIL: 1.0}
    )
    zero_topounit: float = 0.0  # P(a topounit has weight 0: inactive)
    zero_landunit: float = (
        0.0  # P(a soil, crop, urban, polygon or glcmask glacier-mec landunit has weight 0)
    )
    zero_patch: float = 0.0  # P(a drawn pft, CFT or glacier class has weight 0)
    pfts: tuple[int, ...] = (1,)
    max_pfts: int = 3  # pfts per soil column
    cfts: tuple[int, ...] = ()
    max_cfts: int = 2
    densities: tuple[UrbanDensity, ...] = tuple(UrbanDensity)  # (or their names)
    mec_classes: int = 3  # classes 1..n to draw from
    glcmask: float = 0.0  # P(a gridcell is in the glcmask)
    icemask: float = 0.0  # P(a gridcell has icemask > 0)
    elevation: tuple[float, float] = (0.0, 1000.0)
    ties: float = (
        0.0  # P(a topounit takes the elevation of an earlier one in its gridcell)
    )
    copies: int = (
        1  # each drawn gridcell this many times (relations that permute need >= 2)
    )

    KEYS = (
        "count",
        "gridcells",
        "topounits",
        "kinds",
        "zero_topounit",
        "zero_landunit",
        "zero_patch",
        "pfts",
        "max_pfts",
        "cfts",
        "max_cfts",
        "densities",
        "mec_classes",
        "glcmask",
        "icemask",
        "elevation",
        "ties",
        "copies",
    )

    def __post_init__(self):
        bad = sorted(str(k) for k in self.kinds if _kind(k) is None)
        if bad:
            raise DrawError(
                f"[domain.draw] kinds: {bad} are not landunit kinds ({sorted(KINDS)})"
            )
        # names -> kinds (the order is kept: it is the order of the draws)
        object.__setattr__(
            self, "kinds", {_kind(k): float(p) for k, p in self.kinds.items()}
        )
        for k, p in self.kinds.items():
            _prob(p, f"kinds.{k.value}")
        if not any(p > 0 for p in self.kinds.values()):
            raise DrawError(
                "[domain.draw] kinds: give at least one kind a probability > 0"
            )
        if self.kinds.get(LandunitKind.CROP, 0) > 0 and not self.cfts:
            raise DrawError("[domain.draw] crop landunits need `cfts`")
        if not self.pfts:
            raise DrawError(
                "[domain.draw] pfts: at least one (soil landunits always exist)"
            )
        try:
            dens = tuple(
                UrbanDensity(d.lower() if isinstance(d, str) else d)
                for d in self.densities
            )
        except ValueError:
            dens = ()
        if not dens:
            raise DrawError(
                f"[domain.draw] densities: from {[d.value for d in UrbanDensity]}"
            )
        object.__setattr__(self, "densities", dens)
        if (
            self.count < 1
            or self.copies < 1
            or self.gridcells[0] < 1
            or self.topounits[0] < 1
        ):
            raise DrawError(
                "[domain.draw] count, copies, gridcells and topounits are >= 1"
            )
        if self.max_pfts < 1 or self.max_cfts < 1 or self.mec_classes < 1:
            raise DrawError("[domain.draw] max_pfts, max_cfts and mec_classes are >= 1")

    # -- TOML
    def to_table(self) -> dict:
        out: dict[str, Any] = {}
        default = DomainDraw()
        for key in self.KEYS:
            v = getattr(self, key)
            if v != getattr(default, key):
                if key == "kinds":
                    out[key] = {k.value: p for k, p in v.items()}
                elif key == "densities":
                    out[key] = [d.value for d in v]
                else:
                    out[key] = list(v) if isinstance(v, tuple) else v
        return out

    @classmethod
    def from_table(cls, value: Any) -> "DomainDraw":
        if not isinstance(value, dict):
            raise DrawError("[domain.draw]: a table")
        extra = sorted(set(value) - set(cls.KEYS))
        if extra:
            raise DrawError(
                f"[domain.draw]: unknown keys {extra} (known: {list(cls.KEYS)})"
            )
        kw: dict[str, Any] = {}
        for key, v in value.items():
            if key in ("gridcells", "topounits"):
                kw[key] = _pair(v, key)
            elif key == "elevation":
                kw[key] = _pair(v, key, float)
            elif key == "kinds":
                if not isinstance(v, dict):
                    raise DrawError(
                        "[domain.draw] kinds: a table { soil = 1.0, wetland = 0.3, ... }"
                    )
                kw[key] = {str(k): float(p) for k, p in v.items()}
            elif key in ("pfts", "cfts"):
                kw[key] = tuple(int(x) for x in v)
            elif key == "densities":
                kw[key] = tuple(str(x).lower() for x in v)
            elif key in (
                "zero_topounit",
                "zero_landunit",
                "zero_patch",
                "glcmask",
                "icemask",
                "ties",
            ):
                kw[key] = _prob(v, key)
            else:
                kw[key] = int(v)
        return cls(**kw)

    # -- drawing
    def domains(self, seed: int = 0) -> Iterator[list[Gridcell]]:
        """The run's `count` domains for a seed (domain k is the same for every count >= k)."""
        for k in range(self.count):
            yield self.draw(np.random.default_rng([int(seed), k]), prefix=f"d{k}.")

    def draw(self, rng: np.random.Generator, prefix: str = "") -> list[Gridcell]:
        n = int(rng.integers(self.gridcells[0], self.gridcells[1] + 1))
        return [self._gridcell(rng, f"{prefix}g{j + 1}") for j in range(n)]

    def _gridcell(self, rng: np.random.Generator, name: str) -> Gridcell:
        glcmask = bool(rng.random() < self.glcmask)
        icemask = (
            float(np.round(rng.uniform(0.1, 1.0), 3))
            if rng.random() < self.icemask
            else 0.0
        )
        nt = int(rng.integers(self.topounits[0], self.topounits[1] + 1))
        elev: list[float] = []
        for k in range(nt):
            if k and rng.random() < self.ties:
                elev.append(elev[int(rng.integers(0, k))])
            else:
                elev.append(float(np.round(rng.uniform(*self.elevation), 1)))
        zeros = [bool(rng.random() < self.zero_topounit) for _ in range(nt)]
        if all(zeros):
            zeros[int(rng.integers(nt))] = False
        weights = _weights(rng, zeros)
        tops = tuple(
            Topounit(self._landunits(rng, glcmask), w, e) for w, e in zip(weights, elev)
        )
        return Gridcell(tops, self.copies, name, glcmask, icemask)

    def _landunits(
        self, rng: np.random.Generator, glcmask: bool
    ) -> tuple[Landunit, ...]:
        drawn: list[_Pick] = []
        for kind, p in self.kinds.items():
            if rng.random() >= p:
                continue
            if kind is LandunitKind.URBAN:
                dens = [d for d in self.densities if rng.random() < 0.5] or [
                    self.densities[int(rng.integers(len(self.densities)))]
                ]
                drawn += [_Pick(kind, density=d) for d in dens]
            elif kind is LandunitKind.POLYGON:
                types = [t for t in PolygonType if rng.random() < 0.5] or [
                    PolygonType.LOW
                ]
                drawn += [_Pick(kind, polygon=t) for t in types]
            else:
                drawn.append(_Pick(kind))
        if not drawn:  # the most likely kind
            best = max(self.kinds, key=lambda k: self.kinds[k])
            drawn.append(
                _Pick(
                    best,
                    density=(
                        self.densities[0] if best is LandunitKind.URBAN else None
                    ),
                    polygon=(
                        PolygonType.LOW if best is LandunitKind.POLYGON else None
                    ),
                )
            )
        zero_ok = [
            d.kind in ZERO_OK or (d.kind is LandunitKind.GLACIER_MEC and glcmask)
            for d in drawn
        ]
        zeros = [ok and bool(rng.random() < self.zero_landunit) for ok in zero_ok]
        if all(zeros):
            zeros[int(rng.integers(len(zeros)))] = False
        weights = _weights(rng, zeros)
        lus = [self._landunit(rng, d, w, glcmask) for d, w in zip(drawn, weights)]
        if not any(isinstance(lu, Soil) for lu in lus):  # ELM's virtual soil landunit
            lus.insert(0, Soil(*self._pfts(rng), weight=0.0))
        return tuple(lus)

    def _pfts(self, rng: np.random.Generator) -> list[Patch]:
        k = int(rng.integers(1, min(self.max_pfts, len(self.pfts)) + 1))
        pick = sorted(rng.choice(self.pfts, size=k, replace=False).tolist())
        return [
            Patch(int(p), w) for p, w in zip(pick, _weights(rng, self._zeros(rng, k)))
        ]

    def _zeros(self, rng: np.random.Generator, n: int) -> list[bool]:
        z = [bool(rng.random() < self.zero_patch) for _ in range(n)]
        if all(z):
            z[int(rng.integers(n))] = False
        return z

    def _landunit(
        self, rng, pick: "_Pick", weight: Optional[float], glcmask: bool
    ) -> Landunit:
        kind = pick.kind
        if kind is LandunitKind.SOIL:
            return Soil(*self._pfts(rng), weight=weight)
        if kind is LandunitKind.POLYGON:
            assert pick.polygon is not None
            return Polygon(pick.polygon, *self._pfts(rng), weight=weight)
        if kind is LandunitKind.CROP:
            k = int(rng.integers(1, min(self.max_cfts, len(self.cfts)) + 1))
            pick_cfts = sorted(rng.choice(self.cfts, size=k, replace=False).tolist())
            return Crop(
                *(
                    Patch(int(c), w)
                    for c, w in zip(pick_cfts, _weights(rng, self._zeros(rng, k)))
                ),
                weight=weight,
            )
        if kind is LandunitKind.URBAN:
            assert pick.density is not None
            return Urban(
                pick.density,
                weight=weight,
                roof=float(np.round(rng.uniform(0.1, 0.9), 3)),
                road_perv=float(np.round(rng.uniform(0.0, 1.0), 3)),
            )
        if kind is LandunitKind.GLACIER_MEC:
            k = int(rng.integers(1, self.mec_classes + 1))
            classes = sorted(
                rng.choice(
                    np.arange(1, self.mec_classes + 1), size=k, replace=False
                ).tolist()
            )
            ws = _weights(rng, self._zeros(rng, k) if glcmask else [False] * k)
            return GlacierMec(
                [MecClass(int(m), w, 500.0 * m) for m, w in zip(classes, ws)],
                weight=weight,
            )
        return {
            LandunitKind.LAKE: Lake,
            LandunitKind.WETLAND: Wetland,
            LandunitKind.GLACIER: Glacier,
        }[kind](weight=weight)


@dataclass(frozen=True)
class _Pick:
    """A landunit the draw picked, before its weight and contents are drawn."""

    kind: LandunitKind
    density: Optional[UrbanDensity] = None  # urban
    polygon: Optional[PolygonType] = None  # polygons


def _kind(k: LandunitKind | str) -> Optional[LandunitKind]:
    return k if isinstance(k, LandunitKind) else KINDS.get(str(k).lower())


def _weights(rng: np.random.Generator, zeros: Sequence[bool]) -> list[Optional[float]]:
    """Weights that sum to 1: 0.0 where `zeros` says, the others from a flat Dirichlet, rounded;
    the last positive one is None (it takes what the others leave, exactly)."""
    pos = [k for k, z in enumerate(zeros) if not z]
    out: list[Optional[float]] = [0.0] * len(zeros)
    if not pos:
        raise DrawError("no positive weight")
    w = rng.dirichlet(np.ones(len(pos)))
    w = np.maximum(np.round(w, 4), 0.001)
    for k, x in zip(pos[:-1], w[:-1]):
        out[k] = float(x)
    if sum(x for x in out if x) >= 1.0 - 1e-6:  # rounding left nothing for the last one
        for k in pos[:-1]:
            out[k] = None
    out[pos[-1]] = None
    return out


# -- shrinking


def _simple(weights: Sequence[Optional[float]]) -> tuple[Optional[float], ...]:
    """Zeros kept, every other weight shared equally (None); all shared when only zeros are left."""
    out = tuple(0.0 if w == 0.0 else None for w in weights)
    return out if any(w is None for w in out) else tuple(None for _ in out)


def _with_soil(lus: Sequence[Landunit], pft: int) -> tuple[Landunit, ...]:
    """The landunits, with a virtual soil landunit of that pft when they have no soil one."""
    if any(isinstance(lu, Soil) for lu in lus):
        return tuple(lus)
    return (Soil(pft, weight=0.0), *lus)


def _soil_pft(t: Topounit) -> int:
    """The first pft of the topounit's soil landunit (shrinking keeps the domain's pfts)."""
    return next((lu.columns[0].patches[0].pft for lu in t.landunits if isinstance(lu, Soil)), 1)


def _valid(gridcells: list[Gridcell]) -> bool:
    try:
        for g in gridcells:
            g.checked()
            for t in g.topounits:
                if all(lu.weight == 0.0 for lu in t.landunits):
                    return False
            if all(t.weight == 0.0 for t in g.topounits):
                return False
        return True
    except HierarchyError:
        return False


def shrink_candidates(gridcells: list[Gridcell]) -> Iterator[list[Gridcell]]:
    """Smaller or simpler domains than `gridcells`, the biggest reductions first. Each one is
    valid (weights sum to 1, ELM's rules hold)."""

    def emit(cand):
        # (dropping a virtual soil landunit puts it back: no change)
        if cand != gridcells and _valid(cand):
            yield cand

    def with_gridcell(j: int, g: Gridcell) -> list[Gridcell]:
        return gridcells[:j] + [g] + gridcells[j + 1 :]

    n = len(gridcells)
    for j in range(n):  # one gridcell fewer
        if n > 1:
            yield from emit(gridcells[:j] + gridcells[j + 1 :])
    for j, g in enumerate(gridcells):  # fewer copies
        if g.copies > 1:
            yield from emit(with_gridcell(j, replace(g, copies=1)))
    for j, g in enumerate(gridcells):  # one topounit fewer
        if len(g.topounits) > 1:
            for k in range(len(g.topounits)):
                rest = g.without_topounit(k)
                ws = _simple([t.weight for t in rest.topounits])
                tops = tuple(replace(t, weight=w) for t, w in zip(rest.topounits, ws))
                yield from emit(with_gridcell(j, replace(rest, topounits=tops)))
    for j, g in enumerate(gridcells):  # one landunit fewer
        for k, t in enumerate(g.topounits):
            if len(t.landunits) < 2:
                continue
            for m in range(len(t.landunits)):
                rest = t.without_landunit(m).landunits
                ws = _simple([x.weight for x in rest])
                lus = _with_soil(
                    [x.with_weight(w) for x, w in zip(rest, ws)], _soil_pft(t)
                )
                yield from emit(
                    with_gridcell(j, g.with_topounit(k, replace(t, landunits=lus)))
                )
    for j, g in enumerate(gridcells):  # one patch (pft) fewer in a soil or polygon column
        for k, t in enumerate(g.topounits):
            for m, lu in enumerate(t.landunits):
                if not isinstance(lu, (Soil, Polygon)):
                    continue
                for c, col in enumerate(lu.columns):
                    if len(col.patches) < 2:
                        continue
                    for q in range(len(col.patches)):
                        rest = col.patches[:q] + col.patches[q + 1 :]
                        pats = [
                            replace(p, weight=w)
                            for p, w in zip(rest, _simple([p.weight for p in rest]))
                        ]
                        nt = t.with_landunit(m, lu.with_patches(pats, column=c))
                        yield from emit(with_gridcell(j, g.with_topounit(k, nt)))
    # equal weights where they were > 0
    simpler = []
    for g in gridcells:
        tops = []
        for t, tw in zip(g.topounits, _simple([t.weight for t in g.topounits])):
            lus = tuple(
                lu.with_weight(w)
                for lu, w in zip(t.landunits, _simple([x.weight for x in t.landunits]))
            )
            tops.append(replace(t, weight=tw, landunits=lus))
        simpler.append(replace(g, topounits=tuple(tops)))
    if simpler != gridcells:
        yield from emit(simpler)


def size(gridcells: Sequence[Gridcell]) -> int:
    """A domain's size for reports: its elements over all levels (one copy each)."""
    return sum(g.size for g in gridcells)
