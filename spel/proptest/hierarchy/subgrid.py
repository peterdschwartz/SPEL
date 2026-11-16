"""ELM's implicit subgrid elements (subgridMod): what every topounit holds besides the declaration."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, ClassVar, Optional, Sequence

from spel.proptest.hierarchy._table import check_keys, resolve_weights
from spel.proptest.hierarchy.codes import HierarchyError
from spel.proptest.hierarchy.decl import Crop, GlacierMec, Gridcell, Patch, Soil, Topounit


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

    KEYS: ClassVar[tuple[str, ...]] = ("natpft", "cft", "maxpatch_glcmec")

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
        check_keys(value, set(cls.KEYS), where)

        def pair(key: str) -> Optional[tuple[int, int]]:
            v = value.get(key)
            if v is None:
                return None
            if not (isinstance(v, (list, tuple)) and len(v) == 2 and int(v[0]) <= int(v[1])):
                raise HierarchyError(f"{where}: {key} is [lb, ub]")
            return int(v[0]), int(v[1])

        natpft = pair("natpft")
        assert natpft is not None
        return cls(natpft, pair("cft"), int(value.get("maxpatch_glcmec", 0)))

    # -- the implicit elements
    def expand(self, g: Gridcell, where: str) -> list[Topounit]:
        """The gridcell's topounits with every landunit, column and patch ELM creates."""
        return [self._topounit(g, t, f"{where}.topounit[{k}]") for k, t in enumerate(g.topounits)]

    def _topounit(self, g: Gridcell, t: Topounit, where: str) -> Topounit:
        lus = list(t.landunits)
        for j, lu in enumerate(lus):
            lu.require(self, g.glcmask, f"{where}.landunit[{j}]")
        kinds = {type(lu) for lu in lus}
        if Soil not in kinds:
            lus.insert(0, Soil(self.natpft[0], weight=0.0))
        if self.cft is not None and Crop not in kinds:
            lus.append(Crop(self.cft[0], weight=0.0))
        if g.glcmask and self.maxpatch_glcmec and GlacierMec not in kinds:
            lus.append(GlacierMec({1: 1.0}, weight=0.0))
        out = tuple(lu.complete(self, g.glcmask, f"{where}.landunit[{j}]") for j, lu in enumerate(lus))
        return replace(t, landunits=out)

    def natural_patches(self, patches: Sequence[Patch], where: str) -> tuple[Patch, ...]:
        """A natural-vegetation column's patches: one per pft natpft_lb..natpft_ub, the
        undeclared ones of weight 0."""
        lb, ub = self.natpft
        have = [p.pft for p in patches]
        bad = sorted(set(have) - set(range(lb, ub + 1)))
        if bad:
            raise HierarchyError(f"{where}: pfts {bad} are outside natpft_lb:natpft_ub = {self.natpft}")
        w = dict(zip(have, resolve_weights([p.weight for p in patches], f"{where}.patches")))
        return tuple(Patch(m, w.get(m, 0.0)) for m in range(lb, ub + 1))
