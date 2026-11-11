"""
Holder types: the derived type whose fields get compact storage, and where
its instances live.

Before the nutrient transform a flux type (``vegetation_carbon_flux``) has
module instances (``veg_cf``, ``c13_veg_cf``) and type-bound procedures
(Init, SetValues, ...). After it, the fields shared by the C/N/P types live in
a base type (``vegetation_nutrient_flux``) that has no instances of its own:
each concrete type holds one as its component ``core`` (``veg_cf%core%agnpp``),
the N/P-only fields are in role components (``veg_nf%retrans%pool``), and the
shared code is in plain module procedures ``<base>_<binding>``.

A `Holder` is either shape: the type, its `Embedding`s (an instance and the
component path that reaches the holder in it, "" for the instance's own type)
and its routines (type-bound for a concrete type, ``<type>_<binding>`` module
procedures for a base type). The compact rewrite then works on the holder:
one `cbuf` per holder, shared by every instance that embeds it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from spel.scripts.utilityFunctions import Variable

DRIVER_ID = "elm_driver::elm_drv"
BINDINGS = ("init", "setvalues", "zerodwt", "init_history", "restart", "summary", "clean")
ELEMENT_WORDS = {"carbon": "c12", "nitrogen": "n", "phosphorus": "p"}
NUTRIENT_MANIFEST = "data_types/.spel-nutrient.json"


@dataclass(frozen=True)
class Embedding:
    instance: str  # veg_cf, c13_veg_cf
    concrete: str  # the instance's type: vegetation_carbon_flux
    path: str  # component path to the holder in the instance: "" | "core" | "retrans"
    element: str  # c12 | c13 | c14 | n | p | "" (no nutrient element)

    @property
    def prefix(self) -> str:
        """What a field path of the holder starts with: `veg_cf%core` or `veg_cf`."""
        return f"{self.instance}%{self.path}" if self.path else self.instance

    def field(self, comp: str) -> str:
        return f"{self.prefix}%{comp}"


@dataclass
class Holder:
    type: str
    dtype: object
    embeddings: list[Embedding]
    routines: dict[str, str] = field(default_factory=dict)  # binding -> routine id

    @property
    def embedded(self) -> bool:
        """The holder is a component of its instances' types (a nutrient base or role type)."""
        return any(e.path for e in self.embeddings)

    @property
    def instances(self) -> list[str]:
        return [e.instance for e in self.embeddings]

    @property
    def concrete_types(self) -> list[str]:
        out: list[str] = []
        for e in self.embeddings:
            if e.concrete not in out:
                out.append(e.concrete)
        return out

    def split(self, path: str) -> Optional[tuple[Embedding, str]]:
        """(embedding, component) of a field path `veg_cf%core%agnpp`, or None."""
        for emb in sorted(self.embeddings, key=lambda e: -len(e.prefix)):
            if path.startswith(emb.prefix + "%"):
                return emb, path[len(emb.prefix) + 1 :]
        return None


def element_of(instance: str, type_name: str) -> str:
    for iso in ("c13", "c14"):
        if instance.startswith(f"{iso}_"):
            return iso
    for word, element in ELEMENT_WORDS.items():
        if word in type_name:
            return element
    return ""


def is_data_only(type_dict: dict, type_name: str) -> bool:
    """A derived type without type-bound procedures (what `resolve_nested` flattens)."""
    dtype = type_dict.get(type_name)
    return dtype is not None and not dtype.procedures


def embeds(comp: Variable) -> bool:
    """A scalar, non-pointer derived-type component (`type(base) :: core`)."""
    return not comp.nested and comp.dim == 0 and not comp.pointer and not comp.ptrscalar


def base_routine(sub_dict: dict, type_name: str, binding: str) -> Optional[str]:
    """The id of the plain module procedure `<type_name>_<binding>` (any module), or None."""
    suffix = f"::{type_name}_{binding}"
    matches = [sid for sid in sub_dict if sid.endswith(suffix)]
    if len(matches) > 1:
        raise ValueError(f"{type_name}_{binding} is defined in several modules: {matches}")
    return matches[0] if matches else None


def find_holder(fut, type_name: str, instances: Optional[list[str]] = None) -> Holder:
    """
    The holder `type_name`: its own instances if it has any, else every
    instance whose type embeds it as a data-only component. `instances`
    restricts the instances considered.
    """
    type_dict = fut.type_dict
    dtype = type_dict[type_name]
    embs: list[Embedding] = []
    # the analysis also lists a component declaration (`type(base) :: core` inside another
    # type) among a type's instances: a module instance is never a component of a type
    component_names = {
        cname
        for t in type_dict.values()
        for cname, comp in t.components.items()
        if comp.type == type_name and embeds(comp)
    }
    real = [i for i in dtype.instances if i not in component_names]
    if real:
        for inst in real:
            if instances is None or inst in instances:
                embs.append(Embedding(inst, type_name, "", element_of(inst, type_name)))
        routines = {k: v for k, v in dtype.procedures.items() if isinstance(v, str)}
    else:
        for tname, t in type_dict.items():
            for cname, comp in t.components.items():
                if comp.type == type_name and embeds(comp):
                    for inst in t.instances:
                        if inst in component_names or (instances is not None and inst not in instances):
                            continue
                        embs.append(Embedding(inst, tname, cname, element_of(inst, tname)))
        routines = {}
        for binding in BINDINGS:
            sid = base_routine(fut.subroutine_dict, type_name, binding)
            if sid is not None:
                routines[binding] = sid
    if not embs:
        raise ValueError(f"{type_name}: no instances, and no instance's type holds it as a component")
    return Holder(type_name, dtype, embs, routines)


def embedded_holders(type_dict: dict, dtype) -> list[tuple[str, str]]:
    """(component name, holder type) of the data-only derived-type components of `dtype`."""
    return [
        (cname, comp.type)
        for cname, comp in dtype.components.items()
        if embeds(comp) and is_data_only(type_dict, comp.type)
    ]


def srcroot_of(sub_dict: dict) -> Path:
    """components/elm/src of the analyzed checkout (from the driver's path)."""
    return Path(sub_dict[DRIVER_ID].filepath).parents[1]


def nutrient_manifest(fut) -> Optional[dict]:
    """The nutrient transform's manifest of the analyzed sources, if they carry one."""
    path = srcroot_of(fut.subroutine_dict) / NUTRIENT_MANIFEST
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def family_base(manifest: dict, concrete: str) -> Optional[str]:
    """The nutrient base type of a concrete type (the manifest's `families` entry holding it)."""
    for base, fam in (manifest.get("families") or {}).items():
        if isinstance(fam, dict) and concrete in (fam.get("types") or {}):
            return base
    return None


def role_types(manifest: dict, concrete_types: list[str]) -> list[str]:
    """The role holder types (`groups`: mineral/uptake/retrans) of the concrete types."""
    out: list[str] = []
    for role, grp in (manifest.get("groups") or {}).items():
        if isinstance(grp, dict) and any(c in (grp.get("types") or {}) for c in concrete_types):
            if role not in out:
                out.append(role)
    return out
