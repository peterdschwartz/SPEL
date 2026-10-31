"""
Components for the fields the nitrogen and phosphorus types share (carbon
doesn't have them), grouped by the role they play.

Instead of another level of types, each role is a small type held by the N
and P types as a component, like `core` (extract.py):

  mineral  the soil mineral nutrient pool and its fluxes: mineralization,
           immobilization, decomposition-cascade mineral fluxes, deposition,
           supplement, leaching (column level)
  uptake   plant demand and uptake from the soil mineral pool, and its
           deployment (column and vegetation level)
  retrans  the plant pool of retranslocated nutrient and its fluxes: from the
           organs, deployment, mortality/fire/harvest losses (vegetation level)

The type of a role's component is the family's name with ``nutrient``
replaced by the role (column_mineral_flux, vegetation_retrans_state). Fields
keep their base names (common.py), except where that repeats the component
(FIELD_NAMES). The grouping is the starting point for the domain
scientists: changing it is an edit of GROUPS (or a TOML file given with `--groups`) and a regeneration.
"""

from __future__ import annotations

from typing import Optional

from spel.transforms.nutrient.common import Family
from spel.transforms.nutrient.extract import ExtractError, FamilyEdit

NP = ("n", "p")
# role -> family -> base names (common.py) of the N/P fields it holds
GROUPS: dict[str, dict[str, list[str]]] = {
    "mineral": {
        "column_nutrient_state": ["smin", "smin_vr"],
        "column_nutrient_flux": [
            "dep_to_smin",
            "decomp_cascade_smin_flux_vr",
            "decomp_cascade_smin_flux",
            "supplement_to_smin_vr",
            "supplement_to_smin",
            "gross_min_vr",
            "gross_min",
            "net_min_vr",
            "net_min",
            "smin_leached_vr",
            "smin_leached",
            "pmf_decomp_cascade",
            "soil_immob_flux",
            "soil_immob_flux_vr",
            "soil_grossmin_flux",
        ],
    },
    "uptake": {
        "column_nutrient_flux": [
            "plant_demand",
            "plant_demand_vr",
            "col_plant_demand_vr",
            "smin_to_plant_vr",
            "smin_to_plant",
            "plant_uptake_flux",
        ],
        "vegetation_nutrient_flux": [
            "plant_demand",
            "plant_demand_vr",
            "smin_to_plant",
            "smin_to_pool",
            "supplement_to_plant",
            "pool_deploy",
        ],
    },
    "retrans": {
        "vegetation_nutrient_state": ["retrans"],
        "vegetation_nutrient_flux": [
            "m_retrans_to_litter",
            "hrv_retrans_to_litter",
            "m_retrans_to_fire",
            "m_retrans_to_litter_fire",
            "leaf_to_retrans",
            "froot_to_retrans",
            "retrans_to_pool",
            "livestem_to_retrans",
            "livecroot_to_retrans",
            "avail_retrans",
        ],
    },
}
# (role, base name) -> its name in the role's type
FIELD_NAMES = {("retrans", "retrans"): "pool"}


def load_groups(path) -> dict[str, dict[str, list[str]]]:
    """GROUPS from a TOML file: one table per role, family name -> base names:

        [mineral]
        column_nutrient_state = ["smin", "smin_vr"]
    """
    import tomllib
    from pathlib import Path

    data = tomllib.loads(Path(path).read_text())
    for role, per_family in data.items():
        if not isinstance(per_family, dict) or not all(
            isinstance(v, list) and all(isinstance(b, str) for b in v) for v in per_family.values()
        ):
            raise ExtractError(f"{path}: [{role}] must map family names to lists of base names")
    return data


def group_edits(families: list[Family], groups: Optional[dict[str, dict[str, list[str]]]] = None) -> list[FamilyEdit]:
    by_name = {f.name: f for f in families}
    edits = []
    for role, per_family in (GROUPS if groups is None else groups).items():
        for fam_name, bases in per_family.items():
            fam = by_name.get(fam_name)
            if fam is None:
                raise ExtractError(f"{role}: no family {fam_name}")
            pairs = {f.base: f for f in fam.by_elements("np")}
            missing = [b for b in bases if b not in pairs]
            if missing:
                raise ExtractError(f"{role}: {fam_name} has no N/P-only fields {missing}")
            fields = [pairs[b] for b in bases]
            bad = [f.base for f in fields if not f.consistent]
            if bad:
                raise ExtractError(f"{role}: {fam_name} fields declared differently in N and P: {bad}")
            names = {b: n for (r, b), n in FIELD_NAMES.items() if r == role and b in bases}
            edits.append(
                FamilyEdit(
                    fam, fields, fam_name.replace("nutrient", role), role, NP, names,
                    note=f"{role} fields of the N/P types",
                )
            )
    return edits
