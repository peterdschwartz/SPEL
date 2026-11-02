from collections.abc import Iterable
from dataclasses import dataclass

from spel.scripts.types import Dependence, Pairs
from spel.scripts.utilityFunctions import Variable

NML_CASCADES = {
    # filter counts: 0 without use_crop; with it the count is any non-negative
    # integer (the .true. value is representative), see Dependence.trigger_constraint
    "num_pcropp": Dependence(
        cascade_var="num_pcropp",
        trigger="elm_varctl::use_crop",
        val_pairs=[
            Pairs(nml_val=".false.", cascade_val="0"),
            Pairs(nml_val=".true.", cascade_val="10"),
        ],
    ),
    "num_ppercropp": Dependence(
        cascade_var="num_ppercropp",
        trigger="elm_varctl::use_crop",
        val_pairs=[
            Pairs(nml_val=".false.", cascade_val="0"),
            Pairs(nml_val=".true.", cascade_val="10"),
        ],
    ),
    # col_pp%is_fates / veg_pp%is_fates are only set .true. by the FATES interface
    # (elmfates_interfaceMod), e.g. CH4 reads veg_pp%is_bareground (allocated only
    # with use_fates) under .not. col_pp%is_fates(c)'s else branch
    "is_fates": Dependence(
        cascade_var="is_fates",
        trigger="elm_varctl::use_fates",
        val_pairs=[
            Pairs(nml_val=".false.", cascade_val=".false."),
        ],
    ),
    # ice wedge polygon landunits are only added with use_polygonal_tundra
    # (initGridCellsMod), and polygon-only fields (col_ws%iwp_*) are only allocated then
    "ispolygon": Dependence(
        cascade_var="ispolygon",
        trigger="elm_varctl::use_polygonal_tundra",
        val_pairs=[
            Pairs(nml_val=".false.", cascade_val=".false."),
        ],
    ),
    # dynSubgridControlMod's accessors of the dynamic_subgrid namelist (transient PFT and crop
    # data from flanduse_timeseries): dynpft_interp/dyncrop_interp and the weight updates run only
    # with them. The namelist variables are locals of read_namelist, set to .false. before the read.
    **{
        f"get_do_transient_{what}": Dependence(
            cascade_var=f"get_do_transient_{what}",
            trigger=f"dynsubgridcontrolmod::do_transient_{what}",
            val_pairs=[
                Pairs(nml_val=".false.", cascade_val=".false."),
                Pairs(nml_val=".true.", cascade_val=".true."),
            ],
            trigger_defaults=(".false.",),
        )
        for what in ("pfts", "crops")
    },
    # AllocationMod::EvaluateSupplStatus sets these from the nutrient-supplement
    # namelist variables every step; during accelerated spinup (spinup_state == 1)
    # carbon_only is also forced on for the first nyears_ad_carbon_only years, so
    # the mapping is only given for spinup_state == 0 (otherwise unknown)
    **{
        name: Dependence(
            cascade_var=name,
            trigger=["allocationmod::suplnitro", "allocationmod::suplphos", "elm_varctl::spinup_state"],
            trigger_types=["character", "character", "integer"],
            val_pairs=[
                Pairs(nml_val=(nitro, phos, "0"), cascade_val=".true." if (nitro, phos) == on else ".false.")
                for nitro in ("none", "all")
                for phos in ("none", "all")
            ],
        )
        for name, on in (
            ("carbon_only", ("all", "all")),
            ("carbonnitrogen_only", ("none", "all")),
            ("carbonphosphorus_only", ("all", "none")),
        )
    },
}


def cascade_trigger_variables(names: Iterable[str]) -> dict[str, Variable]:
    """Variables for the cascade trigger namelist variables among `names`."""
    triggers: dict[str, tuple[str, str]] = {}
    for dep in NML_CASCADES.values():
        for trigger, ftype in zip(dep.all_triggers(), dep.trigger_types):
            module, _, name = trigger.partition("::")
            triggers.setdefault(name, (module, ftype))
    result: dict[str, Variable] = {}
    for name in names:
        if name in triggers:
            module, ftype = triggers[name]
            result[name] = Variable(ftype, name, "?", 0, 0, declaration=module)
    return result


@dataclass(frozen=True)
class DerivedConstant:
    """A constant a development branch derives from a namelist variable (e.g. in controlMod)."""

    name: str
    type: str
    source: str  # module::variable it is derived from
    fortran: str  # expression of the source variable

    @property
    def source_name(self) -> str:
        return self.source.split("::")[-1]


# Written by `spel create --capture-for` into the reference case's constants, so the
# development case can read them from the reference data
DERIVED_CONSTANTS = {
    # controlMod: select case (trim(urban_hac)) -> urban_hac_on_int/urban_wasteheat_int/off
    "urban_hac_int": DerivedConstant(
        name="urban_hac_int",
        type="integer",
        source="urbanparamstype::urban_hac",
        fortran="merge(1, merge(2, 0, trim(urban_hac) == 'ON_WASTEHEAT'), trim(urban_hac) == 'ON')",
    ),
}
