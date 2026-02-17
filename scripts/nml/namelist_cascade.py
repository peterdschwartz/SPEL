from scripts.types import Dependence, Pairs

NML_CASCADES = {
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
}
