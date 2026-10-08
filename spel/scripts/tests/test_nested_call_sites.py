"""Routines elm_drv reaches through other routines (nested call sites)."""

from types import SimpleNamespace

from spel.scripts.driver_callsites import call_path, callers_of, compose_bindings
from spel.scripts.fortran_parser.symbols import Origin
from spel.scripts.record_access import ArgBinding

DRV = "elm_driver::elm_drv"
ECO = "ecosystemdynmod::ecosystemdynnoleaching2"
SLVT = "soillittverttranspmod::soillittverttransp"
HELPER = "m::helper"


def binding(caller, callee, ln, argn, dummy, actual, origin):
    return ArgBinding(caller, callee, ln, argn, dummy, actual, origin)


def sub(id_, bindings=(), children=()):
    return SimpleNamespace(
        id=id_,
        library=False,
        child_subroutines={c: None for c in children},
        record_access=SimpleNamespace(bindings=list(bindings)),
    )


DRIVER_BINDINGS = [
    binding(DRV, ECO, 1143, 0, "bounds", "bounds_clump", Origin.LOCAL),
    binding(DRV, ECO, 1143, 1, "num_soilc", "filter%num_soilc", Origin.GLOBAL),
    binding(DRV, ECO, 1143, 2, "filter_soilc", "filter%soilc", Origin.GLOBAL),
    binding(DRV, ECO, 1143, 3, "cnstate_vars", "cnstate_vars", Origin.GLOBAL),
    binding(DRV, ECO, 1200, 3, "cnstate_vars", "other_vars", Origin.GLOBAL),
]


def sub_dict():
    eco = sub(
        ECO,
        bindings=[
            binding(ECO, SLVT, 715, 0, "num_soilc", "num_soilc", Origin.DUMMY),
            binding(ECO, SLVT, 715, 1, "filter_soilc", "filter_soilc", Origin.DUMMY),
            binding(ECO, SLVT, 715, 2, "cnstate_vars", "cnstate_vars", Origin.DUMMY),
            binding(ECO, SLVT, 715, 3, "frac", "frac_local", Origin.LOCAL),
            binding(ECO, HELPER, 700, 0, "cs", "cnstate_vars%a", Origin.DUMMY),
        ],
        children=[HELPER],
    )
    helper = sub(
        HELPER,
        bindings=[binding(HELPER, SLVT, 10, 2, "cnstate_vars", "cs", Origin.DUMMY)],
    )
    slvt = sub(SLVT)
    return {ECO: eco, HELPER: helper, SLVT: slvt}


def test_call_path_is_shortest_chain():
    sd = sub_dict()
    assert call_path(sd, {ECO}, SLVT) == [ECO, SLVT]
    assert call_path(sd, {ECO}, HELPER) == [ECO, HELPER]
    assert call_path(sd, {ECO}, "x::unknown") is None
    assert callers_of(sd, SLVT) == [ECO, HELPER]


def test_compose_bindings_to_elm_drv_actuals():
    out = {
        b.dummy: (b.caller, b.actual, b.origin, b.argn)
        for b in compose_bindings(sub_dict(), DRIVER_BINDINGS, [ECO, SLVT])
    }
    assert out == {
        "num_soilc": (DRV, "filter%num_soilc", Origin.GLOBAL, 0),
        "filter_soilc": (DRV, "filter%soilc", Origin.GLOBAL, 1),
        # first elm_drv call site only
        "cnstate_vars": (DRV, "cnstate_vars", Origin.GLOBAL, 2),
        # local of the intermediate routine: kept, with its caller
        "frac": (ECO, "frac_local", Origin.LOCAL, 3),
    }


def test_compose_bindings_through_components():
    out = compose_bindings(sub_dict(), DRIVER_BINDINGS, [ECO, HELPER, SLVT])
    assert [(b.dummy, b.actual, b.origin) for b in out] == [
        ("cnstate_vars", "cnstate_vars%a", Origin.GLOBAL)
    ]

