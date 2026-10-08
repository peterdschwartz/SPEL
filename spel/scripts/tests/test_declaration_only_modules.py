"""
elm_instMod is a declaration-only module: SPEL never compiles its routines and
generates its own elm_instMod providing the names a unit test imports from it.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

import spel.scripts.edit_files as ef
from spel.scripts.fortran_modules import (
    declaration_source,
    imported_from,
    parse_use_stmts,
)
from spel.scripts.functional_unit_test import elminst_lines
from spel.scripts.types import LineTuple, LogicalLineIterator
from spel.scripts.utilityFunctions import Variable

INST_SRC = """module elm_instMod
  use CanopyStateType            , only : canopystate_type
  use SoilWaterRetentionCurveMod , only : soil_water_retention_curve_type
  use WaterfluxType              , only : waterflux_type, wf_vars => waterflux_vars
  use ELMFatesInterfaceMod       , only : hlm_fates_interface_type
  implicit none
  public
  type(canopystate_type)                              :: canopystate_vars  ! canopy
  class(soil_water_retention_curve_type), allocatable :: soil_water_retention_curve
  type(hlm_fates_interface_type)                      :: alm_fates
contains
  subroutine elm_inst_biogeophys(bounds_proc)
    use fileutils, only : getfil
    call getfil('x', bounds_proc)
  end subroutine elm_inst_biogeophys
end module elm_instMod
"""


def uses(text: str) -> list:
    lines = [LineTuple(line=l + "\n", ln=i) for i, l in enumerate(text.split("\n"))]
    it = LogicalLineIterator(lines)
    return parse_use_stmts(
        [LineTuple(fl.line, it.get_start_ln()) for fl in it if fl.line.startswith("use")]
    )


def var(name: str, type_: str, ln: int) -> Variable:
    return Variable(type_, name, "", ln, 0, declaration="elm_instmod")


@pytest.fixture
def inst_mod(tmp_path: Path):
    path = tmp_path / "elm_instMod.F90"
    path.write_text(INST_SRC)
    return SimpleNamespace(
        name="elm_instmod",
        filepath=path,
        end_of_head_ln=10,
        use_stmts=uses(INST_SRC)[:4],
        global_vars={
            "canopystate_vars": var("canopystate_vars", "canopystate_type", 7),
            "soil_water_retention_curve": var(
                "soil_water_retention_curve", "soil_water_retention_curve_type", 8
            ),
            "alm_fates": var("alm_fates", "hlm_fates_interface_type", 9),
        },
    )


def dtype(module: str, *instances: Variable):
    return SimpleNamespace(declaration=module, instances={v.name: v for v in instances})


@pytest.fixture
def type_dict(inst_mod):
    g = inst_mod.global_vars
    return {
        "canopystate_type": dtype("canopystatetype", g["canopystate_vars"]),
        "soil_water_retention_curve_type": dtype(
            "soilwaterretentioncurvemod", g["soil_water_retention_curve"]
        ),
        "waterflux_type": dtype("waterfluxtype"),
    }


def test_imported_from_lists_remote_names():
    user = SimpleNamespace(
        use_stmts=uses(
            "use elm_instMod, only : soil_water_retention_curve, wf => wf_vars\n"
            "use other, only : x"
        )
    )
    mods = {"canopyfluxesmod": user}
    assert imported_from(mods, ["canopyfluxesmod"], "elm_instmod") == {
        "soil_water_retention_curve",
        "wf_vars",
    }


def test_declaration_source(inst_mod, type_dict):
    assert declaration_source(inst_mod, "canopystate_vars", type_dict) == (
        "canopystatetype",
        "canopystate_type",
    )
    # a re-export is used from the module it comes from
    assert declaration_source(inst_mod, "wf_vars", type_dict) == (
        "waterfluxtype",
        "wf_vars => waterflux_vars",
    )
    assert declaration_source(inst_mod, "unknown", type_dict) is None


def test_generated_elm_instmod_provides_imported_names(inst_mod, type_dict):
    text = "\n".join(
        elminst_lines(type_dict, inst_mod, imported={"wf_vars", "soil_water_retention_curve"})
    )
    assert "use canopystatetype, only : canopystate_type" in text
    assert "use soilwaterretentioncurvemod, only : soil_water_retention_curve_type" in text
    assert "use waterfluxtype, only : wf_vars => waterflux_vars" in text
    # declarations are copied from the source (class/allocatable kept, comments dropped)
    assert "class(soil_water_retention_curve_type), allocatable :: soil_water_retention_curve" in text
    assert "type(canopystate_type)" in text and "! canopy" not in text
    assert "alm_fates" not in text
    assert text.count("soil_water_retention_curve_type)") == 1


def test_declarations_only_comments_out_contained_routines():
    lines = [LineTuple(line=l + "\n", ln=i) for i, l in enumerate(INST_SRC.split("\n"))]
    work = [LineTuple(lt.line, lt.ln) for lt in lines]
    kept = ef.declarations_only(work, lines, head_end=10)
    assert [lt.ln for lt in kept] == list(range(10)) + [15, 16]
    assert all(lt.commented for lt in lines[10:15])
    assert not any(lt.commented for lt in lines[:10] + lines[15:])
