"""Run configurations (core/config.py): namelist values decide which branches and routines run."""
from types import SimpleNamespace as NS

import pytest

from spel.scripts.fortran_parser.boolen_expression import Expectation
from spel.transforms.core.config import PRESETS, Configuration, parse_namelist_values


def test_parse_lnd_in_and_cfg(tmp_path):
    p = tmp_path / "lnd_in"
    p.write_text("""\
&elm_inparm
 use_crop = .true.   ! a comment
 fsurdat = '/path/with!bang/surf.nc'
 hist_fincl2 = 'GPP','ER'
 maxpatch_pft = 51
/
&ndepdyn_nml
 stream_year_first_ndep = 1850
/
""")
    vals = parse_namelist_values(p)
    assert vals == {"use_crop": ".true.", "fsurdat": "'/path/with!bang/surf.nc'", "hist_fincl2": "'GPP','ER'",
                    "maxpatch_pft": "51", "stream_year_first_ndep": "1850"}
    cfg = Configuration.load(str(p))
    assert cfg.values["use_crop"] is True and cfg.values["maxpatch_pft"] == 51


def test_presets_load_by_name():
    names = {x.stem for x in PRESETS.glob("*.cfg")}
    assert "CNPCROP" in names
    cfg = Configuration.load("CNPCROP")
    assert cfg.values["use_crop"] is True and cfg.values["use_hydrstress"] is False
    with pytest.raises(FileNotFoundError):
        Configuration.load("NO_SUCH_PRESET")


def flat(start, end, var, constraint):
    return NS(start_ln=start, end_ln=end, nml_vars={var: NS(variable=None)}, nml_cascades={},
              condtional_expectation=Expectation(var, constraint))


def test_dead_branches_and_lines():
    sub = NS(id="m::s", flat_ifs=[
        flat(10, 20, "use_hydrstress", "True"),     # if (use_hydrstress) then ... (lines 11-19)
        flat(20, 25, "use_hydrstress", "False"),    # else ...
        flat(30, 30, "use_fates", "True"),          # if (use_fates) call x
        flat(40, 45, "use_crop", "True"),           # runs
        flat(50, 55, "unknown_flag", "True"),       # undecided: active
    ])
    cfg = Configuration({"use_hydrstress": False, "use_fates": False, "use_crop": True})
    assert cfg.dead_branches(sub) == [(11, 19), (30, 30)]
    assert not cfg.line_active(sub, 15) and cfg.line_active(sub, 22) and cfg.line_active(sub, 10)
    assert not cfg.span_active(sub, 30, 30) and cfg.line_active(sub, 42) and cfg.line_active(sub, 52)
    assert cfg.span_active(sub, 9, 15)  # partly outside the dead branch


def test_defaults_come_from_the_declarations():
    var = NS(default_value="False")
    f = NS(start_ln=1, end_ln=5, nml_vars={"active_dynamic_subgrid": NS(variable=var)}, nml_cascades={},
           condtional_expectation=Expectation("active_dynamic_subgrid", "True"))
    fut = NS(subroutine_dict={"m::s": NS(id="m::s", flat_ifs=[f])})
    cfg = Configuration({"use_crop": True})
    cfg.add_defaults(fut)
    assert cfg.values["active_dynamic_subgrid"] is False and cfg.defaults == {"active_dynamic_subgrid"}
    assert cfg.dead_branches(fut.subroutine_dict["m::s"]) == [(2, 4)]


def test_call_tree_drops_routines_reached_only_through_dead_branches():
    phs = NS(id="m::hybrid_phs", child_subroutines={}, sub_call_desc={}, flat_ifs=[])
    leaf = NS(id="m::leaf", child_subroutines={}, sub_call_desc={}, flat_ifs=[])
    both = NS(id="m::both", child_subroutines={}, sub_call_desc={}, flat_ifs=[])
    root = NS(id="m::root", flat_ifs=[flat(10, 20, "use_hydrstress", "True")],
              child_subroutines={"hybrid_phs": phs, "leaf": leaf, "both": both},
              sub_call_desc={12: NS(fn="hybrid_phs"), 30: NS(fn="leaf"), 13: NS(fn="both"), 31: NS(fn="both")})
    fut = NS(subroutine_dict={s.id: s for s in (root, phs, leaf, both)})
    cfg = Configuration({"use_hydrstress": False})
    assert set(cfg.call_tree(fut, ["m::root"])) == {"m::root", "m::leaf", "m::both"}
    assert set(Configuration({"use_hydrstress": True}).call_tree(fut, ["m::root"])) == set(fut.subroutine_dict)
