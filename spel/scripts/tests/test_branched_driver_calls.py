import textwrap
from types import SimpleNamespace

import spel.scripts.config as cfg
from spel.scripts.driver_callsites import DriverCall, DriverSites
from spel.scripts.functional_unit_test import FunctionalUnitTest

DRIVER = textwrap.dedent("""\
    module elm_driver
    contains
      subroutine elm_drv()
        call t_startf('hydro2 drainage')
        if (use_elm_interface .and. (use_pflotran .and. pf_hmode)) then
          ! pflotran only works on soilc
          call HydrologyDrainage(bounds_clump,           &
             filter(nc)%num_hydrononsoic, filter(nc)%hydrononsoic)
        else
          call HydrologyDrainage(bounds_clump,           &
             filter(nc)%num_hydrologyc, filter(nc)%hydrologyc)
        end if
        call t_stopf('hydro2 drainage')
      end subroutine elm_drv
    end module elm_driver
    """)
HD = "hydrologydrainagemod::hydrologydrainage"


def test_calls_in_both_branches_keep_the_if(tmp_path, monkeypatch):
    path = tmp_path / "components/elm/src/main/elm_driver.F90"
    path.parent.mkdir(parents=True)
    path.write_text(DRIVER)
    monkeypatch.setattr(cfg, "E3SM_SRCROOT", tmp_path)
    sites = DriverSites(
        module="elm_driver",
        routine="elm_drv",
        path="components/elm/src/main/elm_driver.F90",
        calls={HD: [DriverCall(HD, 6, "bounds_clump"), DriverCall(HD, 9, "bounds_clump")]},
        capture_lns=[],
    )
    fake = SimpleNamespace(driver_sites=sites)
    sub = SimpleNamespace(id=HD)
    block, calls, *_ = FunctionalUnitTest._branched_driver_calls(fake, sub)
    assert [ln.split("(")[0].strip() for ln in block] == [
        "if", "call hydrologydrainage", "else", "call hydrologydrainage", "end if"
    ]
    assert "hydrononsoic" in calls[0] and "hydrologyc" in calls[1]

    # globals in the condition are captured as constants and imported
    elm_varctl = SimpleNamespace(
        global_vars={
            v: SimpleNamespace(name=v, parameter=False)
            for v in ("use_elm_interface", "use_pflotran", "pf_hmode")
        }
    )
    fake.module_dict = {"elm_varctl": elm_varctl}
    fake.active_global_vars = {}
    mods = FunctionalUnitTest._capture_condition_globals(fake, block)
    assert sorted(fake.active_global_vars) == ["pf_hmode", "use_elm_interface", "use_pflotran"]
    assert "use elm_varctl, only : use_pflotran\n" in mods


def test_driver_arguments_come_from_the_capture(tmp_path, monkeypatch):
    from spel.scripts import instrument_elm as ie

    driver = textwrap.dedent("""\
        module elm_driver
        contains
          subroutine elm_drv(doalb, nextsw_cday, declinp1, declin, rstwr, nlend, rdate)
            logical,  intent(in) :: doalb
            real(r8), intent(in) :: nextsw_cday ! calendar day for nstep+1
            real(r8), intent(in) :: declinp1
            real(r8), intent(in) :: declin
            call SurfaceAlbedo(bounds_clump, nextsw_cday, declinp1, surfalb_vars)
          end subroutine elm_drv
        end module elm_driver
        """)
    path = tmp_path / "components/elm/src/main/elm_driver.F90"
    path.parent.mkdir(parents=True)
    path.write_text(driver)
    monkeypatch.setattr(cfg, "E3SM_SRCROOT", tmp_path)
    fake = SimpleNamespace(
        driver_sites=SimpleNamespace(path="components/elm/src/main/elm_driver.F90", routine="elm_drv"),
        driver_args={},
    )
    fake._driver_scalar_dummies = lambda: FunctionalUnitTest._driver_scalar_dummies(fake)
    assert fake._driver_scalar_dummies() == {
        "doalb": "logical", "nextsw_cday": "real", "declinp1": "real", "declin": "real"
    }
    calls, used = FunctionalUnitTest._use_driver_args(
        fake, ["call surfacealbedo(bounds_clump, nextsw_cday, declinp1=declinp1, surfalb_vars)",
               "call ecosystemdynnoleaching1(bounds_clump, doalb, cnstate_vars)"]
    )
    assert calls == [
        "call surfacealbedo(bounds_clump, spel_drv_nextsw_cday(1), declinp1=spel_drv_declinp1(1), surfalb_vars)",
        # a logical dummy is stored as an integer (0/1), like the time scalars
        "call ecosystemdynnoleaching1(bounds_clump, (spel_drv_doalb(1) /= 0), cnstate_vars)",
    ]
    assert used == {"nextsw_cday", "declinp1", "doalb"}

    names = ie.CaptureNames.for_case("ref_sa", tuple(fake.driver_args.items()))
    module = ie.generate_capture_module(names, [])
    assert "subroutine spel_capture_inputs_ref_sa(bounds, nextsw_cday, declinp1, doalb)" in module
    assert "logical, intent(in) :: doalb" in module
    assert "spel_drv_declinp1(1) = declinp1" in module
    assert "spel_drv_doalb(1) = merge(1, 0, doalb)" in module
