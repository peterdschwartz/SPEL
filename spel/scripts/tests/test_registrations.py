"""History/restart registrations recorded by the analysis (spel/scripts/registrations.py)."""

from types import SimpleNamespace

from spel.scripts.nml.analyze_ifs import get_if_blocks
from spel.scripts.registrations import find_registrations, parse_routine, registrations
from spel.scripts.types import IfType, LineTuple

SRC = """
subroutine init_history(this, begc, endc)
  class(column_water_state) :: this
  integer, intent(in) :: begc, endc
  real(r8), pointer :: data2dptr(:,:)
  call hist_addfld1d (fname='H2OSNO', units='mm', avgflag='A', long_name='snow depth (liquid water)', &
       ptr_col=this%h2osno, c2l_scale_type='urbanf')
  data2dptr => this%h2osoi_liq(:,-nlevsno+1:0)
  call hist_addfld2d (fname='SNO_LIQ', units='kg/m2', type2d='levsno', avgflag='A', &
       long_name='snow liquid water', ptr_col=data2dptr, default='inactive')
  call hist_addfld1d (fname='QFLX', units='mm/s', avgflag='A', long_name='x', ptr_col=col_wf%qflx_evap_tot)
  call hist_addfld1d (fname='TOTAL', units='mm', avgflag='A', long_name='y', ptr_col=total_water)
  call hist_addfld2d (fname='PCT_LANDUNIT', units='%', type2d='ltype', avgflag='A', &
       long_name='% of each landunit on topounit', ptr_topo=subgrid_weights_diagnostics%pct_landunit)
  if (use_cn) then
     call restartvar(ncid=ncid, flag=flag, varname='H2OSOI_LIQ', xtype=ncd_double, dim1name='column', &
          data=this%h2osoi_liq)
  end if
  call restartvar(ncid=ncid, flag=flag, varname='EFLX', xtype=ncd_double, data=veg_ef%eflx_lh_tot)
end subroutine init_history
"""


def _write(tmp_path):
    f = tmp_path / "x.F90"
    f.write_text(SRC)
    return f, SRC.count("\n") - 1


def test_bases_pointers_and_metadata(tmp_path):
    f, end = _write(tmp_path)
    regs = registrations(parse_routine(f, 1, end))
    by_path = {r.path: r for r in regs}
    assert set(by_path) == {
        "this%h2osno",
        "this%h2osoi_liq",
        "col_wf%qflx_evap_tot",
        "total_water",
        "veg_ef%eflx_lh_tot",
        "subgrid_weights_diagnostics%pct_landunit",
    }
    # topounit-level fields (ptr_topo) were invisible until 2026-10-30: PCT_LANDUNIT's source was
    # missing from the device-data layer and the history kernel aborted
    topo = by_path["subgrid_weights_diagnostics%pct_landunit"]
    assert topo.kind == "hist" and topo.base == "subgrid_weights_diagnostics" and topo.field == "pct_landunit"
    snow = by_path["this%h2osno"]
    assert snow.kind == "hist" and snow.base == "this" and snow.field == "h2osno"
    assert snow.name == "h2osno" and snow.units == "mm" and snow.long_name == "snow depth (liquid water)"
    # the pointer alias is attributed to its target; the same field is also restart-registered
    kinds = {(r.kind, r.inactive) for r in regs if r.path == "this%h2osoi_liq"}
    assert kinds == {("hist", True), ("restart", False)}
    liq_restart = next(r for r in regs if r.path == "this%h2osoi_liq" and r.kind == "restart")
    assert liq_restart.guards == ("use_cn",)
    inst = by_path["col_wf%qflx_evap_tot"]
    assert inst.base == "col_wf" and inst.field == "qflx_evap_tot"
    bare = by_path["total_water"]
    assert bare.base == "total_water" and bare.field == ""


def test_find_registrations_sets_attribute(tmp_path):
    f, end = _write(tmp_path)
    with_regs = SimpleNamespace(id="m::init_history", library=False, filepath=f, startline=1, end_stmt_ln=end, endline=end)
    plain = SimpleNamespace(id="m::other", library=False, filepath=f, startline=2, end_stmt_ln=4, endline=4)
    lib = SimpleNamespace(id="lib::x", library=True, filepath=None, startline=-999, end_stmt_ln=-999, endline=-999)
    n = find_registrations({"a": with_regs, "b": plain, "c": lib})
    assert n == 7
    assert len(with_regs.registrations) == 7
    assert plain.registrations == [] and lib.registrations == []


def test_one_line_if_has_end_line():
    lines = [
        "subroutine demo(begp)",
        "integer, optional :: begp",
        "if (present(begp)) begp = 1",
        "if (use_cn) then",
        "  x = 1",
        "end if",
        "end subroutine demo",
    ]
    sub = SimpleNamespace(
        name="demo", sub_lines=[LineTuple(line=l, ln=i) for i, l in enumerate(lines)], flat_ifs=[], ifs_analyzed=False
    )
    get_if_blocks(sub)
    by_start = {f.start_ln: f for f in sub.flat_ifs}
    assert by_start[2].end_ln == 2 and by_start[2].kind is IfType.SIMPLE
    assert by_start[3].end_ln == 5 and by_start[3].kind is IfType.IF
