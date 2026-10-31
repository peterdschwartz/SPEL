from spel.scripts.fortran_parser.scope_walk import AssignEvent
from spel.scripts.fortran_parser.tests.test_scope_walk import gvar, walk
from spel.transforms.compact.index_forms import IndexContext, compaction_class

SRC = """
subroutine s(num_soilp, filter_soilp, num_soilc, filter_soilc, bounds)
  implicit none
  integer, intent(in) :: num_soilp, filter_soilp(:), num_soilc, filter_soilc(:)
  type(bounds_type), intent(in) :: bounds
  integer :: fp, fc, p, c, pi, q
  associate(lmr => veg_cf%leaf_mr, pcol => veg_pp%column)
  do fp = 1, num_soilp
     p = filter_soilp(fp)
     c = pcol(p)
     lmr(p) = col_cf%hr_vr(c, 1)
     q = p
     veg_cf%gpp(q) = 0._r8
  end do
  do fc = 1, num_soilc
     c = filter_soilc(fc)
     do p = col_pp%pfti(c), col_pp%pftf(c)
        col_cf%gpp(c) = col_cf%gpp(c) + veg_cf%gpp(p)
     end do
  end do
  do c = bounds%begc, bounds%endc
     col_cf%er(c) = 0._r8
  end do
  veg_cf%npp(bounds%begp:bounds%endp) = 0._r8
  end associate
end subroutine s
"""

ENV = {
    "filter_soilp": frozenset({"filter%soilp"}),
    "num_soilp": frozenset({"filter%num_soilp"}),
    "filter_soilc": frozenset({"filter%soilc"}),
    "num_soilc": frozenset({"filter%num_soilc"}),
}


def forms():
    rec = walk(SRC, globals=[gvar(n) for n in ("veg_cf", "veg_pp", "col_cf", "col_pp")])
    ctx = IndexContext("m::s", rec, ENV, {})
    out = {}
    for pos, e in enumerate(rec.events):
        if hasattr(e, "subscripts") and e.path.split("%")[0] in ("veg_cf", "col_cf"):
            sg = "p" if e.path.startswith("veg_cf") else "c"
            form = ctx.access_form(e, pos)
            out.setdefault((e.path, e.ln), (str(form), compaction_class(form, sg)))
    return rec, out


def test_access_subscripts_and_assign_events():
    rec, _ = forms()
    a = next(e for e in rec.accesses(path="col_cf%hr_vr"))
    assert a.subs == ("c", "1")
    assign = next(e for e in rec.events if isinstance(e, AssignEvent) and e.lhs.path == "c")
    assert [r.path for r in assign.reads] == ["p", "veg_pp%column"]


def test_forms():
    _, f = forms()
    assert f[("veg_cf%leaf_mr", 10)] == ("filter%soilp[1..filter%num_soilp]", "main")
    assert f[("col_cf%hr_vr", 10)][1] == "map"  # c = veg_pp%column(p)
    assert f[("veg_cf%gpp", 12)][1] == "main"  # copy q = p
    assert f[("veg_cf%gpp", 17)][1] == "map"  # pfti(c)..pftf(c) from soilc
    assert f[("col_cf%gpp", 17)][1] == "main"
    assert f[("col_cf%er", 21)][1] == "range"  # begc..endc
    assert f[("veg_cf%npp", 23)][1] == "whole"


def test_guard_requires_active_patch():
    from spel.scripts.fortran_parser.spel_parser import Parser
    from spel.scripts.registrations import iter_statements
    from spel.scripts.types import LineTuple
    from spel.transforms.compact.select import _requires_active

    src = """\
if (veg_pp%active(p) .and. (.not. use_fates)) then
   x = 1
else if (veg_pp % active( p ) .or. y) then
   x = 2
else
   x = 3
end if"""
    block = Parser(lines=[LineTuple(line=l, ln=i) for i, l in enumerate(src.split("\n"))]).parse_program()
    guards = {stmt.lineno: g for stmt, g in iter_statements(block)}
    assert [_requires_active(g, "p") for g in guards[1]] == [True]
    assert [_requires_active(g, "p") for g in guards[3]] == [False, False]  # else branch; `.or.`
    assert not any(_requires_active(g, "p") for g in guards[5])
    assert not _requires_active(guards[1][0], "c")
