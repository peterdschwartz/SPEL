"""Loops over the patches of a filtered column, and the filter map that replaces them (core/filter_maps.py)."""
import pytest

from spel.scripts.fortran_parser.tests.test_scope_walk import gvar, walk
from spel.transforms.compact.index_forms import IndexContext
from spel.transforms.core.filter_maps import FILTER_MAPS, child_loops, mapped_header

SRC = """
subroutine s(num_soilc, filter_soilc, num_soilp, filter_soilp)
  implicit none
  integer, intent(in) :: num_soilc, filter_soilc(:), num_soilp, filter_soilp(:)
  integer :: fc, c, p, pi, cc
  associate(pfti => col_pp%pfti, pftf => col_pp%pftf)
  do fc = 1, num_soilc
     c = filter_soilc(fc)
     do p = pfti(c), pftf(c)
        col_cf%gpp(c) = col_cf%gpp(c) + veg_cf%gpp(p) * veg_pp%wtcol(p)
     end do
  end do
  do fc = 1, num_soilc
     c = filter_soilc(fc)
     do pi = 1, maxpatch_pft
        if (pi <= col_pp%npfts(c)) then
           p = col_pp%pfti(c) + pi - 1
           if (veg_pp%active(p) .and. veg_pp%wtcol(p) > 0._r8) then
              col_cf%er(c) = col_cf%er(c) + veg_cf%er(p)
           end if
        end if
     end do
  end do
  do c = 1, 10
     do p = col_pp%pfti(c), col_pp%pftf(c)
        veg_cf%npp(p) = 0._r8
     end do
  end do
  do fc = 1, num_soilc
     cc = filter_soilc(fc)
     c = cc
     do pi = 1, col_pp%npfts(c)
        p = col_pp%pfti(c) + pi - 1
        veg_cf%npp(p) = 1._r8
     end do
  end do
  end associate
end subroutine s
"""

GLOBALS = ("veg_cf", "veg_pp", "col_cf", "col_pp", "maxpatch_pft")


def loops(instance):
    rec = walk(SRC, globals=[gvar(n) for n in GLOBALS])
    env = {"filter_soilc": frozenset({f"{instance}%soilc"}), "num_soilc": frozenset({f"{instance}%num_soilc"}),
           "filter_soilp": frozenset({f"{instance}%soilp"}), "num_soilp": frozenset({f"{instance}%num_soilp"})}
    return {cl.loop.start_ln: cl for cl in child_loops(IndexContext("m::s", rec, env, {}))}


def test_both_styles_found_and_a_loop_over_bounds_is_not():
    found = loops("filter")
    assert sorted(found) == [8, 14, 31]  # the loop over c = 1..10 has no filtered parent
    r, s, copy = found[8], found[14], found[31]
    assert (r.style, r.child, r.parent, r.parent_pos) == ("range", "p", "c", "fc")
    assert (s.style, s.child, s.parent, s.parent_pos, s.assign_ln) == ("slots", "p", "c", "fc", 16)
    assert copy.style == "slots" and copy.parent_pos == "fc"  # c = cc = filter_soilc(fc)
    assert r.filters == {"filter%soilc"} and r.map is FILTER_MAPS["soilc"]


def test_exact_for_inactive_and_active_and_for_active_guards():
    found = loops("filter")
    assert not found[8].exact and "line 10" in found[8].reason  # unguarded body: inactive patches add in
    assert found[14].exact  # everything under veg_pp%active(p)
    assert not found[31].exact
    assert all(cl.exact for cl in loops("filter_inactive_and_active").values())


def test_unknown_filters_are_not_exact():
    rec = walk(SRC, globals=[gvar(n) for n in GLOBALS])
    found = child_loops(IndexContext("m::s", rec, {}, {}))  # dummies unbound
    assert found == []  # the parent filter must be known


def test_mapped_header():
    cl = loops("filter_inactive_and_active")[8]
    assert mapped_header(cl, "filter_inactive_and_active(nc)", "fp", "filter_soilp") == [
        "do fp = filter_inactive_and_active(nc)%soilc_soilp_beg(fc), filter_inactive_and_active(nc)%soilc_soilp_end(fc)",
        "   p = filter_soilp(fp)",
    ]
    assert "soilc_soilp_beg" in cl.describe() and "exact" in cl.describe()


CYCLE = """
subroutine s(num_soilc, filter_soilc)
  implicit none
  integer, intent(in) :: num_soilc, filter_soilc(:)
  integer :: fc, c, p, pi
  do fc = 1, num_soilc
     c = filter_soilc(fc)
     do p = col_pp%pfti(c), col_pp%pftf(c)
        if (.not. veg_pp%active(p) .or. veg_pp%wtcol(p) == 0._r8) cycle
        col_cf%gpp(c) = col_cf%gpp(c) + veg_cf%gpp(p)
     end do
  end do
  do fc = 1, num_soilc
     do pi = 1, maxpatch_pft
        c = filter_soilc(fc)
        if (pi <= col_pp%npfts(c)) then
           p = col_pp%pfti(c) + pi - 1
           if (veg_pp%active(p)) col_cf%er(c) = col_cf%er(c) + veg_cf%er(p)
        end if
     end do
  end do
  do fc = 1, num_soilc
     c = filter_soilc(fc)
     do p = col_pp%pfti(c), col_pp%pftf(c)
        if (veg_cf%gpp(p) > 0._r8) then
           if (.not. veg_pp%active(p)) cycle
        end if
        col_cf%gpp(c) = col_cf%gpp(c) + veg_cf%gpp(p)
     end do
  end do
end subroutine s
"""


def test_cycle_on_inactive_and_parent_set_in_the_loop():
    rec = walk(CYCLE, globals=[gvar(n) for n in GLOBALS])
    env = {"filter_soilc": frozenset({"filter%soilc"}), "num_soilc": frozenset({"filter%num_soilc"})}
    found = {cl.loop.start_ln: cl for cl in child_loops(IndexContext("m::s", rec, env, {}))}
    assert sorted(found) == [7, 13, 23]
    assert found[7].exact  # cycled for inactive patches (an .or. term)
    assert found[13].exact  # c = filter_soilc(fc) in the slot loop is per column
    assert not found[23].exact  # the cycle is conditional


def test_context_groups_the_active_loops(monkeypatch):
    from types import SimpleNamespace as NS

    from spel.transforms.core import filter_maps
    from spel.transforms.core.transform import Context

    def cl(sub, ln):
        return NS(sub_id=sub, loop=NS(start_ln=ln, end_ln=ln + 3))

    found = [cl("m::a", 10), cl("m::a", 40), cl("m::b", 5)]
    monkeypatch.setattr(filter_maps, "scan", lambda fut, routines=None: found)
    fut = NS(subroutine_dict={"m::a": NS(id="m::a"), "m::b": NS(id="m::b")})
    ctx = Context(fut, srcroot="/tmp")
    assert {k: [c.loop.start_ln for c in v] for k, v in ctx.child_loops.items()} == {"m::a": [10, 40], "m::b": [5]}
    dead = NS(call_tree=lambda fut, roots, exclude=None: {"m::a": None},
              span_active=lambda sub, lo, hi: not (sub.id == "m::a" and lo == 40))
    ctx = Context(fut, srcroot="/tmp", config=dead)
    assert {k: [c.loop.start_ln for c in v] for k, v in ctx.child_loops.items()} == {"m::a": [10], "m::b": [5]}
