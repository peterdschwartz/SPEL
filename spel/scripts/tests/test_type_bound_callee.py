"""elm_drv's type-bound calls on module variables resolve to the bound routine."""

from types import SimpleNamespace

from spel.scripts.driver_callsites import type_bound_callee
from spel.scripts.fortran_parser.scope_walk import CallEvent


def _event(name, obj):
    ev = object.__new__(CallEvent)
    object.__setattr__(ev, "name", name)
    object.__setattr__(ev, "passed_object", obj)
    return ev


def test_type_bound_callee():
    sub = SimpleNamespace(id="vegtypes::veg_cs_summary")
    mods = {"vegdata": SimpleNamespace(global_vars={"veg_cs": SimpleNamespace(type="vegetation_carbon_state")}),
            "other": SimpleNamespace(global_vars={})}
    types = {"vegetation_carbon_state": SimpleNamespace(procedures={"summary": sub.id})}
    subs = {sub.id: sub}
    assert type_bound_callee(_event("veg_cs%Summary", "veg_cs"), mods, subs, types) is sub
    assert type_bound_callee(_event("veg_cs%Init", "veg_cs"), mods, subs, types) is None       # not bound
    assert type_bound_callee(_event("x%veg_cs%Summary", "x%veg_cs"), mods, subs, types) is None  # component
    mods["other"].global_vars["veg_cs"] = SimpleNamespace(type="vegetation_carbon_state")
    assert type_bound_callee(_event("veg_cs%Summary", "veg_cs"), mods, subs, types) is None     # ambiguous


def test_window_calls_keeps_calls_between_single_call_routines():
    import logging

    from spel.scripts.analysis_cache import window_calls

    c = lambda ln: SimpleNamespace(ln=ln)
    calls = {"a": [c(550)], "s": [c(407), c(480), c(561)], "b": [c(589)]}
    out = window_calls(calls, logging.getLogger("t"))
    assert [x.ln for x in out["s"]] == [561] and out["a"] == calls["a"]
    assert window_calls({"s": [c(1), c(9)]}, logging.getLogger("t"))["s"] == [c(1), c(9)]


def test_call_guards_give_fortran_conditions_at_call_lines():
    from spel.scripts.driver_callsites import call_guards
    from spel.scripts.fortran_parser.scope_walk import CallEvent
    from spel.scripts.fortran_parser.tests.test_scope_walk import walk

    rec = walk(
        """
subroutine drv(doalb, n, use_cn)
  logical, intent(in) :: doalb, use_cn
  integer, intent(in) :: n
  call a(n)
  if (use_cn .and. doalb) then
     call b(n)
  else
     call c(n)
  end if
end subroutine drv
"""
    )
    guards = call_guards(rec)
    by_name = {e.name: guards.get(e.ln, ()) for e in rec.events if isinstance(e, CallEvent)}
    assert by_name["a"] == ()
    (gb,) = by_name["b"]
    assert gb.names == {"use_cn", "doalb"}
    assert gb.text.replace(" ", "").lower() == "(use_cn.and.doalb)"
    (gc,) = by_name["c"]
    assert gc.text.replace(" ", "").lower() == ".not.((use_cn.and.doalb))"
