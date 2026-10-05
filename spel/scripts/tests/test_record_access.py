"""
record_access: legacy-shaped access maps (`*_access_by_ln`) derived from a
SubroutineRecord. Call/function actual args take the callee's dummy status
at the call line.

Line numbers: the leading newline in each snippet makes the header line 1.
"""

from spel.scripts.fortran_parser.symbols import Origin, Symbol, SymbolKind
from spel.scripts.fortran_parser.tests.test_scope_walk import gvar, walk
from spel.scripts.record_access import (
    AccessMapper,
    AccessMaps,
    ArgBinding,
    access_summary,
    callsite_globals,
    dummy_actuals,
    elmtype_view,
    propagated_access,
    single_instance_actuals,
)
from spel.scripts.types import ReadWrite, Scope

MOD = "m"


def proc(name: str) -> Symbol:
    return Symbol(name, SymbolKind.PROCEDURE, Origin.GLOBAL, target=f"{MOD}::{name}")


GLOBALS = [
    gvar("r8"),
    gvar("col_pp"),
    gvar("veg_pp"),
    gvar("nstep"),
    proc("child"),
    proc("myfunc"),
    proc("parent"),
    proc("gchild"),
    proc("gfunc"),
    proc("mid"),
    proc("root"),
    Symbol("ext_sub", SymbolKind.EXTERNAL, Origin.EXTERNAL),
]


class StubSub:
    """The parts of Subroutine that AccessMapper uses."""

    def __init__(self, txt: str, library: bool = False, host=None, tree_index=None):
        self.record = walk(
            txt,
            globals=GLOBALS,
            host=host.record if host else None,
            tree_index=tree_index,
        )
        self.name = self.record.name
        self.module = MOD
        self.id = f"{MOD}::{self.name}"
        self.host_id = host.id if host else None
        self.library = library
        self.sub_call_desc = {}
        self.sub_lines = []
        self.record_access = None

    def walk_syntax_tree(self, scopes):
        return self.record


CHILD = """
subroutine child(a, b, c)
  implicit none
  real(r8), intent(in) :: a
  type(foo_type), intent(inout) :: b
  real(r8), intent(out) :: c
  b%f = a
end subroutine child
"""

MYFUNC = """
function myfunc(y) result(z)
  implicit none
  real(r8), intent(in) :: y
  real(r8) :: z
  z = 2*y
end function myfunc
"""

PARENT = """
subroutine parent(x, n)
  implicit none
  real(r8), intent(inout) :: x
  integer, intent(in) :: n
  real(r8) :: loc
  type(foo_type) :: t
  associate(snl => col_pp%snl)
  x = x + snl(n)
  call child(col_pp%z(n), t, c=loc)
  loc = myfunc(x)
  call ext_sub(x)
  if (n > 0) call child(x, t, loc)
  end associate
end subroutine parent
"""


def rws(*pairs):
    return [ReadWrite(s, ln, None) for s, ln in pairs]


def mapped():
    subs = {s.id: s for s in (StubSub(CHILD), StubSub(MYFUNC), StubSub(PARENT))}
    mapper = AccessMapper(subs)
    return mapper, subs


def test_leaf_maps_and_intent_fallback():
    mapper, subs = mapped()
    child = mapper.maps(subs["m::child"])
    assert child.args == {"a": rws(("r", 6)), "b%f": rws(("w", 6))}
    # unused intent(out) dummy: status from its intent
    assert child.dummy_status("c") == {"c": "w"}
    assert child.dummy_status("b") == {"b%f": "w"}
    assert child.elmtype == {} and child.locals == {}


def test_function_result_is_local():
    mapper, subs = mapped()
    f = mapper.maps(subs["m::myfunc"])
    assert f.args == {"y": rws(("r", 5))}
    assert f.locals == {"z": rws(("w", 5))}


def test_parent_maps():
    mapper, subs = mapped()
    p = mapper.maps(subs["m::parent"])
    assert p is subs["m::parent"].record_access
    # associate names are expanded; same-line read+write merge to "rw";
    # unknown (external) subroutine actuals are conservatively "rw"
    assert p.args == {
        "x": rws(("rw", 8), ("r", 10), ("rw", 11), ("r", 12)),
        "n": rws(("r", 8), ("r", 9), ("r", 12)),
    }
    assert p.elmtype == {
        "col_pp%snl": rws(("r", 8)),
        "col_pp%z": rws(("r", 9)),
    }
    # derived-type actual takes the dummy's component keys; keyword arg c=loc
    assert p.locals == {
        "t%f": rws(("w", 9), ("w", 12)),
        "loc": rws(("w", 9), ("w", 10), ("w", 12)),
    }


def test_unwalked_callee_is_walked_on_demand():
    mapper, subs = mapped()
    mapper.maps(subs["m::parent"])
    assert subs["m::child"].record_access is not None


def test_pointer_accesses_reach_targets():
    txt = """
subroutine ptrs(n)
  implicit none
  integer, intent(in) :: n
  real(r8), pointer :: ci(:)
  ci => col_pp%z
  ci(n) = 1._r8
  call child(ci(n), col_pp, ci(1))
end subroutine ptrs
"""
    mapper, subs = mapped()
    sub = StubSub(txt)
    subs[sub.id] = sub
    p = mapper.maps(sub)
    # the association itself doesn't access the target
    assert p.elmtype == {
        "col_pp%z": rws(("w", 6), ("rw", 7)),
        "col_pp%f": rws(("w", 7)),
    }
    assert p.locals == {"ci": rws(("w", 5), ("w", 6), ("rw", 7))}


def test_recursive_call_is_conservative():
    rec_txt = """
recursive subroutine parent(n)
  implicit none
  integer, intent(inout) :: n
  call parent(n)
end subroutine parent
"""
    sub = StubSub(rec_txt)
    p = AccessMapper({sub.id: sub}).maps(sub)
    assert p.args == {"n": rws(("rw", 4))}


def test_callee_globals_merge_at_call_line():
    gchild = """
subroutine gchild(n)
  implicit none
  integer, intent(in) :: n
  col_pp%snl(n) = col_pp%dz(n)
  nstep = n
end subroutine gchild
"""
    gfunc = """
function gfunc(p) result(v)
  implicit none
  integer, intent(in) :: p
  integer :: v
  v = veg_pp%itype(p)
end function gfunc
"""
    mid = """
subroutine mid(n)
  implicit none
  integer, intent(in) :: n
  integer :: k
  k = col_pp%snl(n)
  call gchild(n)
  k = gfunc(k)
end subroutine mid
"""
    top = """
subroutine top(n)
  implicit none
  integer, intent(in) :: n
  call mid(n)
end subroutine top
"""
    subs = {s.id: s for s in map(StubSub, (gchild, gfunc, mid, top))}
    mapper = AccessMapper(subs)
    m = mapper.maps(subs["m::mid"])
    # direct accesses stay separate from accesses made by callees
    assert m.elmtype == {"col_pp%snl": rws(("r", 5))}
    # each callee's overall status lands on the call line (functions too)
    assert m.child_elmtype == {
        "col_pp%snl": rws(("w", 6)),
        "col_pp%dz": rws(("r", 6)),
        "veg_pp%itype": rws(("r", 7)),
    }
    assert m.child_globals == {"nstep": rws(("w", 6))}
    assert m.all_elmtype()["col_pp%snl"] == rws(("r", 5), ("w", 6))

    # transitive: grandchildren reach the top through mid's call line
    t = mapper.maps(subs["m::top"])
    assert t.elmtype == {}
    assert t.child_elmtype == {
        "col_pp%snl": rws(("rw", 4)),
        "col_pp%dz": rws(("r", 4)),
        "veg_pp%itype": rws(("r", 4)),
    }
    assert t.all_globals() == {"nstep": rws(("w", 4))}


HOSTR = """
subroutine hostr(x)
  implicit none
  real(r8), intent(inout) :: x
  real(r8) :: h, shadow
  h = 0._r8
  call inner()
contains
  subroutine inner()
    implicit none
    real(r8) :: shadow
    shadow = h + x
    x = col_pp%z(1)
    call sibling()
  end subroutine inner
  subroutine sibling()
    implicit none
    shadow = 1._r8
  end subroutine sibling
end subroutine hostr
"""


def test_host_accesses_bind_at_call_line():
    host = StubSub(HOSTR)
    inner = StubSub(HOSTR, host=host, tree_index=0)
    sibling = StubSub(HOSTR, host=host, tree_index=1)
    subs = {s.id: s for s in (host, inner, sibling)}
    mapper = AccessMapper(subs)

    i = mapper.maps(inner)
    # sibling writes the host's `shadow`, not inner's local of the same name
    assert i.locals == {"shadow": rws(("w", 11))}
    assert i.host == {
        "h": rws(("r", 11)),
        "x": rws(("r", 11), ("w", 12)),
        "shadow": rws(("w", 13)),
    }

    h = mapper.maps(host)
    # host variables used by an internal routine bind like arguments
    assert h.locals == {"h": rws(("w", 5), ("r", 6)), "shadow": rws(("w", 6))}
    assert h.args == {"x": rws(("rw", 6))}
    assert h.child_elmtype == {"col_pp%z": rws(("r", 6))}
    assert h.host == {}


def test_root_globals_at_driver_call_site():
    root = """
subroutine root(cs, n)
  implicit none
  type(foo_type), intent(inout) :: cs
  integer, intent(in) :: n
  cs%f(n) = col_pp%dz(n)
end subroutine root
"""
    drv = """
subroutine drv()
  implicit none
  integer :: nc
  nc = 1
  call root(veg_pp, nstep)
  col_pp%snl(nc) = 0
end subroutine drv
"""
    subs = {s.id: s for s in map(StubSub, (root, drv))}
    mapper = AccessMapper(subs)
    d = mapper.maps(subs["m::drv"])
    calls = {c.ln for c in subs["m::drv"].record.calls if c.name == "root"}
    # the global passed to each dummy takes the dummy's overall status at the
    # call line, along with the root's own global accesses
    assert callsite_globals(d, calls) == {
        "veg_pp%f": rws(("w", 5)),
        "col_pp%dz": rws(("r", 5)),
        "nstep": rws(("r", 5)),
    }


def test_call_bindings_are_recorded():
    mapper, subs = mapped()
    p = mapper.maps(subs["m::parent"])
    b = lambda callee, ln, n, d, a, o: ArgBinding("m::parent", f"m::{callee}", ln, n, d, a, o)
    # only analyzable callees bind (not ext_sub); keyword actuals bind by name
    assert p.bindings == [
        b("child", 9, 0, "a", "col_pp%z", Origin.GLOBAL),
        b("child", 9, 1, "b", "t", Origin.LOCAL),
        b("child", 9, 2, "c", "loc", Origin.LOCAL),
        b("myfunc", 10, 0, "y", "x", Origin.DUMMY),
        b("child", 12, 0, "a", "x", Origin.DUMMY),
        b("child", 12, 1, "b", "t", Origin.LOCAL),
        b("child", 12, 2, "c", "loc", Origin.LOCAL),
    ]


def test_pointer_actual_binds_its_targets():
    txt = """
subroutine ptrs(n)
  implicit none
  integer, intent(in) :: n
  real(r8), pointer :: ci(:)
  ci => col_pp%z
  call child(ci(n), col_pp, n)
end subroutine ptrs
"""
    mapper, subs = mapped()
    sub = StubSub(txt)
    subs[sub.id] = sub
    p = mapper.maps(sub)
    assert [(x.dummy, x.actual, x.origin) for x in p.bindings] == [
        ("a", "col_pp%z", Origin.GLOBAL),
        ("b", "col_pp", Origin.GLOBAL),
        ("c", "n", Origin.DUMMY),
    ]


def test_elmtype_view_truncates_binds_dummies_and_fans_out_pointers():
    maps = AccessMaps(
        dummies=["cs", "n"],
        elmtype={"col_pp%z": rws(("r", 5)), "col_pp%x%y": rws(("w", 5))},
        child_elmtype={"col_pp%z": rws(("w", 6)), "col_pp%x": rws(("r", 5))},
        args={"cs%f": rws(("w", 4)), "cs": rws(("r", 7)), "n": rws(("r", 4))},
    )
    view = elmtype_view(
        maps,
        dummy_actuals={"cs": {"veg_pp"}, "n": {"nstep"}},
        pointer_components={"veg_pp%f": ["veg_pp%g"]},
    )
    assert view == {
        # direct and callee accesses; deeper paths at `inst%field` depth
        "col_pp%z": rws(("r", 5), ("w", 6)),
        "col_pp%x": rws(("rw", 5)),
        # dummy components land on the global bound to the dummy (same lines)
        "veg_pp%f": rws(("w", 4)),
        # a pointer component also accesses its targets
        "veg_pp%g": rws(("w", 4)),
    }
    summary = access_summary(view)
    assert {k: rw.status for k, rw in summary.items()} == {
        "col_pp%z": "rw",
        "col_pp%x": "rw",
        "veg_pp%f": "w",
        "veg_pp%g": "w",
    }
    assert all(rw.ln == -1 for rw in summary.values())


def test_dummy_actuals_from_driver_bindings():
    root = """
subroutine root(cs, n, k)
  implicit none
  type(foo_type), intent(inout) :: cs
  integer, intent(in) :: n, k
  cs%f(n) = k
end subroutine root
"""
    drv = """
subroutine drv()
  implicit none
  integer :: nc
  nc = 1
  call root(veg_pp, nstep, nc)
  call root(col_pp, nstep, nc)
end subroutine drv
"""
    subs = {s.id: s for s in map(StubSub, (root, drv))}
    d = AccessMapper(subs).maps(subs["m::drv"])
    # only globals are bound (nc is elm_drv's local)
    assert dummy_actuals(d.bindings, "m::root") == {
        "cs": {"veg_pp", "col_pp"},
        "n": {"nstep"},
    }
    assert dummy_actuals(d.bindings, "m::other") == {}


def test_single_instance_actuals():
    arg_types = {"cs": "foo_type", "b": "bar_type", "n": "integer"}
    instances = {"foo_type": ["veg_pp"], "bar_type": ["b1", "b2"]}
    # ambiguous types (several instances) stay unbound
    assert single_instance_actuals(arg_types, instances) == {"cs": {"veg_pp"}}


def test_propagated_access_from_bindings():
    mapper, subs = mapped()
    mapper.maps(subs["m::parent"])
    prop = propagated_access(subs)
    child = subs["m::child"].record_access
    got = sorted(
        (k, p.tag.caller, p.tag.call_ln, p.dummy, p.scope.name, p.binding.argn)
        for k, ps in prop["m::child"].items()
        for p in ps
    )
    # one entry per (call site, accessed dummy key); unused dummy c has no lines
    assert got == [
        ("col_pp%z", "m::parent", 9, "a", "ELMTYPE", 0),
        ("t%f", "m::parent", 9, "b", "LOCAL", 1),
        ("t%f", "m::parent", 12, "b", "LOCAL", 1),
        ("x", "m::parent", 12, "a", "ARG", 0),
    ]
    p = prop["m::child"]["col_pp%z"][0]
    assert p.rw_statuses == child.args["a"]
    assert (p.binding.var_name, p.binding.member_path, p.binding.nested_level) == ("col_pp", "z", 0)
    assert [(k, p.tag.call_ln) for k, ps in prop["m::myfunc"].items() for p in ps] == [("x", 10)]
