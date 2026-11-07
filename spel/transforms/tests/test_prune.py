import shutil
import subprocess

import pytest

from types import SimpleNamespace

from spel.scripts.fortran_parser.spel_ast import TypeDef, VariableDecl
from spel.scripts.utilityFunctions import Variable
from spel.transforms.core import driver
from spel.transforms.core.transform import Context
from spel.transforms.prune import Prune, Pruner, classify, component_refs
from spel.transforms.split.modules import load_source


def type_dict(*paths):
    """What the analysis records for each type: its file and each component's declaration line."""
    out = {}
    for path in paths:
        for p in load_source(path).head:
            if isinstance(p.node, TypeDef):
                comps = {}
                for m in p.node.members.statements:
                    if isinstance(m, VariableDecl):
                        for e in m.entities:
                            n = e.token.literal.lower()
                            comps[n] = Variable("real", n, "", m.lineno, 1)
                out[p.node.name.lower()] = SimpleNamespace(filepath=path, components=comps)
    return out

TYPES = """\
module thing_mod
  implicit none
  integer, parameter :: r8 = selected_real_kind(12)
  real(r8), parameter :: spval = 1.e36_r8
  type, public :: thing_type
     real(r8), pointer :: used(:)    => null() ! read below
     real(r8), pointer :: dead(:)    => null() ! only allocated and reset
     real(r8), pointer :: written(:) => null() ! computed write: an access
     real(r8), pointer :: hist(:)    => null() ! history pointer: an access
     real(r8), pointer :: guarded(:) => null()
  end type thing_type
  type(thing_type), public, target :: thing
contains
  subroutine thing_init(this, n)
    type(thing_type) :: this
    integer, intent(in) :: n
    allocate(this%used(n)) ; this%used(:) = spval
    allocate(this%dead(n)) ; this%dead(:) = spval
    allocate(this%written(n), this%hist(n))
    allocate(this%guarded(n))
    this%hist(:) = 0._r8
  end subroutine thing_init
  subroutine thing_setvalues(this, n, value_x)
    type(thing_type) :: this
    integer, intent(in) :: n
    real(r8), intent(in) :: value_x
    integer :: i
    do i = 1, n
       this%dead(i) = value_x
       this%written(i) = this%used(i) * 2._r8
    end do
  end subroutine thing_setvalues
  subroutine thing_clean(this)
    type(thing_type) :: this
    deallocate(this%used)
    deallocate(this%dead)
    if (associated(this%guarded)) deallocate(this%guarded)
    this%guarded => null()
  end subroutine thing_clean
end module thing_mod
"""

USER = """\
module user_mod
  use thing_mod, only : thing, r8
  implicit none
contains
  subroutine report(x)
    real(r8), intent(out) :: x
    x = sum(thing%used)
    call hist_addfld(ptr=thing%hist)
  end subroutine report
  subroutine hist_addfld(ptr)
    real(r8), pointer :: ptr(:)
  end subroutine hist_addfld
end module user_mod
"""


def run_prune(root, *type_files):
    """`spel transform prune` on `root` (in place)."""
    ctx = Context(SimpleNamespace(type_dict=type_dict(*type_files)), srcroot=root)
    return driver.run(Prune(), ctx, Prune.Options())


@pytest.fixture
def pruned(tmp_path):
    src = tmp_path / "components/elm/src/data_types"
    src.mkdir(parents=True)
    (src / "thing_mod.F90").write_text(TYPES)
    (tmp_path / "components/elm/src/user_mod.F90").write_text(USER)
    p = Pruner(tmp_path, type_dict(src / "thing_mod.F90"))
    p.build()
    return tmp_path, p


def test_component_refs():
    assert component_refs("this%core%leaf(p) = veg_cf%psn(p) + a(i)%b") == ["core", "leaf", "psn", "b"]


def test_classify():
    names = {"x", "y"}
    assert classify("this%x(:) = spval", names) == ({"x": "delete"}, set())
    assert classify("this%x(p) = max(0._r8, spval)", names) == ({"x": "delete"}, set())
    assert classify("this%x(p) = this%y(p)", names) == ({}, {"x", "y"})
    assert classify("this%x(p) = f(p)", names) == ({}, {"x"})  # maybe a function call
    assert classify("allocate(this%x(n), this%y(n))", names) == ({"x": "object", "y": "object"}, set())
    assert classify("if (associated(this%x)) deallocate(this%x)", names) == ({"x": "delete"}, set())
    assert classify("p => this%x", names) == ({}, {"x"})
    assert classify("!$acc update device(this%x)", names)[0] == {}


def test_prune_deletes_only_unaccessed_members(pruned):
    root, p = pruned
    assert sorted(p.dead) == ["dead", "guarded"]
    assert set(p.kept) >= {"used", "written", "hist"}
    out = "\n".join(p.files[root / "components/elm/src/data_types/thing_mod.F90"])
    for gone in ("this%dead", ":: dead", "this%guarded", ":: guarded"):
        assert gone not in out
    # `;`-joined statements keep their other half; multi-object allocate keeps its objects
    assert "allocate(this%used(n)) ; this%used(:) = spval" in out
    assert "allocate(this%written(n), this%hist(n))" in out
    assert "this%written(i) = this%used(i) * 2._r8" in out


@pytest.mark.skipif(not shutil.which("gfortran"), reason="needs gfortran")
def test_pruned_code_compiles(tmp_path):
    src = tmp_path / "components/elm/src/data_types"
    src.mkdir(parents=True)
    (src / "thing_mod.F90").write_text(TYPES)
    (tmp_path / "components/elm/src/user_mod.F90").write_text(USER)
    res = run_prune(tmp_path, src / "thing_mod.F90")
    assert res.written == [src / "thing_mod.F90"]
    root = tmp_path
    for f in ("components/elm/src/data_types/thing_mod.F90", "components/elm/src/user_mod.F90"):
        r = subprocess.run(["gfortran", "-c", str(root / f)], cwd=root, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr


def test_prune_keeps_crlf_line_endings(tmp_path):
    src = tmp_path / "components/elm/src/data_types"
    src.mkdir(parents=True)
    (src / "thing_mod.F90").write_bytes(TYPES.replace("\n", "\r\n").encode())
    (tmp_path / "components/elm/src/user_mod.F90").write_text(USER)
    run_prune(tmp_path, src / "thing_mod.F90")
    data = (src / "thing_mod.F90").read_bytes()
    assert b"this%dead" not in data and data.count(b"\n") == data.count(b"\r\n")
