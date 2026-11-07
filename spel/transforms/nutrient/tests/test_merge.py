import shutil
import subprocess
from types import SimpleNamespace

from spel.transforms.nutrient.common import Family, Variant, match_fields
from spel.transforms.nutrient.merge import Merger

SRC = """\
module m
  implicit none
  integer, parameter :: r8 = selected_real_kind(12)
  type, public :: veg_nutrient_flux
     real(r8), pointer :: leaf(:) => null()
  end type veg_nutrient_flux
  type, public :: veg_carbon_flux
     type(veg_nutrient_flux) :: core
     real(r8), pointer :: gpp(:) => null()
   contains
     procedure, public :: SetValues => c_setvalues
  end type veg_carbon_flux
  type, public :: veg_nitrogen_flux
     type(veg_nutrient_flux) :: core
   contains
     procedure, public :: SetValues => n_setvalues
  end type veg_nitrogen_flux
contains
  subroutine c_setvalues(this, num_patch, filter_patch, value_patch)
    class(veg_carbon_flux) :: this
    integer, intent(in) :: num_patch
    integer, intent(in) :: filter_patch(:)
    real(r8), intent(in) :: value_patch
    integer :: fi, i
    do fi = 1, num_patch
       i = filter_patch(fi)
       this%core%leaf(i) = value_patch
       this%gpp(i) = value_patch
    end do
  end subroutine c_setvalues
  subroutine n_setvalues(this, num_patch, filter_patch, value_patch)
    class(veg_nitrogen_flux) :: this
    integer, intent(in) :: num_patch
    integer, intent(in) :: filter_patch(:)
    real(r8), intent(in) :: value_patch
    integer :: fi, i
    ! the shared field
    do fi = 1, num_patch
       i = filter_patch(fi)
       this%core%leaf(i) = value_patch
    end do
  end subroutine n_setvalues
  subroutine caller(nf)
    type(veg_nitrogen_flux) :: nf
    integer :: f(1)
    call nf%SetValues(num_patch=1, filter_patch=f, value_patch=0._r8)
  end subroutine caller
end module m
"""


def setup(tmp_path, src=SRC):
    f = tmp_path / "m.F90"
    f.write_text(src)
    lines = src.split("\n")

    def sub(name):
        s = next(i for i, l in enumerate(lines) if l.strip().startswith(f"subroutine {name}"))
        e = next(i for i, l in enumerate(lines) if l.strip().startswith(f"end subroutine {name}"))
        return SimpleNamespace(filepath=f, startline=s, endline=e)

    v = lambda n: Variant(n, "real", 1, "begp:endp", "pointer", 0)
    comp = lambda *ns: {n: SimpleNamespace() for n in ns}
    fam = Family("veg_nutrient_flux", {"c": "veg_carbon_flux", "n": "veg_nitrogen_flux"})
    fam.fields = match_fields({"c": {"gpp": v("gpp")}, "n": {}})
    td = {
        "veg_nutrient_flux": SimpleNamespace(type_name="veg_nutrient_flux", components=comp("leaf"), procedures={}),
        "veg_carbon_flux": SimpleNamespace(
            type_name="veg_carbon_flux", components=comp("core", "core%leaf", "gpp"),
            procedures={"setvalues": "m::c_setvalues"}, filepath=f, instances={},
        ),
        "veg_nitrogen_flux": SimpleNamespace(
            type_name="veg_nitrogen_flux", components=comp("core", "core%leaf"),
            procedures={"setvalues": "m::n_setvalues"}, filepath=f, instances={},
        ),
    }
    sd = {f"m::{n}": sub(n) for n in ("c_setvalues", "n_setvalues") if f"subroutine {n}" in src}
    return f, fam, SimpleNamespace(type_dict=td, subroutine_dict=sd)


def test_merge_setvalues(tmp_path):
    f, fam, fut = setup(tmp_path)
    m = Merger([fam], fut, bindings=["setvalues"])
    m.build()
    assert not m.refused
    (res,) = m.results
    assert res.moved == {"c": 1, "n": 1}
    out = "\n".join(m.apply()[f])
    # base routine: a plain module procedure on the core component
    assert "subroutine veg_nutrient_flux_setvalues(this, num_patch, filter_patch, value_patch)" in out
    assert "type(veg_nutrient_flux), intent(inout) :: this" in out
    assert "this%leaf(i) = value_patch" in out
    # concrete routines keep their names and dummies and call it with their core
    assert "call veg_nutrient_flux_setvalues(this%core, num_patch, filter_patch, value_patch)" in out
    assert out.count("call veg_nutrient_flux_setvalues(this%core, num_patch, filter_patch, value_patch)") == 2
    assert "this%gpp(i) = value_patch" in out and "this%core%leaf" not in out
    # nitrogen has nothing of its own left: no loop, no comment about it
    n_body = out[out.index("subroutine n_setvalues") : out.index("end subroutine n_setvalues")]
    assert "do fi" not in n_body and "shared field" not in n_body
    # the keyword call is untouched
    assert "call nf%SetValues(num_patch=1, filter_patch=f, value_patch=0._r8)" in out
    if shutil.which("gfortran"):
        (tmp_path / "out.F90").write_text(out)
        r = subprocess.run(
            ["gfortran", "-c", "-fsyntax-only", str(tmp_path / "out.F90")],
            cwd=tmp_path, capture_output=True, text=True,
        )
        assert r.returncode == 0, r.stderr


FATES = """\
module m
  implicit none
  integer, parameter :: r8 = selected_real_kind(12)
  logical :: use_fates = .false.
  type, public :: veg_nutrient_flux
     integer :: species
     real(r8), pointer :: leaf(:) => null()
  end type veg_nutrient_flux
  type, public :: veg_carbon_flux
     type(veg_nutrient_flux) :: core
     real(r8), pointer :: gpp(:) => null()
  end type veg_carbon_flux
  type, public :: veg_nitrogen_flux
     type(veg_nutrient_flux) :: core
  end type veg_nitrogen_flux
contains
  pure logical function is_carbon(species)
    integer, intent(in) :: species
    is_carbon = species <= 3
  end function is_carbon
  subroutine c_setvalues(this, num_patch, filter_patch, value_patch)
    class(veg_carbon_flux) :: this
    integer, intent(in) :: num_patch
    integer, intent(in) :: filter_patch(:)
    real(r8), intent(in) :: value_patch
    integer :: fi, i
    if (.not. use_fates) then
       do fi = 1, num_patch
          i = filter_patch(fi)
          this%core%leaf(i) = value_patch
          this%gpp(i) = value_patch
       end do
    end if
  end subroutine c_setvalues
  subroutine n_setvalues(this, num_patch, filter_patch, value_patch)
    class(veg_nitrogen_flux) :: this
    integer, intent(in) :: num_patch
    integer, intent(in) :: filter_patch(:)
    real(r8), intent(in) :: value_patch
    integer :: fi, i
    do fi = 1, num_patch
       i = filter_patch(fi)
       this%core%leaf(i) = value_patch
    end do
  end subroutine n_setvalues
end module m
"""


def test_merge_keeps_carbon_fates_guard(tmp_path):
    f, fam, fut = setup(tmp_path, FATES)
    fut.type_dict["veg_nutrient_flux"].components = {"species": None, "leaf": None}
    m = Merger([fam], fut, bindings=["setvalues"])
    m.build()
    (res,) = m.results
    assert res.moved == {"c": 1, "n": 1} and res.fates == 1
    out = "\n".join(m.apply()[f])
    base = out[out.index("subroutine veg_nutrient_flux_setvalues") : out.index("end subroutine veg_nutrient_flux_setvalues")]
    assert "if (.not. (is_carbon(this%species) .and. use_fates)) then" in base
    assert "this%leaf(i) = value_patch" in base
    c_body = out[out.index("subroutine c_setvalues") : out.index("end subroutine c_setvalues")]
    assert "if (.not. use_fates) then" in c_body and "this%gpp(i)" in c_body and "leaf" not in c_body
    if shutil.which("gfortran"):
        (tmp_path / "out.F90").write_text(out)
        r = subprocess.run(
            ["gfortran", "-c", "-fsyntax-only", str(tmp_path / "out.F90")],
            cwd=tmp_path, capture_output=True, text=True,
        )
        assert r.returncode == 0, r.stderr


INIT = """\
module m
  implicit none
  integer, parameter :: r8 = selected_real_kind(12)
  real(r8), parameter :: spval = 1.e36_r8
  type, public :: veg_nutrient_flux
     integer :: species
     real(r8), pointer :: leaf(:) => null()
     real(r8), pointer :: root(:) => null()
  end type veg_nutrient_flux
  type, public :: veg_carbon_flux
     type(veg_nutrient_flux) :: core
     real(r8), pointer :: gpp(:) => null()
  end type veg_carbon_flux
  type, public :: veg_nitrogen_flux
     type(veg_nutrient_flux) :: core
  end type veg_nitrogen_flux
contains
  subroutine c_init(this, begp, endp, carbon_type)
    class(veg_carbon_flux) :: this
    integer, intent(in) :: begp, endp
    character(len=3), intent(in) :: carbon_type
    this%core%species = 1
    allocate(this%core%leaf(begp:endp)) ; this%core%leaf(:) = spval
    allocate(this%gpp(begp:endp))
    if (carbon_type == 'c13') then
       this%core%root(:) = 0._r8
    end if
    call touch(this%gpp)
    allocate(this%core%root(begp:endp))
  end subroutine c_init
  subroutine n_init(this, begp, endp)
    class(veg_nitrogen_flux) :: this
    integer, intent(in) :: begp, endp
    this%core%species = 4
    allocate(this%core%leaf(begp:endp)) ; this%core%leaf(:) = spval
    this%core%root(:) = 0._r8
    allocate(this%core%root(begp:endp))
  end subroutine n_init
  subroutine touch(x)
    real(r8), pointer :: x(:)
  end subroutine touch
end module m
"""


def test_merge_init(tmp_path):
    f, fam, fut = setup(tmp_path, INIT)
    lines = INIT.split("\n")
    sub = lambda n: SimpleNamespace(
        filepath=f,
        startline=next(i for i, l in enumerate(lines) if l.strip().startswith(f"subroutine {n}")),
        endline=next(i for i, l in enumerate(lines) if l.strip().startswith(f"end subroutine {n}")),
    )
    fut.subroutine_dict = {"m::c_init": sub("c_init"), "m::n_init": sub("n_init")}
    td = fut.type_dict
    td["veg_nutrient_flux"].components = {"species": None, "leaf": None, "root": None}
    td["veg_carbon_flux"].procedures = {"init": "m::c_init"}
    td["veg_nitrogen_flux"].procedures = {"init": "m::n_init"}
    for t in ("veg_carbon_flux", "veg_nitrogen_flux"):
        td[t].components.update({"core%leaf": None, "core%root": None, "core%species": None})
    m = Merger([fam], fut, bindings=["init"])
    m.build()
    assert not m.refused, m.refused
    (res,) = m.results
    # the leaf allocation moves; root's allocation follows a kept statement on root
    # (carbon's isotope-guarded one, nitrogen's own) and a call: it stays
    assert res.moved == {"c": 2, "n": 2}
    out = "\n".join(m.apply()[f])
    base = out[out.index("subroutine veg_nutrient_flux_init") : out.index("end subroutine veg_nutrient_flux_init")]
    assert base.startswith("subroutine veg_nutrient_flux_init(this, begp, endp)")
    assert "carbon_type" not in base and "type(veg_nutrient_flux), intent(inout) :: this" in base
    assert "allocate(this%leaf(begp:endp)) ; this%leaf(:) = spval" in base and "root" not in base
    c_body = out[out.index("subroutine c_init") : out.index("end subroutine c_init")]
    # the base call comes after the species assignment, with the dummies it takes
    assert c_body.index("this%core%species = 1") < c_body.index("call veg_nutrient_flux_init(this%core, begp, endp)")
    assert "if (carbon_type == 'c13') then" in c_body and "allocate(this%core%root(begp:endp))" in c_body
    if shutil.which("gfortran"):
        (tmp_path / "out.F90").write_text(out)
        r = subprocess.run(
            ["gfortran", "-c", "-fsyntax-only", str(tmp_path / "out.F90")],
            cwd=tmp_path, capture_output=True, text=True,
        )
        assert r.returncode == 0, r.stderr


def test_merge_moves_directives_with_their_loop(tmp_path):
    src = SRC.replace(
        "    do fi = 1, num_patch\n       i = filter_patch(fi)\n       this%core%leaf(i) = value_patch\n       this%gpp(i)",
        "    !$acc parallel loop\n    do fi = 1, num_patch\n       i = filter_patch(fi)\n       this%core%leaf(i) = value_patch\n       this%gpp(i)",
    ).replace(
        "    ! the shared field\n    do fi = 1, num_patch\n       i = filter_patch(fi)\n       this%core%leaf(i) = value_patch\n    end do",
        "    ! the shared field\n    !$acc parallel loop\n    do fi = 1, num_patch\n       i = filter_patch(fi)\n       this%core%leaf(i) = value_patch\n    end do",
    )
    assert src.count("!$acc parallel loop") == 2
    f, fam, fut = setup(tmp_path, src)
    m = Merger([fam], fut, bindings=["setvalues"])
    m.build()
    out = "\n".join(m.apply()[f])
    base = out[out.index("subroutine veg_nutrient_flux_setvalues") : out.index("end subroutine veg_nutrient_flux_setvalues")]
    c_body = out[out.index("subroutine c_setvalues") : out.index("end subroutine c_setvalues")]
    n_body = out[out.index("subroutine n_setvalues") : out.index("end subroutine n_setvalues")]
    # the loop is in the base and in carbon (its gpp): each copy keeps its directive, before the loop
    assert base.count("!$acc parallel loop\n    do fi") == 1
    assert c_body.count("!$acc parallel loop\n    do fi") == 1
    assert c_body.index("call veg_nutrient_flux_setvalues") < c_body.index("!$acc parallel loop")
    # nitrogen has no loop left, and no directive
    assert "!$acc" not in n_body


REGISTER = """\
module m
  implicit none
  integer, parameter :: r8 = selected_real_kind(12)
  type, public :: veg_nutrient_flux
     integer :: species
     real(r8), pointer :: leaf(:,:) => null()
  end type veg_nutrient_flux
  type, public :: veg_carbon_flux
     type(veg_nutrient_flux) :: core
  end type veg_carbon_flux
  type, public :: veg_nitrogen_flux
     type(veg_nutrient_flux) :: core
  end type veg_nitrogen_flux
contains
  subroutine c_init(this, begp, endp)
    class(veg_carbon_flux) :: this
    integer, intent(in) :: begp, endp
    integer :: l
    real(r8), pointer :: ptr(:)
    character(len=8) :: fname
    allocate(this%core%leaf(begp:endp, 2))
    do l = 1, 2
       ptr => this%core%leaf(:,l)
       fname = 'LEAF'
       call register(fname, ptr)
    end do
  end subroutine c_init
  subroutine n_init(this, begp, endp)
    class(veg_nitrogen_flux) :: this
    integer, intent(in) :: begp, endp
    integer :: l
    real(r8), pointer :: ptr(:)
    character(len=8) :: fname
    allocate(this%core%leaf(begp:endp, 2))
    do l = 1, 2
       ptr => this%core%leaf(:,l)
       fname = 'LEAF'
       call register(fname, ptr)
    end do
  end subroutine n_init
  subroutine register(name, p)
    character(len=*) :: name
    real(r8), pointer :: p(:)
  end subroutine register
end module m
"""


def test_merge_keeps_registration_loops_whole(tmp_path):
    f, fam, fut = setup(tmp_path, REGISTER)
    lines = REGISTER.split("\n")
    sub = lambda n: SimpleNamespace(
        filepath=f,
        startline=next(i for i, l in enumerate(lines) if l.strip().startswith(f"subroutine {n}")),
        endline=next(i for i, l in enumerate(lines) if l.strip().startswith(f"end subroutine {n}")),
    )
    fut.subroutine_dict = {"m::c_init": sub("c_init"), "m::n_init": sub("n_init")}
    td = fut.type_dict
    td["veg_nutrient_flux"].components = {"species": None, "leaf": None}
    td["veg_carbon_flux"].procedures = {"init": "m::c_init"}
    td["veg_nitrogen_flux"].procedures = {"init": "m::n_init"}
    td["veg_carbon_flux"].components = {"core": None, "core%leaf": None, "core%species": None}
    td["veg_nitrogen_flux"].components = {"core": None, "core%leaf": None, "core%species": None}
    fam.fields = []
    m = Merger([fam], fut, bindings=["init"])
    m.build()
    assert not m.refused, m.refused
    (res,) = m.results
    # the allocation moves; the pointer assignment writes a local the call reads: it stays
    assert res.moved == {"c": 1, "n": 1}
    out = "\n".join(m.apply()[f])
    base = out[out.index("subroutine veg_nutrient_flux_init") : out.index("end subroutine veg_nutrient_flux_init")]
    assert "allocate(this%leaf(begp:endp, 2))" in base and "register" not in base and "do l" not in base
    for name in ("c_init", "n_init"):
        body = out[out.index(f"subroutine {name}") : out.index(f"end subroutine {name}")]
        assert body.count("call register(fname, ptr)") == 1 and "ptr => this%core%leaf(:,l)" in body
    if shutil.which("gfortran"):
        (tmp_path / "out.F90").write_text(out)
        r = subprocess.run(
            ["gfortran", "-c", "-fsyntax-only", str(tmp_path / "out.F90")],
            cwd=tmp_path, capture_output=True, text=True,
        )
        assert r.returncode == 0, r.stderr


def test_merge_keeps_blocks_nothing_moved_out_of(tmp_path):
    src = REGISTER.replace(
        "    character(len=8) :: fname\n    allocate(this%core%leaf(begp:endp, 2))\n    do l = 1, 2\n       ptr => this%core%leaf(:,l)\n       fname = 'LEAF'",
        "    character(len=8) :: fname, suffix\n    allocate(this%core%leaf(begp:endp, 2))\n    if (endp > 1) then\n       suffix = '_vr'\n    end if\n    do l = 1, 2\n       ptr => this%core%leaf(:,l)\n       fname = 'LEAF'//suffix",
    )
    assert src.count("suffix = '_vr'") == 2
    f, fam, fut = setup(tmp_path, src)
    lines = src.split("\n")
    sub = lambda n: SimpleNamespace(
        filepath=f,
        startline=next(i for i, l in enumerate(lines) if l.strip().startswith(f"subroutine {n}")),
        endline=next(i for i, l in enumerate(lines) if l.strip().startswith(f"end subroutine {n}")),
    )
    fut.subroutine_dict = {"m::c_init": sub("c_init"), "m::n_init": sub("n_init")}
    td = fut.type_dict
    td["veg_nutrient_flux"].components = {"species": None, "leaf": None}
    td["veg_carbon_flux"].procedures = {"init": "m::c_init"}
    td["veg_nitrogen_flux"].procedures = {"init": "m::n_init"}
    td["veg_carbon_flux"].components = {"core": None, "core%leaf": None, "core%species": None}
    td["veg_nitrogen_flux"].components = {"core": None, "core%leaf": None, "core%species": None}
    fam.fields = []
    m = Merger([fam], fut, bindings=["init"])
    m.build()
    out = "\n".join(m.apply()[f])
    for name in ("c_init", "n_init"):
        body = out[out.index(f"subroutine {name}") : out.index(f"end subroutine {name}")]
        assert "if (endp > 1) then\n       suffix = '_vr'\n    end if" in body


def test_merge_keeps_statements_reading_locals_set_by_kept_code(tmp_path):
    src = SRC.replace(
        "    integer :: fi, i\n    do fi = 1, num_patch\n       i = filter_patch(fi)\n       this%core%leaf(i) = value_patch",
        "    integer :: fi, i\n    real(r8) :: v\n    call pick(v)\n    do fi = 1, num_patch\n       i = filter_patch(fi)\n       this%core%leaf(i) = v",
    ).replace(
        "    integer :: fi, i\n    ! the shared field\n    do fi = 1, num_patch\n       i = filter_patch(fi)\n       this%core%leaf(i) = value_patch",
        "    integer :: fi, i\n    real(r8) :: v\n    call pick(v)\n    ! the shared field\n    do fi = 1, num_patch\n       i = filter_patch(fi)\n       this%core%leaf(i) = v",
    )
    assert src.count("call pick(v)") == 2
    f, fam, fut = setup(tmp_path, src)
    m = Merger([fam], fut, bindings=["setvalues"])
    m.build()
    # `v` is set by a call the concrete routines keep: the assignment can't run first in a base routine
    assert not m.results


def test_merge_rebuilds_associate_for_the_base(tmp_path):
    src = SRC.replace(
        "    integer :: fi, i\n    do fi = 1, num_patch\n       i = filter_patch(fi)\n       this%core%leaf(i) = value_patch\n       this%gpp(i) = value_patch\n    end do",
        "    integer :: fi, i\n    associate(lf => this%core%leaf, g => this%gpp)\n    do fi = 1, num_patch\n       i = filter_patch(fi)\n       lf(i) = value_patch\n       g(i) = value_patch\n    end do\n    end associate",
    ).replace(
        "    ! the shared field\n    do fi = 1, num_patch\n       i = filter_patch(fi)\n       this%core%leaf(i) = value_patch\n    end do",
        "    ! the shared field\n    associate(lf => this%core%leaf)\n    do fi = 1, num_patch\n       i = filter_patch(fi)\n       lf(i) = value_patch\n    end do\n    end associate",
    )
    assert src.count("associate(lf") == 2
    f, fam, fut = setup(tmp_path, src)
    m = Merger([fam], fut, bindings=["setvalues"])
    m.build()
    assert m.results, m.refused
    out = "\n".join(m.apply()[f])
    base = out[out.index("subroutine veg_nutrient_flux_setvalues") : out.index("end subroutine veg_nutrient_flux_setvalues")]
    # only the name the moved statement uses, pointed at the core argument
    assert "lf => this%leaf" in base and "g =>" not in base and "gpp" not in base
    if shutil.which("gfortran"):
        (tmp_path / "out.F90").write_text(out)
        r = subprocess.run(
            ["gfortran", "-c", "-fsyntax-only", str(tmp_path / "out.F90")],
            cwd=tmp_path, capture_output=True, text=True,
        )
        assert r.returncode == 0, r.stderr


SUMMARY = """\
module m
  implicit none
  integer, parameter :: r8 = selected_real_kind(12)
  type, public :: veg_nutrient_flux
     real(r8), pointer :: leaf(:) => null()
  end type veg_nutrient_flux
  type, public :: veg_carbon_flux
     type(veg_nutrient_flux) :: core
  end type veg_carbon_flux
  type, public :: veg_nitrogen_flux
     type(veg_nutrient_flux) :: core
  end type veg_nitrogen_flux
contains
  subroutine c_setvalues(this, num_soilp, filter_soilp, num_soilc, filter_soilc)
    class(veg_carbon_flux) :: this
    integer, intent(in) :: num_soilp, num_soilc
    integer, intent(in) :: filter_soilp(:), filter_soilc(:)
    integer :: fp, p
    do fp = 1, num_soilp
       p = filter_soilp(fp)
       this%core%leaf(p) = 0._r8
    end do
  end subroutine c_setvalues
  subroutine n_setvalues(this, num_soilc, filter_soilc, num_soilp, filter_soilp)
    class(veg_nitrogen_flux) :: this
    integer, intent(in) :: num_soilp, num_soilc
    integer, intent(in) :: filter_soilp(:), filter_soilc(:)
    integer :: fp, p
    do fp = 1, num_soilp
       p = filter_soilp(fp)
       this%core%leaf(p) = 0._r8
    end do
  end subroutine n_setvalues
end module m
"""


def test_merge_matches_dummies_by_name_not_position(tmp_path):
    # the C/N summaries take the patch and column filters in different orders:
    # nitrogen's call must pass its patch filter, not its 1st/2nd dummies
    f, fam, fut = setup(tmp_path, SUMMARY)
    m = Merger([fam], fut, bindings=["setvalues"])
    m.build()
    assert m.results, m.refused
    out = "\n".join(m.apply()[f])
    assert out.count("(this%core, num_soilp, filter_soilp)") == 2


def test_merge_matches_repeated_statements_by_occurrence(tmp_path):
    # carbon's history section sets leaf to spval again under an isotope guard; that
    # copy (the 2nd occurrence once x(begp:endp) is x(:)) mustn't keep the 1st, after
    # the allocation, out of the base routine
    src = INIT.replace(
        "    call touch(this%gpp)\n    allocate(this%core%root(begp:endp))\n  end subroutine c_init",
        "    call touch(this%gpp)\n    allocate(this%core%root(begp:endp))\n"
        "    if (carbon_type == 'c12') then\n       this%core%leaf(begp:endp) = spval\n    end if\n  end subroutine c_init",
    ).replace(
        "    allocate(this%core%root(begp:endp))\n  end subroutine n_init",
        "    allocate(this%core%root(begp:endp))\n    this%core%leaf(begp:endp) = spval\n  end subroutine n_init",
    )
    f, fam, fut = setup(tmp_path, src)
    lines = src.split("\n")
    sub = lambda n: SimpleNamespace(
        filepath=f,
        startline=next(i for i, l in enumerate(lines) if l.strip().startswith(f"subroutine {n}")),
        endline=next(i for i, l in enumerate(lines) if l.strip().startswith(f"end subroutine {n}")),
    )
    fut.subroutine_dict = {"m::c_init": sub("c_init"), "m::n_init": sub("n_init")}
    td = fut.type_dict
    td["veg_nutrient_flux"].components = {"species": None, "leaf": None, "root": None}
    td["veg_carbon_flux"].procedures = {"init": "m::c_init"}
    td["veg_nitrogen_flux"].procedures = {"init": "m::n_init"}
    for t in ("veg_carbon_flux", "veg_nitrogen_flux"):
        td[t].components.update({"core%leaf": None, "core%root": None, "core%species": None})
    m = Merger([fam], fut, bindings=["init"])
    m.build()
    (res,) = m.results
    assert res.moved == {"c": 2, "n": 2}
    out = "\n".join(m.apply()[f])
    base = out[out.index("subroutine veg_nutrient_flux_init") : out.index("end subroutine veg_nutrient_flux_init")]
    assert "allocate(this%leaf(begp:endp)) ; this%leaf(:) = spval" in base
    assert "this%leaf(begp:endp) = spval" not in base
    for name in ("c_init", "n_init"):
        body = out[out.index(f"subroutine {name}") : out.index(f"end subroutine {name}")]
        assert body.count("this%core%leaf(begp:endp) = spval") == 1


def test_file_edits_as_core_edits_give_the_same_text():
    from spel.transforms.core.edits import EditSet
    from spel.transforms.nutrient.merge import FileEdits
    from spel.transforms.nutrient.transforms import core_edits

    lines = [f"l{i}" for i in range(6)]
    fe = FileEdits(insert={1: ["i1"], 3: ["i3a", "i3b"]})
    fe.add_replace(1, 2, ["r1"])
    fe.add_replace(4, 4, [])
    assert EditSet().extend(core_edits(fe, lines)).apply(lines) == fe.apply(lines)
