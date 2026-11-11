from types import SimpleNamespace

from spel.transforms.nutrient.common import Family, Variant, match_fields
from spel.transforms.nutrient.procedures import Normalizer, compare_procedure

SRC = """\
module m
contains
  subroutine c_init(this, begp, endp, carbon_type)
    class(c_t) :: this
    integer :: begp, endp
    character(len=3) :: carbon_type
    allocate(this%leafc(begp:endp)) ; this%leafc(begp:endp) = spval
    if (carbon_type == 'c13') then
       call hist_addfld1d (fname='C13_LEAFC', units='gC13/m^2', &
            long_name='C13 leaf C', ptr_patch=this%leafc) ! comment
    end if
    this%gpp(begp:endp) = 0._r8
  end subroutine c_init
  subroutine n_init(this, begp, endp)
    class(n_t) :: this
    integer :: begp, endp
    allocate(this%leafn(begp:endp)) ; this%leafn(:) = spval
    call hist_addfld1d (fname='LEAFN', units='gN/m^2', &
         long_name='leaf nitrogen', ptr_patch=this%leafn)
  end subroutine n_init
end module m
"""


def test_normalizer_strings_and_components():
    norm = Normalizer("n", {"leafn": "leaf"}, ["this", "begp", "endp"])
    body, fields = norm.code("call hist_addfld1d(fname='LEAFN', units='gN/m^2', long_name='leaf N', ptr_patch=this%leafn)")
    assert body == "callhist_addfld1d(fname='{leaf}',units='g#/m^2',long_name='leaf #',ptr_patch=this%{leaf})"
    assert fields == {"leaf"}


def test_compare_procedure(tmp_path):
    f = tmp_path / "m.F90"
    f.write_text(SRC)
    v = lambda n: Variant(n, "real", 1, "begp:endp", "pointer", 0)
    fam = Family("vegetation_nutrient_state", {"c": "c_t", "n": "n_t"})
    fam.fields = match_fields({"c": {"leafc": v("leafc"), "gpp": v("gpp")}, "n": {"leafn": v("leafn")}})
    subs = {
        "m::c_init": SimpleNamespace(filepath=f, startline=2, endline=12),
        "m::n_init": SimpleNamespace(filepath=f, startline=13, endline=19),
    }
    types = {
        "c_t": SimpleNamespace(procedures={"init": "m::c_init"}, components={}, filepath=f),
        "n_t": SimpleNamespace(procedures={"init": "m::n_init"}, components={}, filepath=f),
    }
    rep = compare_procedure(fam, "init", subs, types)
    kinds = {e: [s.kind for s in ss] for e, ss in rep.stmts.items()}
    # isotope prefix and carbon_type guard dropped; only long_name wording differs
    assert kinds["c"] == ["generic", "generic", "longname", "own"]
    assert kinds["n"] == ["generic", "generic", "longname"]


def test_normalizer_nested_components():
    norm = Normalizer("n", {"mineral%smin_vr": "mineral%smin_vr", "core%leaf": "core%leaf"}, ["this"])
    norm.aliases["sminn_vr"] = "mineral%smin_vr"
    body, fields = norm.code("this%core%leaf(c) = sminn_vr(c,j) + this%dribbler%amount(c)")
    assert body == "this%{core%leaf}(c)={mineral%smin_vr}(c,j)+this%dribbler%amount(c)"
    assert fields == {"core%leaf", "mineral%smin_vr", "?dribbler"}


def test_type_bound_call_on_this_is_own():
    norm = Normalizer("n", {"core%leaf": "core%leaf"}, ["this"])
    _, fields = norm.code("call this%SetValues(num_column=n, filter_column=f, value_column=0._r8)")
    assert fields == {"?setvalues"}  # not empty: compare_procedure classes it `own`, not `other`
