"""spel restore: only the developer's edits reach the srcroot, not the harness edits of the copy."""
from spel.scripts.restore_files import (ORIG_DIR, apply_patch, changed_against_snapshot, developer_patch,
                                        snapshot_case)

SRC = """module foo
  use elm_time_manager, only : is_first_step
  implicit none
  real(r8), private :: x(10)
contains
  subroutine bar(n, y)
    integer :: n, i
    real(r8) :: y(:)
    do i = 1, n
       y(i) = 2._r8 * y(i)
    end do
    call endrun('done')
  end subroutine bar
end module foo
"""

COPY = """module foo
#include "unittest_defs.h"
  use spel_timemod, only : is_first_step
  implicit none
  real(r8), public :: x(10)
contains
  subroutine bar(n, y)
    integer :: n, i
    real(r8) :: y(:)
    do i = 1, n
       y(i) = 2._r8 * y(i)
    end do
    !#py call endrun('done')
  end subroutine bar
end module foo
"""


def test_restore_applies_only_the_edits(tmp_path):
    src = tmp_path / "srcroot" / "foo.F90"
    src.parent.mkdir()
    src.write_text(SRC)
    case = tmp_path / "case"
    case.mkdir()
    (case / "foo.F90").write_text(COPY)
    (case / "other.F90").write_text("module other\nend module other\n")
    snapshot_case(case)
    assert (case / ORIG_DIR / "foo.F90").read_text() == COPY
    assert changed_against_snapshot(case, set()) == []
    edited = COPY.replace("    do i = 1, n\n", "    !$acc parallel loop gang vector default(present)\n    do i = 1, n\n")
    (case / "foo.F90").write_text(edited)
    assert [p.name for p in changed_against_snapshot(case, set())] == ["foo.F90"]
    diff = developer_patch(case / ORIG_DIR / "foo.F90", case / "foo.F90", src)
    ok, out = apply_patch(diff, src, dry_run=False)
    assert ok, out
    assert src.read_text() == SRC.replace("    do i = 1, n\n", "    !$acc parallel loop gang vector default(present)\n    do i = 1, n\n")


def test_conflicting_edit_is_refused(tmp_path):
    src = tmp_path / "foo.F90"
    src.write_text(SRC.replace("y(i) = 2._r8 * y(i)", "y(i) = 3._r8 * y(i)"))  # changed since `spel create`
    case = tmp_path / "case"
    case.mkdir()
    (case / "foo.F90").write_text(COPY)
    snapshot_case(case)
    (case / "foo.F90").write_text(COPY.replace("y(i) = 2._r8 * y(i)", "y(i) = 4._r8 * y(i)"))
    ok, _ = apply_patch(developer_patch(case / ORIG_DIR / "foo.F90", case / "foo.F90", src), src, dry_run=True)
    assert not ok
    assert "3._r8" in src.read_text()


def test_second_restore_applies_only_the_new_edits(tmp_path, monkeypatch):
    import json
    from spel.scripts import restore_files as rf

    src = tmp_path / "src" / "components/elm/src/main" / "foo.F90"
    src.parent.mkdir(parents=True)
    src.write_text(SRC)
    case = tmp_path / "cases" / "c1"
    case.mkdir(parents=True)
    (case / "foo.F90").write_text(COPY)
    (case / rf.CASE_META_FILENAME).write_text(json.dumps({"e3sm_srcroot": str(tmp_path / "src")}))
    snapshot_case(case)
    monkeypatch.setattr(rf, "unittests_dir", tmp_path / "cases")
    monkeypatch.setattr(rf, "source_files_exclude_set", lambda: set())
    monkeypatch.setattr(rf, "_resolve_module_filepath", lambda m, e, s: str(src))
    one = COPY.replace("2._r8 * y(i)", "3._r8 * y(i)")
    (case / "foo.F90").write_text(one)
    assert rf.restore_case("c1", yes=True)
    two = one.replace("    do i = 1, n\n", "    !$acc parallel loop\n    do i = 1, n\n")
    (case / "foo.F90").write_text(two)
    assert rf.restore_case("c1", yes=True)
    assert src.read_text() == SRC.replace("2._r8 * y(i)", "3._r8 * y(i)").replace("    do i = 1, n\n", "    !$acc parallel loop\n    do i = 1, n\n")
