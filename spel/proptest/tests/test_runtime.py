"""ctypes access and the worker process, on a hand-written library with the shim's interface
(shim.py): no SPEL case needed, only gfortran."""

import json
import shutil
import subprocess

import numpy as np
import pytest

from spel.proptest.worker import FortranAbort, Worker

pytestmark = pytest.mark.skipif(shutil.which("gfortran") is None, reason="needs gfortran")

SOURCE = """
module SpelPyMod
   use iso_c_binding
   implicit none
   real(c_double), allocatable, target :: x(:), y(:), never(:), absent(:)
   integer(c_int), target :: count = 0
   logical, target :: wet(2)
contains
subroutine spel_py_init(path, n) bind(C, name='spel_py_init')
   integer(c_int), value, intent(in) :: n
   character(kind=c_char), intent(in) :: path(n)
   allocate(x(0:2), y(0:2), never(0:2))
   x = [1.0d0, 2.0d0, 3.0d0]
   y = 0.0d0
   never = 0.0d0
   wet = [.true., .false.]
end subroutine
subroutine spel_py_compute() bind(C, name='spel_py_compute')
   if (any(x < 0.0d0)) error stop "negative x"
   y = 2.0d0 * x
   if (wet(2)) y = y + 1.0d0
   count = count + 1
end subroutine
subroutine spel_py_trap(on) bind(C, name='spel_py_trap')
   integer(c_int), value, intent(in) :: on
end subroutine
function spel_py_true() result(t) bind(C, name='spel_py_true')
   integer(c_int) :: t
   t = transfer(.true., 0_c_int)
end function
function spel_py_view(id, addr, lb, ub) result(rank) bind(C, name='spel_py_view')
   integer(c_int), value, intent(in) :: id
   type(c_ptr), intent(out) :: addr
   integer(c_int), intent(out) :: lb(7), ub(7)
   integer(c_int) :: rank
   addr = c_null_ptr
   lb = 1
   ub = 0
   rank = -1
   select case (id)
   case (1)
      addr = c_loc(x(0)); rank = 1; lb(1) = lbound(x, 1); ub(1) = ubound(x, 1)
   case (2)
      addr = c_loc(y(0)); rank = 1; lb(1) = lbound(y, 1); ub(1) = ubound(y, 1)
   case (3)
      addr = c_loc(never(0)); rank = 1; lb(1) = lbound(never, 1); ub(1) = ubound(never, 1)
   case (4)
      addr = c_loc(count); rank = 0
   case (5)
      addr = c_loc(wet(1)); rank = 1; lb(1) = 1; ub(1) = 2
   case (6)
      if (allocated(absent)) rank = 1
   case default
      rank = -2
   end select
end function
end module SpelPyMod
"""

ENTRIES = [("x", "real", "input"), ("y", "real", "output"), ("never", "real", "output"), ("count", "integer", "inout"),
           ("wet", "logical", "input"), ("absent", "real", "output")]


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    d = tmp_path_factory.mktemp("lib")
    (d / "SpelPyMod.f90").write_text(SOURCE)
    res = subprocess.run(["gfortran", "-shared", "-fPIC", "-o", "libspelcase.so", "SpelPyMod.f90"], cwd=d,
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    manifest = {"entries": [{"id": k + 1, "name": n, "type": t, "kind": kind} for k, (n, t, kind) in enumerate(ENTRIES)]}
    (d / "manifest.json").write_text(json.dumps(manifest))
    return d / "libspelcase.so", d / "manifest.json"


def test_worker_runs_resets_and_survives_an_abort(library, tmp_path):
    lib, manifest = library
    logs = tmp_path / "logs"
    with Worker(lib, manifest, "unused/", log_dir=logs) as w:
        got = w.get(["x", "wet", "absent"])
        assert got["x"].lb == (0,) and got["x"].data.tolist() == [1.0, 2.0, 3.0]  # Fortran lower bound kept
        assert got["wet"].data.tolist() == [True, False] and got["absent"] is None  # logical as bool; unallocated

        out = w.run({"x": np.array([1.0, 1.5, 2.0])}, ["y", "never", "count"])
        assert out["y"].data.tolist() == [2.0, 3.0, 4.0]
        assert np.isnan(out["never"].data).all()  # an output the routine does not write stays NaN
        assert int(out["count"].data) == 1

        out = w.run({"wet": np.array([True, True])}, ["y", "count"])  # x is back at its initial value
        assert out["y"].data.tolist() == [3.0, 5.0, 7.0] and int(out["count"].data) == 1

        with pytest.raises(FortranAbort, match="negative x"):
            w.run({"x": np.array([-1.0, 0.0, 0.0])}, ["y"])
        assert w.run({}, ["y"])["y"].data.tolist() == [2.0, 4.0, 6.0] and w.starts == 2  # a fresh process

        with pytest.raises(RuntimeError, match="not exposed"):
            w.get(["nosuch"])
        log = w.log
    assert log.is_file() and log.parent == logs  # kept: the Fortran code ended a process

    with Worker(lib, manifest, "unused/", log_dir=logs) as w:
        w.run({}, ["y"])
        log = w.log
    assert not log.exists()  # removed after a clean stop


DIRECT = """
module SpelPyMod
   use iso_c_binding
   use spel_varMod, only: spel_var_desc, spel_var_allocate, SPEL_MAXRANK
   implicit none
   real(c_double), allocatable, target :: x(:), y(:, :)
   integer(c_int), target :: nsetup = 0, n_seen = 0
   type(spel_var_desc), save :: spel_vars(4)
contains
subroutine spel_py_init(path, n) bind(C, name='spel_py_init')
   integer(c_int), value, intent(in) :: n
   character(kind=c_char), intent(in) :: path(n)
   error stop "files init called"
end subroutine
subroutine spel_py_begin(path, n) bind(C, name='spel_py_begin')
   integer(c_int), value, intent(in) :: n
   character(kind=c_char), intent(in) :: path(n)
end subroutine
subroutine spel_py_set_bounds(id, name, nlen, rank, lb, ub, has_fill, fill_r8, fill_int) &
     bind(C, name='spel_py_set_bounds')
   integer(c_int), value, intent(in) :: id, nlen, rank, has_fill, fill_int
   character(kind=c_char), intent(in) :: name(nlen)
   integer(c_int), intent(in) :: lb(*), ub(*)
   real(c_double), value, intent(in) :: fill_r8
   spel_vars(id)%rank = rank
   spel_vars(id)%lb(1:rank) = lb(1:rank)
   spel_vars(id)%ub(1:rank) = ub(1:rank)
   spel_vars(id)%has_bounds = .true.
   spel_vars(id)%has_fill = has_fill /= 0
   spel_vars(id)%fill_r8 = fill_r8
   spel_vars(id)%fill_int = fill_int
end subroutine
subroutine spel_py_allocate() bind(C, name='spel_py_allocate')
   if (spel_vars(1)%has_bounds) call spel_var_allocate(x, spel_vars(1))
   if (spel_vars(2)%has_bounds) call spel_var_allocate(y, spel_vars(2))
end subroutine
subroutine spel_py_setup(ng, nt, nl, nc, np) bind(C, name='spel_py_setup')
   integer(c_int), value, intent(in) :: ng, nt, nl, nc, np
   nsetup = nsetup + 1
   n_seen = 10000 * ng + 1000 * nt + 100 * nl + 10 * nc + np
end subroutine
subroutine spel_py_compute() bind(C, name='spel_py_compute')
   y(:, 1) = 2.0d0 * x
end subroutine
subroutine spel_py_trap(on) bind(C, name='spel_py_trap')
   integer(c_int), value, intent(in) :: on
end subroutine
function spel_py_true() result(t) bind(C, name='spel_py_true')
   integer(c_int) :: t
   t = transfer(.true., 0_c_int)
end function
function spel_py_view(id, addr, lb, ub) result(rank) bind(C, name='spel_py_view')
   integer(c_int), value, intent(in) :: id
   type(c_ptr), intent(out) :: addr
   integer(c_int), intent(out) :: lb(7), ub(7)
   integer(c_int) :: rank
   addr = c_null_ptr; lb = 1; ub = 0; rank = -1
   select case (id)
   case (1)
      if (allocated(x)) then
         addr = c_loc(x(lbound(x, 1))); rank = 1; lb(1) = lbound(x, 1); ub(1) = ubound(x, 1)
      end if
   case (2)
      if (allocated(y)) then
         addr = c_loc(y(lbound(y, 1), lbound(y, 2))); rank = 2
         lb(1:2) = lbound(y); ub(1:2) = ubound(y)
      end if
   case (3)
      addr = c_loc(nsetup); rank = 0
   case (4)
      addr = c_loc(n_seen); rank = 0
   case default
      rank = -2
   end select
end function
end module SpelPyMod
"""


def test_direct_init_allocates_from_descriptors(tmp_path):
    from spel.scripts.io.netcdf_io import gen_spel_var_module

    (tmp_path / "spel_varMod.F90").write_text(gen_spel_var_module())
    (tmp_path / "SpelPyMod.f90").write_text(DIRECT)
    res = subprocess.run(["gfortran", "-shared", "-fPIC", "-o", "libspelcase.so", "spel_varMod.F90", "SpelPyMod.f90"],
                         cwd=tmp_path, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    entries = [("x", "real", "input", True), ("y", "real", "output", True), ("nsetup", "integer", "time_io", False),
               ("n_seen", "integer", "time_io", False)]
    manifest = {"case": "t", "direct": True, "entries": [
        {"id": k + 1, "name": n, "type": t, "kind": kind, "alloc": a} for k, (n, t, kind, a) in enumerate(entries)]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    direct = {"sizes": {"g": 1, "t": 1, "l": 1, "c": 3, "p": 3}, "alloc": {"x": ([(0, 2)], "real"), "y": ([(1, 3), (-1, 1)], "real")},
              "values": {"x": np.array([1.0, 2.0, 3.0])}}
    with Worker(tmp_path / "libspelcase.so", tmp_path / "manifest.json", "unused/", log_dir=tmp_path / "logs",
                direct=direct) as w:
        got = w.get(["x", "y", "nsetup", "n_seen"])
        assert got["x"].lb == (0,) and got["x"].data.tolist() == [1.0, 2.0, 3.0]  # bounds from Python, value set
        assert got["y"].lb == (1, -1) and got["y"].data.shape == (3, 3)
        assert (got["y"].data == 1.0e36).all()  # the descriptor's fill
        assert int(got["nsetup"].data) == 1 and int(got["n_seen"].data) == 11133  # the sizes in g, t, l, c, p order
        out = w.run({"x": np.array([0.5, 1.0, 1.5])}, ["y"])
        assert out["y"].data[:, 2].tolist() == [1.0, 2.0, 3.0]  # y(:, 1): the second dimension starts at -1
