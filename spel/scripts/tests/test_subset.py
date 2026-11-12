import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from spel.scripts.config import spel_mods_dir
from spel.scripts.instrument_elm import CaptureNames, generate_capture_module
from spel.scripts.subset import (
    MANIFEST_SUFFIX,
    PRESETS,
    RankManifest,
    SubsetError,
    SubsetSpec,
    capture_select_routine,
    diff_datasets,
    level_keep,
    merge_datasets,
    merge_plan,
    merge_rank_files,
    subset_dataset,
)


def test_parse_presets():
    spec = SubsetSpec.parse("urban")
    assert spec.terms == (PRESETS["urban"][0],)
    assert spec.max_cells == 100 and not spec.custom
    spec = SubsetSpec.parse(" Urban, lake ; max=7")
    assert spec.mask == "urban,lake" and spec.max_cells == 7 and len(spec.terms) == 2
    assert SubsetSpec.parse("crop", max_cells=3).max_cells == 3
    assert "landunit_varcon, only : istcrop" in SubsetSpec.parse("crop").uses()


def test_parse_custom():
    spec = SubsetSpec.parse("landunit: lun_pp%itype(l) == istsoil .and. lun_pp%wtgcell(l) > 0.5_r8")
    assert spec.custom and spec.terms[0].level == "l"
    assert "pftvarcon" in spec.uses()
    # a comma inside a custom expression is not a preset list
    spec = SubsetSpec.parse("patch: max(veg_pp%wtgcell(p), 0._r8) > 0.1_r8")
    assert spec.terms[0].level == "p"
    assert SubsetSpec.from_json(spec.to_json()) == spec


@pytest.mark.parametrize(
    "text",
    ["suburban", "landunit: ", "landunit: lun_pp%urbpoi(c)", "landunit: f(l; x)",
     "column: (col_pp%active(c)", "urban; max=0"],
)
def test_parse_errors(text):
    with pytest.raises(SubsetError):
        SubsetSpec.parse(text)


def test_capture_module():
    names = CaptureNames.for_case("urb_sub")
    plain = generate_capture_module(names, ["energyflux_vars"])
    assert "spel_select_subset" not in plain and "nc_subset" not in plain
    spec = SubsetSpec.parse("urban; max=5")
    text = generate_capture_module(names, ["energyflux_vars"], subset=spec)
    assert "integer, parameter :: spel_subset_max = 5" in text
    assert "if (lun_pp%urbpoi(l) .and. lun_pp%wtgcell(l) > 0._r8) match(lun_pp%gridcell(l)) = .true." in text
    assert "call write_elmtypes(io_inputs, spel_bounds_sub," in text
    assert "call write_elmtypes(io_outputs, spel_bounds_sub," in text
    assert text.count("call nc_subset_activate(spel_sub)") == 3
    assert text.count("call nc_subset_deactivate()") == 3
    # every rank writes its own files; the master rank writes the manifest
    assert "write (spel_rank_pre, '(a,i0.5,a)') 'rank', iam, '.'" in text
    assert "base_fn='urb_sub.'//trim(spel_rank_pre)//'spel-inputs'" in text
    assert "call mpi_gather(n, 1, MPI_INTEGER, nkeep, 1, MPI_INTEGER, 0, mpicom, ierr)" in text
    assert "file='urb_sub.spel-subset-ranks.txt'" in text
    assert "MPI_MAX" not in text


# --------------------------------------------------------------------------
# nc_io's subset writer vs the offline subsetter
# --------------------------------------------------------------------------
# 4 gridcells with 2/1/3/2 landunits, one topounit each. The process bounds
# don't start at 1 (as on an MPI rank other than 0).
ROUNDTRIP = r"""
program subset_roundtrip
  use netcdf
  use nc_io
  implicit none
  type :: bounds_t
     integer :: beg(5), end(5)
  end type
  integer, parameter :: begg = 11, ng = 4, begt = 11, begl = 101, nl = 8
  integer, parameter :: begc = 201, nc = 10, begp = 301, np = 13
  integer, parameter :: nl_g(ng) = [2, 1, 3, 2]
  integer, parameter :: nc_l(nl) = [1, 2, 1, 1, 2, 1, 1, 1]
  integer, parameter :: np_c(nc) = [1, 2, 1, 1, 1, 2, 1, 1, 2, 1]
  integer :: top_gridcell(begt:begt+ng-1), top_downhill(begt:begt+ng-1)
  integer :: grc_coli(begg:begg+ng-1), grc_colf(begg:begg+ng-1), grc_gindex(begg:begg+ng-1)
  integer :: lun_gridcell(begl:begl+nl-1)
  logical :: lun_urbpoi(begl:begl+nl-1)
  integer :: col_landunit(begc:begc+nc-1), col_gridcell(begc:begc+nc-1)
  integer :: veg_column(begp:begp+np-1), veg_gridcell(begp:begp+np-1)
  real(8) :: veg_x(begp:begp+np-1, 3)
  type(spel_subset_type), target :: sub
  type(bounds_t) :: b_full, b_sub
  logical :: keep(begg:begg+ng-1)
  integer :: g, l, c, p, k

  l = begl; c = begc; p = begp
  do g = begg, begg + ng - 1
     top_gridcell(g) = g
     grc_gindex(g) = 1000 + g
     grc_coli(g) = c
     do k = 1, nl_g(g - begg + 1)
        lun_gridcell(l) = g
        lun_urbpoi(l) = mod(l, 3) == 0
        do c = c, c + nc_l(l - begl + 1) - 1
           col_landunit(c) = l
           col_gridcell(c) = g
           veg_column(p:p + np_c(c - begc + 1) - 1) = c
           veg_gridcell(p:p + np_c(c - begc + 1) - 1) = g
           p = p + np_c(c - begc + 1)
        end do
        l = l + 1
     end do
     grc_colf(g) = c - 1
  end do
  ! kept t12 -> kept t14, kept t14 -> dropped t13
  top_downhill = [13, 14, -9999, 13]
  do k = 1, 3
     do p = begp, begp + np - 1
        veg_x(p, k) = p + 0.25d0*k
     end do
  end do

  keep = .false.
  keep(begg + 1) = .true.
  keep(begg + 3) = .true.
  call nc_subset_set_level(sub, 1, begg, keep)
  call nc_subset_set_level(sub, 2, begt, keep(top_gridcell))
  call nc_subset_set_level(sub, 3, begl, keep(lun_gridcell))
  call nc_subset_set_level(sub, 4, begc, keep(col_gridcell))
  call nc_subset_set_level(sub, 5, begp, keep(veg_gridcell))
  sub%mask = 'test mask'
  sub%gindex = pack(grc_gindex, keep)

  b_full%beg = [begg, begt, begl, begc, begp]
  b_full%end = b_full%beg + [ng, ng, nl, nc, np] - 1
  b_sub%beg = 1
  do k = 1, 5
     b_sub%end(k) = nc_subset_count(sub, k)
  end do

  call write_file('full.nc', b_full)
  call nc_subset_activate(sub)
  call write_file('sub.nc', b_sub)
  call nc_subset_deactivate()

contains

  subroutine write_file(fn, b)
    character(len=*), intent(in) :: fn
    type(bounds_t), intent(in) :: b
    character(len=1), parameter :: lev(5) = ['g', 't', 'l', 'c', 'p']
    character(len=32) :: dn(2)
    integer :: ncid, varid, t, i
    ncid = nc_create_or_open_file(fn, create_file)
    do i = 1, 5
       call nc_define_var(ncid, 0, [0], dn, 'bounds__beg'//lev(i), nf90_int, varid, .false.)
       call nc_define_var(ncid, 0, [0], dn, 'bounds__end'//lev(i), nf90_int, varid, .false.)
    end do
    call def1(ncid, 'topo', 'top_pp__gridcell', nf90_int, lbound(top_gridcell), ubound(top_gridcell))
    call def1(ncid, 'topo', 'top_pp__downhill_ti', nf90_int, lbound(top_downhill), ubound(top_downhill))
    call def1(ncid, 'gridcell', 'grc_pp__coli', nf90_int, lbound(grc_coli), ubound(grc_coli))
    call def1(ncid, 'gridcell', 'grc_pp__colf', nf90_int, lbound(grc_colf), ubound(grc_colf))
    call def1(ncid, 'gridcell', 'grc_pp__gindex', nf90_int, lbound(grc_gindex), ubound(grc_gindex))
    call def1(ncid, 'landunit', 'lun_pp__gridcell', nf90_int, lbound(lun_gridcell), ubound(lun_gridcell))
    call def1(ncid, 'landunit', 'lun_pp__urbpoi', nf90_int, lbound(lun_urbpoi), ubound(lun_urbpoi))
    call def1(ncid, 'column', 'col_pp__landunit', nf90_int, lbound(col_landunit), ubound(col_landunit))
    call def1(ncid, 'column', 'col_pp__gridcell', nf90_int, lbound(col_gridcell), ubound(col_gridcell))
    call def1(ncid, 'patch', 'veg_pp__column', nf90_int, lbound(veg_column), ubound(veg_column))
    call def1(ncid, 'patch', 'veg_pp__gridcell', nf90_int, lbound(veg_gridcell), ubound(veg_gridcell))
    dn(1) = 'patch'; dn(2) = 'nlev'
    call nc_define_var(ncid, 2, shape(veg_x), dn, 'veg_x', nf90_double, varid, .true.)
    call nc_put_bounds_att(ncid, varid, 2, dn, lbound(veg_x), ubound(veg_x))
    call check(nf90_enddef(ncid))
    do i = 1, 5
       call nc_write_var_scalar(ncid, b%beg(i), 'bounds__beg'//lev(i))
       call nc_write_var_scalar(ncid, b%end(i), 'bounds__end'//lev(i))
    end do
    do t = 1, 2
       call w1(ncid, 'topo', top_gridcell, 'top_pp__gridcell', t)
       call w1(ncid, 'topo', top_downhill, 'top_pp__downhill_ti', t)
       call w1(ncid, 'gridcell', grc_coli, 'grc_pp__coli', t)
       call w1(ncid, 'gridcell', grc_colf, 'grc_pp__colf', t)
       call w1(ncid, 'gridcell', grc_gindex, 'grc_pp__gindex', t)
       call w1(ncid, 'landunit', lun_gridcell, 'lun_pp__gridcell', t)
       call nc_write_var_array(ncid, 1, shape(lun_urbpoi), [character(len=32) :: 'landunit'], &
                               lun_urbpoi, 'lun_pp__urbpoi', t)
       call w1(ncid, 'column', col_landunit, 'col_pp__landunit', t)
       call w1(ncid, 'column', col_gridcell, 'col_pp__gridcell', t)
       call w1(ncid, 'patch', veg_column, 'veg_pp__column', t)
       call w1(ncid, 'patch', veg_gridcell, 'veg_pp__gridcell', t)
       call nc_write_var_array(ncid, 2, shape(veg_x), [character(len=32) :: 'patch', 'nlev'], &
                               reshape(veg_x, [size(veg_x)]), 'veg_x', t)
    end do
    call check(nf90_close(ncid))
  end subroutine write_file

  subroutine def1(ncid, dim, name, xtype, lb, ub)
    integer, intent(in) :: ncid
    character(len=*), intent(in) :: dim, name
    integer, intent(in) :: xtype, lb(1), ub(1)
    character(len=32) :: dn(1)
    integer :: varid
    dn(1) = dim
    call nc_define_var(ncid, 1, ub - lb + 1, dn, name, xtype, varid, .true.)
    call nc_put_bounds_att(ncid, varid, 1, dn, lb, ub)
  end subroutine def1

  subroutine w1(ncid, dim, var, name, t)
    integer, intent(in) :: ncid
    character(len=*), intent(in) :: dim, name
    integer, intent(in) :: var(:), t
    character(len=32) :: dn(1)
    dn(1) = dim
    call nc_write_var_array(ncid, 1, shape(var), dn, var, name, t)
  end subroutine w1
end program subset_roundtrip
"""


def _netcdf_fortran_flags() -> list[str] | None:
    if shutil.which("gfortran") is None or shutil.which("pkg-config") is None:
        return None
    res = subprocess.run(
        ["pkg-config", "--cflags", "--libs", "netcdf-fortran"], capture_output=True, text=True
    )
    if res.returncode != 0:
        return None
    var = lambda v: subprocess.run(
        ["pkg-config", f"--variable={v}", "netcdf-fortran"], capture_output=True, text=True
    ).stdout.strip()
    libdir = var("libdir")
    # pkg-config drops -I/-L of directories on CPATH/LIBRARY_PATH (spack modules
    # set them), but gfortran looks for .mod files only in -I directories.
    incdirs = [f"-I{d}" for d in dict.fromkeys((var("fmoddir"), var("includedir"))) if d]
    return incdirs + res.stdout.split() + ([f"-L{libdir}", f"-Wl,-rpath,{libdir}"] if libdir else [])


def _build_and_run(tmp_path_factory, source: str) -> Path:
    flags = _netcdf_fortran_flags()
    if flags is None:
        pytest.skip("needs gfortran and netcdf-fortran (pkg-config)")
    d = tmp_path_factory.mktemp("subset")
    (d / "prog.F90").write_text(source)
    cmd = ["gfortran", "-ffree-line-length-none", str(Path(spel_mods_dir) / "nc_io.F90"),
           "prog.F90", "-o", "prog", *flags]
    res = subprocess.run(cmd, cwd=d, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    res = subprocess.run(["./prog"], cwd=d, capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
    return d


@pytest.fixture(scope="module")
def roundtrip(tmp_path_factory) -> Path:
    return _build_and_run(tmp_path_factory, ROUNDTRIP)


def test_fortran_subset_matches_offline(roundtrip):
    full, sub = roundtrip / "full.nc", roundtrip / "sub.nc"
    keep = level_keep(full, [2, 4])
    assert [int(k.sum()) for k in keep.values()] == [2, 2, 3, 3, 4]
    offline = roundtrip / "offline.nc"
    subset_dataset(full, offline, keep, mask="test mask", gindex=[1012, 1014])
    assert diff_datasets(sub, offline) == []


def test_fortran_subset_renumbers(roundtrip):
    import netCDF4

    with netCDF4.Dataset(roundtrip / "sub.nc") as ds:
        ds.set_auto_mask(False)
        val = lambda n: ds[n][1].tolist()  # second time step
        assert {n: int(ds[f"bounds__{n}"][...]) for n in ("begg", "endg", "endl", "endc", "endp")} == \
            {"begg": 1, "endg": 2, "endl": 3, "endc": 3, "endp": 4}
        # kept: g12, g14; l103, l107, l108; c204, c209, c210; p305, p311-313
        assert val("lun_pp__gridcell") == [1, 2, 2]
        assert val("lun_pp__urbpoi") == [0, 0, 1]
        assert val("col_pp__landunit") == [1, 2, 3]
        assert val("col_pp__gridcell") == [1, 2, 2]
        assert val("veg_pp__column") == [1, 2, 2, 3]
        assert val("grc_pp__coli") == [1, 2] and val("grc_pp__colf") == [1, 3]
        assert val("grc_pp__gindex") == [1, 2]
        assert val("top_pp__downhill_ti") == [2, -9999]
        x = ds["veg_x"][1]  # netCDF order (nlev, patch)
        np.testing.assert_array_equal(x[0], [305.25, 311.25, 312.25, 313.25])
        np.testing.assert_array_equal(x[2], [305.75, 311.75, 312.75, 313.75])
        assert ds["veg_x"].lbounds.tolist() == [1, 1] and ds["veg_x"].ubounds.tolist() == [4, 3]
        assert ds.spel_subset_mask == "test mask"
        assert ds.spel_subset_gindex.tolist() == [1012, 1014]


# --------------------------------------------------------------------------
# Gridcells with several topounits (IM2 hillslope hydrology)
# --------------------------------------------------------------------------
# 3 gridcells with 3/1/2 topounits, one column and one patch per landunit;
# process bounds not at 1. downhill_ti as ELM's set_topounit makes it for
# elevations (300, 150, 50) in g21 (the outlet t43 points at itself: the
# dn_index bug), none in g22 (-1) and (50, 150) in g23. landunit_indices is
# (max_lunit, begt:endt): the topounit dimension is the second one, and
# ispval (-9999) marks landunit types a topounit lacks.
TOPO_ROUNDTRIP = r"""
program subset_topounits
  use netcdf
  use nc_io
  implicit none
  integer, parameter :: begg = 21, ng = 3, begt = 41, nt = 6, begl = 101, nl = 8
  integer, parameter :: begc = 201, nc = 8, begp = 301, np = 8, max_lunit = 3, ispval = -9999
  integer :: grc_topi(begg:begg+ng-1) = [41, 44, 45], grc_topf(begg:begg+ng-1) = [43, 44, 46]
  integer :: grc_gindex(begg:begg+ng-1) = [5021, 5022, 5023]
  integer :: top_gridcell(begt:begt+nt-1) = [21, 21, 21, 22, 23, 23]
  integer :: top_downhill(begt:begt+nt-1) = [42, 43, 43, -1, -1, 45]
  integer :: top_lndi(begt:begt+nt-1) = [101, 103, 104, 105, 106, 107]
  integer :: top_lndf(begt:begt+nt-1) = [102, 103, 104, 105, 106, 108]
  integer :: top_lunidx(max_lunit, begt:begt+nt-1)
  real(8) :: top_from_uphill(begt:begt+nt-1)
  integer :: lun_gridcell(begl:begl+nl-1) = [21, 21, 21, 21, 22, 23, 23, 23]
  integer :: lun_topounit(begl:begl+nl-1) = [41, 41, 42, 43, 44, 45, 46, 46]
  integer :: col_landunit(begc:begc+nc-1), veg_column(begp:begp+np-1)
  type(spel_subset_type), target :: sub
  integer :: bbeg(5), bend(5)
  logical :: keep(begg:begg+ng-1)
  integer :: t, k

  top_lunidx = ispval
  top_lunidx(1, 41) = 101; top_lunidx(3, 41) = 102; top_lunidx(1, 42) = 103
  top_lunidx(1, 43) = 104; top_lunidx(1, 44) = 105; top_lunidx(3, 45) = 106
  top_lunidx(1, 46) = 107; top_lunidx(2, 46) = 108
  do t = begt, begt + nt - 1
     top_from_uphill(t) = 0.5d0*t
  end do
  col_landunit = [(begl + k, k = 0, nl - 1)]
  veg_column = [(begc + k, k = 0, nc - 1)]

  keep = [.true., .false., .true.]
  call nc_subset_set_level(sub, 1, begg, keep)
  call nc_subset_set_level(sub, 2, begt, keep(top_gridcell))
  call nc_subset_set_level(sub, 3, begl, keep(lun_gridcell))
  call nc_subset_set_level(sub, 4, begc, keep(lun_gridcell))  ! one column per landunit
  call nc_subset_set_level(sub, 5, begp, keep(lun_gridcell))  ! one patch per column
  sub%mask = 'topounit: top_pp%downhill_ti(t) == t'
  sub%gindex = pack(grc_gindex, keep)

  bbeg = [begg, begt, begl, begc, begp]
  bend = bbeg + [ng, nt, nl, nc, np] - 1
  call write_file('full.nc', bbeg, bend)
  call nc_subset_activate(sub)
  bbeg = 1
  do k = 1, 5
     bend(k) = nc_subset_count(sub, k)
  end do
  call write_file('sub.nc', bbeg, bend)
  call nc_subset_deactivate()

contains

  subroutine write_file(fn, bbeg, bend)
    character(len=*), intent(in) :: fn
    integer, intent(in) :: bbeg(5), bend(5)
    character(len=1), parameter :: lev(5) = ['g', 't', 'l', 'c', 'p']
    character(len=32) :: dn(2)
    integer :: ncid, varid, step, i
    ncid = nc_create_or_open_file(fn, create_file)
    do i = 1, 5
       call nc_define_var(ncid, 0, [0], dn, 'bounds__beg'//lev(i), nf90_int, varid, .false.)
       call nc_define_var(ncid, 0, [0], dn, 'bounds__end'//lev(i), nf90_int, varid, .false.)
    end do
    call def1(ncid, 'gridcell', 'grc_pp__topi', nf90_int, lbound(grc_topi), ubound(grc_topi))
    call def1(ncid, 'gridcell', 'grc_pp__topf', nf90_int, lbound(grc_topf), ubound(grc_topf))
    call def1(ncid, 'gridcell', 'grc_pp__gindex', nf90_int, lbound(grc_gindex), ubound(grc_gindex))
    call def1(ncid, 'topo', 'top_pp__gridcell', nf90_int, lbound(top_gridcell), ubound(top_gridcell))
    call def1(ncid, 'topo', 'top_pp__downhill_ti', nf90_int, lbound(top_downhill), ubound(top_downhill))
    call def1(ncid, 'topo', 'top_pp__lndi', nf90_int, lbound(top_lndi), ubound(top_lndi))
    call def1(ncid, 'topo', 'top_pp__lndf', nf90_int, lbound(top_lndf), ubound(top_lndf))
    call def1(ncid, 'topo', 'top_ws__from_uphill', nf90_double, lbound(top_from_uphill), ubound(top_from_uphill))
    dn(1) = 'max_lunit'; dn(2) = 'topo'
    call nc_define_var(ncid, 2, shape(top_lunidx), dn, 'top_pp__landunit_indices', nf90_int, varid, .true.)
    call nc_put_bounds_att(ncid, varid, 2, dn, lbound(top_lunidx), ubound(top_lunidx))
    call def1(ncid, 'landunit', 'lun_pp__gridcell', nf90_int, lbound(lun_gridcell), ubound(lun_gridcell))
    call def1(ncid, 'landunit', 'lun_pp__topounit', nf90_int, lbound(lun_topounit), ubound(lun_topounit))
    call def1(ncid, 'column', 'col_pp__landunit', nf90_int, lbound(col_landunit), ubound(col_landunit))
    call def1(ncid, 'column', 'col_pp__gridcell', nf90_int, lbound(lun_gridcell), ubound(lun_gridcell))
    call def1(ncid, 'column', 'col_pp__topounit', nf90_int, lbound(lun_topounit), ubound(lun_topounit))
    call def1(ncid, 'patch', 'veg_pp__column', nf90_int, lbound(veg_column), ubound(veg_column))
    call def1(ncid, 'patch', 'veg_pp__gridcell', nf90_int, lbound(lun_gridcell), ubound(lun_gridcell))
    call def1(ncid, 'patch', 'veg_pp__topounit', nf90_int, lbound(lun_topounit), ubound(lun_topounit))
    call check(nf90_enddef(ncid))
    do i = 1, 5
       call nc_write_var_scalar(ncid, bbeg(i), 'bounds__beg'//lev(i))
       call nc_write_var_scalar(ncid, bend(i), 'bounds__end'//lev(i))
    end do
    do step = 1, 2
       call w1(ncid, 'gridcell', grc_topi, 'grc_pp__topi', step)
       call w1(ncid, 'gridcell', grc_topf, 'grc_pp__topf', step)
       call w1(ncid, 'gridcell', grc_gindex, 'grc_pp__gindex', step)
       call w1(ncid, 'topo', top_gridcell, 'top_pp__gridcell', step)
       call w1(ncid, 'topo', top_downhill, 'top_pp__downhill_ti', step)
       call w1(ncid, 'topo', top_lndi, 'top_pp__lndi', step)
       call w1(ncid, 'topo', top_lndf, 'top_pp__lndf', step)
       call nc_write_var_array(ncid, 1, shape(top_from_uphill), [character(len=32) :: 'topo'], &
                               top_from_uphill + step, 'top_ws__from_uphill', step)
       call nc_write_var_array(ncid, 2, shape(top_lunidx), [character(len=32) :: 'max_lunit', 'topo'], &
                               reshape(top_lunidx, [size(top_lunidx)]), 'top_pp__landunit_indices', step)
       call w1(ncid, 'landunit', lun_gridcell, 'lun_pp__gridcell', step)
       call w1(ncid, 'landunit', lun_topounit, 'lun_pp__topounit', step)
       call w1(ncid, 'column', col_landunit, 'col_pp__landunit', step)
       call w1(ncid, 'column', lun_gridcell, 'col_pp__gridcell', step)
       call w1(ncid, 'column', lun_topounit, 'col_pp__topounit', step)
       call w1(ncid, 'patch', veg_column, 'veg_pp__column', step)
       call w1(ncid, 'patch', lun_gridcell, 'veg_pp__gridcell', step)
       call w1(ncid, 'patch', lun_topounit, 'veg_pp__topounit', step)
    end do
    call check(nf90_close(ncid))
  end subroutine write_file

  subroutine def1(ncid, dim, name, xtype, lb, ub)
    integer, intent(in) :: ncid
    character(len=*), intent(in) :: dim, name
    integer, intent(in) :: xtype, lb(1), ub(1)
    character(len=32) :: dn(1)
    integer :: varid
    dn(1) = dim
    call nc_define_var(ncid, 1, ub - lb + 1, dn, name, xtype, varid, .true.)
    call nc_put_bounds_att(ncid, varid, 1, dn, lb, ub)
  end subroutine def1

  subroutine w1(ncid, dim, var, name, step)
    integer, intent(in) :: ncid
    character(len=*), intent(in) :: dim, name
    integer, intent(in) :: var(:), step
    character(len=32) :: dn(1)
    dn(1) = dim
    call nc_write_var_array(ncid, 1, shape(var), dn, var, name, step)
  end subroutine w1
end program subset_topounits
"""


@pytest.fixture(scope="module")
def topo_roundtrip(tmp_path_factory) -> Path:
    return _build_and_run(tmp_path_factory, TOPO_ROUNDTRIP)


def test_topounit_subset_matches_offline(topo_roundtrip):
    full, sub = topo_roundtrip / "full.nc", topo_roundtrip / "sub.nc"
    keep = level_keep(full, [1, 3])
    assert [int(k.sum()) for k in keep.values()] == [2, 5, 7, 7, 7]
    assert keep["t"].tolist() == [True, True, True, False, True, True]
    offline = topo_roundtrip / "offline.nc"
    subset_dataset(full, offline, keep, mask="topounit: top_pp%downhill_ti(t) == t",
                   gindex=[5021, 5023])
    assert diff_datasets(sub, offline) == []


def test_topounit_subset_renumbers(topo_roundtrip):
    import netCDF4

    with netCDF4.Dataset(topo_roundtrip / "sub.nc") as ds:
        ds.set_auto_mask(False)
        val = lambda n: ds[n][1].tolist()  # second time step
        assert {n: int(ds[f"bounds__{n}"][...]) for n in ("begt", "endt", "endl")} == \
            {"begt": 1, "endt": 5, "endl": 7}
        # kept: g21, g23; t41-43, t45, t46; l101-104, l106-108
        assert val("grc_pp__topi") == [1, 4] and val("grc_pp__topf") == [3, 5]
        assert val("top_pp__gridcell") == [1, 1, 1, 2, 2]
        # neighbours stay within their gridcell; the self-loop stays a self-loop
        # and -1 (no neighbour) is not an index, so it is kept as is
        assert val("top_pp__downhill_ti") == [2, 3, 3, -1, 4]
        assert val("top_pp__lndi") == [1, 3, 4, 5, 6] and val("top_pp__lndf") == [2, 3, 4, 5, 7]
        assert val("lun_pp__topounit") == [1, 1, 2, 3, 4, 5, 5]
        assert val("col_pp__topounit") == [1, 1, 2, 3, 4, 5, 5]
        assert val("veg_pp__topounit") == [1, 1, 2, 3, 4, 5, 5]
        # netCDF order (topo, max_lunit); ispval entries untouched
        assert val("top_pp__landunit_indices") == [
            [1, -9999, 2], [3, -9999, -9999], [4, -9999, -9999], [-9999, -9999, 5], [6, 7, -9999],
        ]
        assert ds["top_pp__landunit_indices"].lbounds.tolist() == [1, 1]
        assert ds["top_pp__landunit_indices"].ubounds.tolist() == [3, 5]
        np.testing.assert_array_equal(ds["top_ws__from_uphill"][1], [22.5, 23.0, 23.5, 24.5, 25.0])
        assert ds.spel_subset_gindex.tolist() == [5021, 5023]


# --------------------------------------------------------------------------
# spel_select_subset with stub ELM modules and a one-process "MPI"
# --------------------------------------------------------------------------
SELECT_STUBS = r"""
module stub_elm
  implicit none
  integer, parameter :: r8 = selected_real_kind(12)
  type bounds_type
     integer :: begg, endg, begt, endt, begl, endl, begc, endc, begp, endp
  end type
  type grc_t
     integer, allocatable :: gindex(:)
     real(r8), allocatable :: latdeg(:), londeg(:)
  end type
  type sub_t
     integer, allocatable :: gridcell(:)
  end type
  type(grc_t) :: grc_pp
  type(sub_t) :: top_pp, lun_pp, col_pp, veg_pp
  logical :: masterproc = .true.
  integer :: iam = 0, npes = 1, mpicom = 0
  integer, parameter :: MPI_INTEGER = 0, MPI_SUM = 0
  type(bounds_type) :: proc
contains
  subroutine get_proc_bounds(b)
    type(bounds_type), intent(out) :: b
    b = proc
  end subroutine
  integer function get_proc_clumps()
    get_proc_clumps = 1
  end function
  subroutine get_clump_bounds(n, b)
    integer, intent(in) :: n
    type(bounds_type), intent(out) :: b
    b = proc
  end subroutine
end module stub_elm

module capture
  use stub_elm
  use nc_io, only : spel_subset_type, nc_subset_set_level, nc_subset_count
  implicit none
  integer, parameter :: spel_subset_max = 2
  type(spel_subset_type), target :: spel_sub
  type(bounds_type) :: spel_bounds_sub
  integer :: spel_clump_begg = -1
  logical :: spel_writer = .false.
contains
!ROUTINE
end module capture

program select
  use stub_elm
  use capture
  implicit none
  integer :: k
  ! 4 gridcells (gindex 1001-1004) with 2 topounits each, one landunit,
  ! column and patch per topounit; process bounds not at 1
  proc = bounds_type(11, 14, 21, 28, 31, 38, 41, 48, 51, 58)
  allocate (grc_pp%gindex(11:14), grc_pp%latdeg(11:14), grc_pp%londeg(11:14))
  grc_pp%gindex = [1001, 1002, 1003, 1004]
  grc_pp%latdeg = [1._r8, 2._r8, 3._r8, 4._r8]
  grc_pp%londeg = 0._r8
  allocate (top_pp%gridcell(21:28), source=[(11 + (k - 1)/2, k = 1, 8)])
  allocate (lun_pp%gridcell(31:38), source=top_pp%gridcell)
  allocate (col_pp%gridcell(41:48), source=top_pp%gridcell)
  allocate (veg_pp%gridcell(51:58), source=top_pp%gridcell)
  call spel_select_subset(proc)
  print '(a,l1)', 'writer ', spel_writer
  print '(a,*(1x,i0))', 'gindex', spel_sub%gindex
  print '(a,*(1x,i0))', 'counts', (nc_subset_count(spel_sub, k), k = 1, 5)
end program select

subroutine mpi_allreduce(sendbuf, recvbuf, n, dtype, op, comm, ierr)
  integer :: sendbuf, recvbuf, n, dtype, op, comm, ierr
  recvbuf = sendbuf
  ierr = 0
end subroutine
subroutine mpi_gather(sendbuf, n, stype, recvbuf, m, rtype, root, comm, ierr)
  integer :: sendbuf, n, stype, recvbuf(*), m, rtype, root, comm, ierr
  recvbuf(1) = sendbuf
  ierr = 0
end subroutine
"""


def test_select_routine_writes_share_and_manifest(tmp_path_factory):
    routine = capture_select_routine(SubsetSpec.parse("gridcell: grc_pp%gindex(g) > 1001"), "sel")
    d = _build_and_run(tmp_path_factory, SELECT_STUBS.replace("!ROUTINE", routine))
    res = subprocess.run(["./prog"], cwd=d, capture_output=True, text=True)
    lines = {ln.split()[0]: ln.split()[1:] for ln in res.stdout.splitlines() if ln.strip()}
    # 3 matches, max 2: the share is min(3, ceiling(2*3/3)) = 2, evenly spaced: 1002, 1003
    assert lines["writer"] == ["T"]
    assert lines["gindex"] == ["1002", "1003"]
    assert lines["counts"] == ["2", "4", "4", "4", "4"]
    manifest = RankManifest.read(d / f"sel.{MANIFEST_SUFFIX}")
    assert manifest == RankManifest(max_cells=2, matches=3, ranks={0: 2})


# --------------------------------------------------------------------------
# Merging per-rank files
# --------------------------------------------------------------------------
TOPO_MASK = "topounit: top_pp%downhill_ti(t) == t"


def _rank_files(topo_roundtrip: Path, d: Path) -> tuple[Path, Path, Path]:
    """
    Two "ranks" of the topounit capture: rank 3 = the Fortran subset of g21
    and g23, rank 7 = g22 cut offline (so the merge has to interleave them by
    gindex); and the whole capture renumbered from 1, which their merge must
    reproduce.
    """
    full = topo_roundtrip / "full.nc"
    r3, r7, whole = d / "r3.nc", d / "r7.nc", d / "whole.nc"
    shutil.copy(topo_roundtrip / "sub.nc", r3)
    subset_dataset(full, r7, level_keep(full, [2]), mask=TOPO_MASK, gindex=[5022])
    subset_dataset(full, whole, level_keep(full, [1, 2, 3]), mask=TOPO_MASK, gindex=[5021, 5022, 5023])
    return r3, r7, whole


def test_merge_of_ranks_is_the_whole(topo_roundtrip, tmp_path):
    r3, r7, whole = _rank_files(topo_roundtrip, tmp_path)
    plan = merge_plan({3: r3, 7: r7})
    assert plan.gridcell_rank == [3, 7, 3]
    merged = tmp_path / "merged.nc"
    assert merge_datasets({3: r3, 7: r7}, merged, plan) == []
    assert diff_datasets(merged, whole, ignore_attrs=["spel_subset_rank"]) == []
    import netCDF4

    with netCDF4.Dataset(merged) as ds:
        assert ds.spel_subset_rank.tolist() == [3, 7, 3]
        assert ds["top_pp__downhill_ti"][1].tolist() == [2, 3, 3, -1, -1, 5]


def test_merge_cuts_to_max_evenly(topo_roundtrip, tmp_path):
    r3, r7, _ = _rank_files(topo_roundtrip, tmp_path)
    full = topo_roundtrip / "full.nc"
    merged, expect = tmp_path / "merged.nc", tmp_path / "expect.nc"
    # 3 gridcells, max 2: picks (k*3)//2 for k = 0, 1 -> gindex 5021, 5022
    merge_datasets({3: r3, 7: r7}, merged, merge_plan({3: r3, 7: r7}, max_cells=2))
    subset_dataset(full, expect, level_keep(full, [1, 2]), mask=TOPO_MASK, gindex=[5021, 5022])
    assert diff_datasets(merged, expect, ignore_attrs=["spel_subset_rank"]) == []


def test_merge_refuses_rank_differences(topo_roundtrip, tmp_path):
    import netCDF4

    r3, r7, _ = _rank_files(topo_roundtrip, tmp_path)
    for path, value in ((r3, 1.0), (r7, 2.0)):
        with netCDF4.Dataset(path, "a") as ds:
            ds.createVariable("spel_t_dtime", "f8", ())[...] = value
    plan = merge_plan({3: r3, 7: r7})
    with pytest.raises(SubsetError, match="spel_t_dtime has no subgrid dimension but differs"):
        merge_datasets({3: r3, 7: r7}, tmp_path / "m.nc", plan)
    assert merge_datasets({3: r3, 7: r7}, tmp_path / "m.nc", plan, allow_differ=["spel_t_dtime"]) == \
        ["spel_t_dtime"]
    with pytest.raises(SubsetError, match="several ranks"):
        merge_plan({3: r3, 4: r3})


def test_merge_rank_files_reads_the_manifest(topo_roundtrip, tmp_path):
    r3, r7, whole = _rank_files(topo_roundtrip, tmp_path)
    run, dest = tmp_path / "run", tmp_path / "data"
    run.mkdir()
    for rank, src in ((3, r3), (7, r7), (9, r7)):  # rank 9: left over from an earlier run
        for name in ("spel-constants0001.nc", "spel-inputs0001.nc"):
            shutil.copy(src, run / f"hs.rank{rank:05d}.{name}")
    assert merge_rank_files(run, dest, "hs.") is None  # no manifest: not a per-rank capture
    (run / f"hs.{MANIFEST_SUFFIX}").write_text("# SPEL\nmax 100\nmatches 3\n3 2\n7 1\n")
    files = merge_rank_files(run, dest, "hs.")
    assert [f.name for f in files] == ["spel-constants0001.nc", "spel-inputs0001.nc"]
    for f in files:
        assert diff_datasets(f, whole, ignore_attrs=["spel_subset_rank"]) == []
    (run / "hs.rank00007.spel-inputs0001.nc").unlink()
    with pytest.raises(SubsetError, match=r"missing for ranks \[7\]"):
        merge_rank_files(run, dest, "hs.")
    (run / f"hs.{MANIFEST_SUFFIX}").write_text("max 100\nmatches 0\n")
    assert merge_rank_files(run, dest, "hs.") == []


def _drop_vars(src: Path, dst: Path, drop: set[str]) -> Path:
    """Copy of a capture without some variables (captures hold only what the routines read)."""
    import netCDF4

    with netCDF4.Dataset(src) as si, netCDF4.Dataset(dst, "w", format=si.data_model) as so:
        si.set_auto_mask(False)
        so.setncatts({a: si.getncattr(a) for a in si.ncattrs()})
        for name, dim in si.dimensions.items():
            so.createDimension(name, None if dim.isunlimited() else len(dim))
        for name, vi in si.variables.items():
            if name in drop:
                continue
            attrs = {a: vi.getncattr(a) for a in vi.ncattrs()}
            vo = so.createVariable(name, vi.datatype, vi.dimensions, fill_value=attrs.pop("_FillValue", None))
            vo.setncatts(attrs)
            vo[...] = vi[...]
    return dst


# As in the hs-im2 capture: only col_pp%gridcell, the rest through *%topounit / *%landunit
PARTIAL_LINKS = {"top_pp__gridcell", "lun_pp__gridcell", "veg_pp__gridcell", "veg_pp__column"}


def test_parents_through_other_levels(topo_roundtrip, tmp_path):
    full = topo_roundtrip / "full.nc"
    part = _drop_vars(full, tmp_path / "part.nc", PARTIAL_LINKS)
    for a, b in zip(level_keep(part, [1, 3]).values(), level_keep(full, [1, 3]).values()):
        np.testing.assert_array_equal(a, b)
    # nothing links the patches to a gridcell any more
    bare = _drop_vars(part, tmp_path / "bare.nc", {"veg_pp__topounit"})
    with pytest.raises(SubsetError, match="patches"):
        level_keep(bare, [1])


def test_merge_with_partial_links(topo_roundtrip, tmp_path):
    r3, r7, whole = _rank_files(topo_roundtrip, tmp_path)
    p3 = _drop_vars(r3, tmp_path / "p3.nc", PARTIAL_LINKS)
    p7 = _drop_vars(r7, tmp_path / "p7.nc", PARTIAL_LINKS)
    pw = _drop_vars(whole, tmp_path / "pw.nc", PARTIAL_LINKS)
    merged = tmp_path / "merged.nc"
    merge_datasets({3: p3, 7: p7}, merged, merge_plan({3: p3, 7: p7}))
    assert diff_datasets(merged, pw, ignore_attrs=["spel_subset_rank"]) == []


def test_merge_dimension_differing_between_ranks(topo_roundtrip, tmp_path):
    """A flattened array (col_wf%mflx_neg_snow_1d, 'ncells') is process-sized on every rank."""
    import netCDF4

    r3, r7, _ = _rank_files(topo_roundtrip, tmp_path)
    for path, n in ((r3, 5), (r7, 8)):
        with netCDF4.Dataset(path, "a") as ds:
            ds.createDimension("ncells", n)
            ds.createVariable("col_wf__flat_1d", "f8", ("ncells",))[...] = np.arange(n)
    plan = merge_plan({3: r3, 7: r7})
    with pytest.raises(SubsetError, match=r"dimension ncells \(of col_wf__flat_1d\)"):
        merge_datasets({3: r3, 7: r7}, tmp_path / "m.nc", plan)
    assert merge_datasets({3: r3, 7: r7}, tmp_path / "m.nc", plan, allow_differ=["col_wf__flat_1d"]) == \
        ["col_wf__flat_1d"]
    with netCDF4.Dataset(tmp_path / "m.nc") as ds:
        assert ds["col_wf__flat_1d"][:].tolist() == [0, 1, 2, 3, 4]
