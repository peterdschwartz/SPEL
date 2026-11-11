"""The synthetic domain files, built from an inventory without a case behind it (tests/fake.py)."""

from types import SimpleNamespace

import numpy as np
import pytest

from spel.proptest import domain as dom
from spel.proptest.inventory import VarInfo
from spel.proptest.tests import fake
from spel.scripts.io.time_scalars import TimeScalar

netCDF4 = pytest.importorskip("netCDF4")


def test_domain_files_layout_and_what_was_defaulted(tmp_path):
    inv = fake.inventory(
        tmp_path / "case",
        [
            fake.var("col_es%t_soisno", "begc:endc,-nlevsno+1:nlev", "input"),
            fake.var("pftcon%slatop", "0:npft", "parameter"),
            VarInfo(
                name="nextsw_cday",
                instance="",
                field="nextsw_cday",
                type="real",
                rank=0,
                dims=[],
                status="r",
                kind="input",
                driver=True,
            ),
        ],
    )
    inv.parameters.update({"npft": 3})
    inv.unit_test = SimpleNamespace(
        time_scalars=[
            TimeScalar("jday_mod", "timeinfomod", "integer", "", ""),
            TimeScalar("dtime_mod", "timeinfomod", "real", "", ""),
            TimeScalar("my_flag", "m", "logical", "", ""),
        ],
        driver_args={},
    )
    spec = dom.DomainSpec.from_table(
        {
            "pfts": [1, 13],
            "per_pft": 2,
            "nsets": 2,
            "dims": {"NLEVSNO": 5},
            "time": {"jday_mod": 10},
        },
        tmp_path,
    )
    spec.check(inv)
    d = dom.build_domain(inv, spec, tmp_path / "domain")

    assert d.layout.count("p") == 8 and d.extents["col_es%t_soisno"] == [(1, 8), (-4, 4)]
    assert d.extents["nextsw_cday"] == [] and "nextsw_cday" in d.written
    assert d.filled == {
        "pftcon%slatop": "no reference capture"
    }  # a parameter with fill values: reported
    assert d.time_defaulted == {"dtime_mod": 1800.0, "my_flag": 0}  # jday_mod was given
    assert sorted(p.name for p in (tmp_path / "domain").iterdir()) == [
        dom.CONSTANTS_FILE,
        dom.INPUTS_FILE,
    ]
    with netCDF4.Dataset(tmp_path / "domain" / dom.INPUTS_FILE) as ds:
        t = ds.variables["col_es__t_soisno"]
        assert t.shape == (9, 8) and list(t.getncattr("lbounds")) == [
            1,
            -4,
        ]  # file dims are the Fortran dims reversed
        assert ds.variables["veg_pp__itype"][:].tolist() == [1, 1, 13, 13, 1, 1, 13, 13]
        assert ds.variables["spel_drv_nextsw_cday"].shape == (1,)
        assert (
            int(ds.variables["spel_t_jday_mod"][0]) == 10
            and float(ds.variables["spel_t_dtime_mod"][0]) == 1800.0
        )


def test_unknown_topology_field_stops_the_build(tmp_path):
    inv = fake.inventory(
        tmp_path / "case",
        [fake.var("veg_pp%mystery", "begp:endp", "topology", "integer")],
    )
    with pytest.raises(
        dom.DomainError,
        match="no synthetic-domain rule for topology field veg_pp%mystery",
    ):
        dom.build_domain(inv, dom.DomainSpec(pfts=[1]), tmp_path / "domain")


def test_dimension_without_a_value_names_the_fix(tmp_path):
    inv = fake.inventory(
        tmp_path / "case", [fake.var("a%z", "begc:endc,1:nlevmissing", "input")]
    )
    with pytest.raises(dom.DomainError, match=r"nlevmissing.*\[domain.dims\]"):
        dom.build_domain(inv, dom.DomainSpec(pfts=[1]), tmp_path / "domain")


def test_direct_init_keeps_bounds_and_values_instead_of_an_inputs_file(tmp_path):
    inv = fake.inventory(
        tmp_path / "case",
        [
            fake.var("col_es%t_soisno", "begc:endc,-nlevsno+1:nlev", "input"),
            fake.var("pftcon%slatop", "0:npft", "parameter"),
        ],
    )
    inv.parameters.update({"npft": 3})
    inv.unit_test = SimpleNamespace(
        time_scalars=[TimeScalar("jday_mod", "timeinfomod", "integer", "", "")], driver_args={}
    )
    spec = dom.DomainSpec.from_table(
        {"pfts": [1, 13], "per_pft": 2, "dims": {"nlevsno": 5}, "time": {"jday_mod": 10}, "init": "direct"},
        tmp_path,
    )
    files = tmp_path / "domain"
    files.mkdir()
    (files / dom.INPUTS_FILE).write_text("left by a files-mode run")
    d = dom.build_domain(inv, spec, files)

    assert sorted(p.name for p in files.iterdir()) == [dom.CONSTANTS_FILE]  # no inputs file
    assert d.direct["sizes"] == {lev: 4 for lev in "gtlcp"}
    # every variable is allocated with Fortran bounds; fill values are left to the descriptor
    assert d.direct["alloc"]["col_es%t_soisno"] == ([(1, 4), (-4, 4)], "real")
    assert d.direct["alloc"]["pftcon%slatop"] == ([(0, 3)], "real")
    assert "col_es%t_soisno" not in d.direct["values"] and "pftcon%slatop" not in d.direct["values"]
    # topology and the per-step arrays are values
    assert d.direct["values"]["veg_pp%itype"].tolist() == [1, 1, 13, 13]
    assert int(d.direct["values"]["spel_t_jday_mod"]) == 10
    assert d.filled == {"pftcon%slatop": "no reference capture"}  # reported as in files mode


def test_init_mode_is_checked(tmp_path):
    with pytest.raises(dom.DomainError, match="init must be one of"):
        dom.DomainSpec.from_table({"pfts": [1], "init": "memory"}, tmp_path)


def test_urban_and_lake_sites_have_their_own_columns():
    lay = dom.Layout([1], per_pft=1, nsets=2, urban=("md",), lake=True)
    # per set: a pft-1 site (1 column), an urban site (5), a lake (1)
    assert lay.sizes() == {"g": 6, "t": 6, "l": 6, "c": 14, "p": 14}
    assert lay.site_of("c").tolist() == [0, 1, 1, 1, 1, 1, 2, 3, 4, 4, 4, 4, 4, 5]
    assert lay.set_of("p").tolist() == [0] * 7 + [1] * 7 and lay.set_of("l").tolist() == [0, 0, 0, 1, 1, 1]
    assert lay.first_child("l", "c").tolist() == [1, 2, 7, 8, 9, 14]
    assert lay.last_child("l", "p").tolist() == [1, 6, 7, 8, 13, 14]
    assert lay.parent("p", "l").tolist() == [1, 2, 2, 2, 2, 2, 3, 4, 5, 5, 5, 5, 5, 6]
    assert lay.parent("p", "c").tolist() == list(range(1, 15))
    assert lay.pft_of("p").tolist()[:7] == [1, -1, -1, -1, -1, -1, -1]  # per-pft rules: vegetated only
    assert lay.one_to_one("p", "c") and not lay.one_to_one("l", "c")
    assert dom.Layout([1, 2], nsets=2).one_to_one("l", "c")
    assert "urban md, column 5 of 5 (itype 75)" in lay.describe("c", 6)
    # a permutation of sites moves each site's columns as a block
    two = dom.Layout([], per_pft=2, nsets=2, urban=("hd",))
    assert two.row_permutation("c", [1, 0]).tolist() == [5, 6, 7, 8, 9, 0, 1, 2, 3, 4]
    assert two.row_permutation("l", [1, 0]).tolist() == [1, 0]


def test_urban_and_lake_topology(tmp_path):
    topo = [
        fake.var("lun_pp%itype", "begl:endl", "topology", "integer"),
        fake.var("lun_pp%urbpoi", "begl:endl", "topology", "logical"),
        fake.var("lun_pp%lakpoi", "begl:endl", "topology", "logical"),
        fake.var("lun_pp%ifspecial", "begl:endl", "topology", "logical"),
        fake.var("lun_pp%coli", "begl:endl", "topology", "integer"),
        fake.var("lun_pp%colf", "begl:endl", "topology", "integer"),
        fake.var("lun_pp%pfti", "begl:endl", "topology", "integer"),
        fake.var("lun_pp%pftf", "begl:endl", "topology", "integer"),
        fake.var("lun_pp%ncolumns", "begl:endl", "topology", "integer"),
        fake.var("col_pp%itype", "begc:endc", "topology", "integer"),
        fake.var("col_pp%landunit", "begc:endc", "topology", "integer"),
        fake.var("col_pp%gridcell", "begc:endc", "topology", "integer"),
        fake.var("col_pp%wtlunit", "begc:endc", "topology"),
        fake.var("col_pp%hydrologically_active", "begc:endc", "topology", "logical"),
        fake.var("col_pp%is_soil", "begc:endc", "topology", "logical"),
        fake.var("col_pp%is_lake", "begc:endc", "topology", "logical"),
        fake.var("veg_pp%column", "begp:endp", "topology", "integer"),
        fake.var("lun_es%taf", "begl:endl", "output", status="w"),
    ]
    inv = fake.inventory(tmp_path / "case", topo)
    inv.unit_test = SimpleNamespace(time_scalars=[], driver_args={})
    spec = dom.DomainSpec.from_table({"pfts": [1], "urban": ["md"], "lake": True, "init": "direct"}, tmp_path)
    d = dom.build_domain(inv, spec, tmp_path / "domain")
    v = d.direct["values"]
    assert d.direct["sizes"] == {"g": 3, "t": 3, "l": 3, "c": 7, "p": 7}
    assert d.direct["alloc"]["lun_es%taf"] == ([(1, 3)], "real")
    assert d.direct["alloc"]["a%prof"][0] == [(1, 7), (1, fake.NLEV)]
    assert v["lun_pp%itype"].tolist() == [1, 9, 5]
    assert v["lun_pp%urbpoi"].tolist() == [False, True, False]
    assert v["lun_pp%lakpoi"].tolist() == [False, False, True]
    assert v["lun_pp%ifspecial"].tolist() == [False, True, True]
    assert v["lun_pp%coli"].tolist() == [1, 2, 7] and v["lun_pp%colf"].tolist() == [1, 6, 7]
    assert v["lun_pp%pfti"].tolist() == [1, 2, 7] and v["lun_pp%pftf"].tolist() == [1, 6, 7]
    assert v["lun_pp%ncolumns"].tolist() == [1, 5, 1]
    assert v["col_pp%itype"].tolist() == [1, 71, 72, 73, 74, 75, 5]
    assert v["col_pp%landunit"].tolist() == [1, 2, 2, 2, 2, 2, 3] == v["col_pp%gridcell"].tolist()
    assert v["veg_pp%column"].tolist() == list(range(1, 8))
    assert v["veg_pp%itype"].tolist() == [1, 0, 0, 0, 0, 0, 0]  # urban and lake patches are bare
    w = v["col_pp%wtlunit"]
    assert w[0] == 1.0 and w[6] == 1.0 and np.isclose(w[1:6].sum(), 1.0)  # set_landunit_urban's weights
    assert v["col_pp%hydrologically_active"].tolist() == [True, False, False, False, False, True, False]
    assert v["col_pp%is_soil"].tolist() == [True] + [False] * 6
    assert v["col_pp%is_lake"].tolist() == [False] * 6 + [True]


def test_urban_spec_is_checked(tmp_path):
    with pytest.raises(dom.DomainError, match="densities"):
        dom.DomainSpec.from_table({"urban": ["high"]}, tmp_path)
    with pytest.raises(dom.DomainError, match="list of densities"):
        dom.DomainSpec.from_table({"urban": "md"}, tmp_path)
    with pytest.raises(dom.DomainError, match="give the sites"):
        dom.DomainSpec.from_table({"per_pft": 2}, tmp_path)
    assert dom.DomainSpec.from_table({"lake": True}, tmp_path).pfts == []
    spec = dom.DomainSpec.from_table({"pfts": [17], "landunit": "crop", "urban": ["MD", "tbd"]}, tmp_path)
    assert spec.landunit is dom.SiteKind.CROP and spec.urban == [dom.UrbanDensity.MD, dom.UrbanDensity.TBD]
    with pytest.raises(dom.DomainError, match="soil or crop"):
        dom.DomainSpec.from_table({"pfts": [1], "landunit": "urban"}, tmp_path)
    lay = dom.Layout.of(spec)
    assert [s.kind for s in lay.set_sites] == [dom.SiteKind.CROP, dom.SiteKind.URBAN, dom.SiteKind.URBAN]
    assert lay.set_sites[0].ctypes == (217,) and lay.set_sites[2].ltype == 7  # istcrop*100 + cft; isturb_tbd
    assert lay.is_kind("c", dom.SiteKind.URBAN).tolist() == [False] + [True] * 10


def test_type_codes_must_match_the_case(tmp_path):
    with pytest.raises(dom.DomainError, match="isturb_md = 3"):
        dom.check_type_codes({"isturb_md": 3})
