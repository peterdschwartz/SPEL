"""The synthetic domain files, built from an inventory without a case behind it (tests/fake.py)."""

from types import SimpleNamespace

import numpy as np
import pytest

from spel.proptest import domain as dom
from spel.proptest.hierarchy import (Crop, Gridcell, HierarchyError, Lake, Layout, Patch, Soil, Topounit,
                                     Urban)
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
            "gridcell": fake.vegetated(1, 13, copies=2),
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
        dom.build_domain(inv, dom.DomainSpec([Gridcell.of(Soil(1))]), tmp_path / "domain")


def test_dimension_without_a_value_names_the_fix(tmp_path):
    inv = fake.inventory(
        tmp_path / "case", [fake.var("a%z", "begc:endc,1:nlevmissing", "input")]
    )
    with pytest.raises(dom.DomainError, match=r"nlevmissing.*\[domain.dims\]"):
        dom.build_domain(inv, dom.DomainSpec([Gridcell.of(Soil(1))]), tmp_path / "domain")


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
        {"gridcell": fake.vegetated(1, 13, copies=2), "dims": {"nlevsno": 5}, "time": {"jday_mod": 10}, "init": "direct"},
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
        dom.DomainSpec.from_table({"gridcell": fake.vegetated(1), "init": "memory"}, tmp_path)


def test_urban_and_lake_landunits_have_their_own_columns():
    lay = Layout([Gridcell.of(Soil(1)), Gridcell.of(Urban("md")), Gridcell.of(Lake())], nsets=2)
    # per set: a pft-1 gridcell (1 column), an urban one (5), a lake (1)
    assert lay.sizes() == {"g": 6, "t": 6, "l": 6, "c": 14, "p": 14}
    assert lay.gridcell_of("c").tolist() == [0, 1, 1, 1, 1, 1, 2, 3, 4, 4, 4, 4, 4, 5]
    assert lay.set_of("p").tolist() == [0] * 7 + [1] * 7 and lay.set_of("l").tolist() == [0, 0, 0, 1, 1, 1]
    assert lay.first_child("l", "c").tolist() == [1, 2, 7, 8, 9, 14]
    assert lay.last_child("l", "p").tolist() == [1, 6, 7, 8, 13, 14]
    assert lay.parent("p", "l").tolist() == [1, 2, 2, 2, 2, 2, 3, 4, 5, 5, 5, 5, 5, 6]
    assert lay.parent("p", "c").tolist() == list(range(1, 15))
    assert lay.itype("p").tolist()[:7] == [1, 0, 0, 0, 0, 0, 0]
    assert lay.one_to_one("p", "c") and not lay.one_to_one("l", "c")
    assert Layout([Gridcell.of(Soil(1)), Gridcell.of(Soil(2))], nsets=2).one_to_one("l", "c")
    assert lay.describe("c", 6) == "set 0, urban md, urban md, column road_perv"
    # a permutation of gridcells moves each gridcell's columns as a block
    two = Layout([Gridcell.of(Urban("hd"), copies=2)], nsets=2)
    assert two.row_permutation("c", [1, 0]).tolist() == [5, 6, 7, 8, 9, 0, 1, 2, 3, 4]
    assert two.row_permutation("l", [1, 0]).tolist() == [1, 0]


def test_soil_column_with_several_pfts_and_weights():
    lay = Layout([Gridcell.of(Soil(Patch(1, 0.6), 13, 4), Urban("md", weight=0.2), name="mixed")])
    assert lay.sizes() == {"g": 1, "t": 1, "l": 2, "c": 6, "p": 8}
    assert lay.children("c", "p").tolist() == [3, 1, 1, 1, 1, 1]  # one soil column, three pfts
    assert np.allclose(lay.weight("p", "c")[:3], [0.6, 0.2, 0.2])  # the missing weights share the rest
    assert np.allclose(lay.weight("l", "t"), [0.8, 0.2])
    assert np.isclose(lay.weight("p", "g").sum(), 1.0)  # every level sums to 1 in its gridcell
    assert lay.parent("p", "c").tolist()[:3] == [1, 1, 1]
    assert "pft 13" in lay.describe("p", 2) and "column soil" in lay.describe("p", 2)


def test_crop_has_one_column_per_cft():
    lu = Crop(Patch(17, 0.25), 19)
    assert [c.itype for c in lu.columns] == [217, 219] and [c.weight for c in lu.columns] == [0.25, None]
    assert Layout([Gridcell.of(lu)]).weight("c", "l").tolist() == [0.25, 0.75]


def test_topounits_numbers_activity_and_downhill():
    hill = Gridcell(name="hill", topounits=(
        Topounit((Soil(1, 13),), weight=0.5, elevation=300.0),
        Topounit((Soil(13),), weight=0.5, elevation=150.0),
        Topounit((Soil(1),), weight=0.0, elevation=50.0),
    ))
    lay = Layout([Gridcell.of(Lake()), hill])
    assert lay.sizes() == {"g": 2, "t": 4, "l": 4, "c": 4, "p": 5}
    assert lay.topo_index().tolist() == [1, 1, 2, 3]  # topo_grc_ind: the number within the gridcell
    assert lay.active("t").tolist() == [True, True, True, False]  # weight 0: inactive
    assert lay.active("p").tolist() == [True, True, True, True, False]
    assert lay.parent("t", "g").tolist() == [1, 2, 2, 2]
    assert lay.first_child("g", "t").tolist() == [1, 2] and lay.last_child("g", "t").tolist() == [1, 4]
    # set_topounit: the nearest lower topounit; the lowest keeps the previous dn_index (ELM's loop)
    assert lay.downhill().tolist() == [-9999, 3, 4, 4]
    assert lay.max_topounits == 3
    li = lay.landunit_indices("t")
    assert li.shape == (12, 4) and li[0].tolist() == [-9999, 2, 3, 4] and li[4].tolist() == [1, -9999, -9999, -9999]
    assert lay.landunit_indices("g")[0].tolist() == [-9999, 2]  # the gridcell's first soil landunit
    assert "topounit 2/3" in lay.describe("p", 4)


def test_weights_and_structure_are_checked():
    with pytest.raises(HierarchyError, match="sum to 1"):
        Layout([Gridcell.of(Soil(Patch(1, 0.7), Patch(2, 0.7)))])
    with pytest.raises(HierarchyError, match="sum to 1"):
        Layout([Gridcell.of(Soil(Patch(1, 1.0), 2))])  # nothing left for pft 2
    with pytest.raises(HierarchyError, match="no patch"):
        Layout([Gridcell.of(Soil())])
    with pytest.raises(HierarchyError, match="elevation"):
        Layout([Gridcell((Topounit((Soil(1),)), Topounit((Soil(1),))))]).downhill()


def test_gridcell_tables_round_trip(tmp_path):
    cells = [
        Gridcell.of(Soil(Patch(1, 0.6), 13, weight=0.7), Urban("md", weight=0.2), Lake(), copies=2, name="mixed"),
        Gridcell.of(Crop(Patch(17, 0.3), 19), Urban("hd", roof=0.4, road_perv=0.25)),
        Gridcell((Topounit((Soil(1),), weight=0.4, elevation=10.0), Topounit((Lake(),), weight=0.6, elevation=5.0)),
                 name="hill"),
    ]
    spec = dom.DomainSpec(cells, nsets=2)
    table = spec.to_table()
    back = dom.DomainSpec.from_table(table, tmp_path)
    assert back.gridcells == cells and back.to_table() == table
    assert table["gridcell"][0]["landunits"][0] == {"kind": "soil", "patches": [{"pft": 1, "weight": 0.6}, 13],
                                                  "weight": 0.7}
    assert table["gridcell"][1]["landunits"][1] == {"kind": "urban", "density": "hd", "roof": 0.4, "road_perv": 0.25}
    with pytest.raises(dom.DomainError, match="unknown keys"):
        dom.DomainSpec.from_table({"pfts": [1]}, tmp_path)
    with pytest.raises(dom.DomainError, match="density"):
        dom.DomainSpec.from_table({"gridcell": [{"landunits": [{"kind": "urban", "density": "high"}]}]}, tmp_path)
    with pytest.raises(dom.DomainError, match="landunits.*or.*topounits"):
        dom.DomainSpec.from_table({"gridcell": [{"name": "x"}]}, tmp_path)


def test_urban_lake_and_topounit_topology(tmp_path):
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
        fake.var("lun_pp%wttopounit", "begl:endl", "topology"),
        fake.var("col_pp%itype", "begc:endc", "topology", "integer"),
        fake.var("col_pp%landunit", "begc:endc", "topology", "integer"),
        fake.var("col_pp%gridcell", "begc:endc", "topology", "integer"),
        fake.var("col_pp%wtlunit", "begc:endc", "topology"),
        fake.var("col_pp%hydrologically_active", "begc:endc", "topology", "logical"),
        fake.var("col_pp%is_soil", "begc:endc", "topology", "logical"),
        fake.var("col_pp%is_lake", "begc:endc", "topology", "logical"),
        fake.var("veg_pp%column", "begp:endp", "topology", "integer"),
        fake.var("veg_pp%wtgcell", "begp:endp", "topology"),
        fake.var("top_pp%topo_grc_ind", "begt:endt", "topology", "integer"),
        fake.var("top_pp%active", "begt:endt", "topology", "logical"),
        fake.var("top_pp%landunit_indices", "1:max_lunit,begt:endt", "topology", "integer"),
        fake.var("grc_pp%ntopounits", "begg:endg", "topology", "integer"),
        fake.var("lun_es%taf", "begl:endl", "output", status="w"),
    ]
    inv = fake.inventory(tmp_path / "case", topo)
    inv.parameters["max_lunit"] = 12
    inv.unit_test = SimpleNamespace(time_scalars=[], driver_args={})
    cells = [Gridcell.of(Soil(1, 13), name="veg"), Gridcell.of(Urban("md")),
             Gridcell((Topounit((Lake(),), weight=0.6), Topounit((Soil(1),), weight=0.4)), name="two")]
    spec = dom.DomainSpec(cells, init="direct")
    d = dom.build_domain(inv, spec, tmp_path / "domain")
    v = d.direct["values"]
    assert d.direct["sizes"] == {"g": 3, "t": 4, "l": 4, "c": 8, "p": 9}
    assert d.direct["alloc"]["lun_es%taf"] == ([(1, 4)], "real")
    assert d.direct["alloc"]["a%prof"][0] == [(1, 9), (1, fake.NLEV)]
    assert v["lun_pp%itype"].tolist() == [1, 9, 5, 1]
    assert v["lun_pp%urbpoi"].tolist() == [False, True, False, False]
    assert v["lun_pp%lakpoi"].tolist() == [False, False, True, False]
    assert v["lun_pp%ifspecial"].tolist() == [False, True, True, False]
    assert v["lun_pp%coli"].tolist() == [1, 2, 7, 8] and v["lun_pp%colf"].tolist() == [1, 6, 7, 8]
    assert v["lun_pp%pfti"].tolist() == [1, 3, 8, 9] and v["lun_pp%pftf"].tolist() == [2, 7, 8, 9]
    assert v["lun_pp%ncolumns"].tolist() == [1, 5, 1, 1]
    assert v["col_pp%itype"].tolist() == [1, 71, 72, 73, 74, 75, 5, 1]
    assert v["col_pp%landunit"].tolist() == [1, 2, 2, 2, 2, 2, 3, 4]
    assert v["col_pp%gridcell"].tolist() == [1, 2, 2, 2, 2, 2, 3, 3]
    assert v["veg_pp%column"].tolist() == [1, 1, 2, 3, 4, 5, 6, 7, 8]
    assert v["veg_pp%itype"].tolist() == [1, 13, 0, 0, 0, 0, 0, 0, 1]  # urban and lake patches are bare
    assert np.allclose(v["veg_pp%wtgcell"][:2], [0.5, 0.5]) and np.allclose(v["veg_pp%wtgcell"][7:], [0.6, 0.4])
    assert np.allclose(v["lun_pp%wttopounit"], 1.0)
    w = v["col_pp%wtlunit"]
    assert w[0] == 1.0 and w[6] == 1.0 and np.isclose(w[1:6].sum(), 1.0)  # set_landunit_urban's weights
    assert v["col_pp%hydrologically_active"].tolist() == [True, False, False, False, False, True, False, True]
    assert v["col_pp%is_soil"].tolist() == [True] + [False] * 6 + [True]
    assert v["col_pp%is_lake"].tolist() == [False] * 6 + [True, False]
    assert v["top_pp%topo_grc_ind"].tolist() == [1, 1, 1, 2]
    assert v["top_pp%landunit_indices"].shape == (12, 4) and v["top_pp%landunit_indices"][4].tolist() == [
        -9999, -9999, 3, -9999]
    assert v["grc_pp%ntopounits"].tolist() == [1, 1, 2]
    assert d.topology["lun_pp%itype"].tolist() == [1, 9, 5, 1]  # what BuiltDomain compares against


def test_domain_decides_the_topounit_constants_and_pins_elevations(tmp_path):
    from spel.proptest.inventory import ConstInfo

    inv = fake.inventory(tmp_path / "case", [fake.var("top_pp%elevation", "begt:endt", "input"),
                                             fake.var("top_pp%downhill_ti", "begt:endt", "topology", "integer")])
    inv.constants["max_topounits"] = ConstInfo("max_topounits", "topounit_varcon", "integer", 0, [], None)
    inv.constants["has_topounit"] = ConstInfo("has_topounit", "topounit_varcon", "logical", 0, [], None)
    inv.unit_test = SimpleNamespace(time_scalars=[], driver_args={})
    hill = Gridcell((Topounit((Soil(1),), elevation=200.0), Topounit((Soil(1),), elevation=100.0)))
    d = dom.build_domain(inv, dom.DomainSpec([hill], init="direct"), tmp_path / "domain")
    assert d.constants["max_topounits"] == 2 and d.constants["has_topounit"] == 1
    assert d.pinned["top_pp%elevation"].tolist() == [200.0, 100.0]
    assert d.direct["values"]["top_pp%elevation"].tolist() == [200.0, 100.0]
    assert d.direct["values"]["top_pp%downhill_ti"].tolist() == [2, 2]
    with pytest.raises(dom.DomainError, match="follow from the declared gridcells"):
        dom.build_domain(inv, dom.DomainSpec([hill], constants={"max_topounits": 1}), tmp_path / "d2")


def test_type_codes_must_match_the_case(tmp_path):
    with pytest.raises(dom.DomainError, match="isturb_md = 3"):
        dom.DomainSpec([Gridcell.of(Soil(1))]).layout({"isturb_md": 3})
