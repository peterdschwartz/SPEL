"""Drawn domains (domains.py): the draw, its rules, shrinking, and the runner's loop over domains."""

import io
import tomllib
from pathlib import Path

import numpy as np
import pytest

from spel.proptest import domain as dom
from spel.proptest.domains import DomainDraw, DrawError, shrink_candidates, size
from spel.proptest.hierarchy import Gridcell, LandunitKind, Layout, Lake, Soil, Topounit

ALL = {"soil": 0.8, "crop": 0.3, "urban": 0.3, "lake": 0.3, "wetland": 0.3, "glacier": 0.3,
       "glacier_mec": 0.3, "polygon": 0.3}


def test_draws_are_reproducible_and_follow_elm_rules():
    d = DomainDraw(count=30, gridcells=(1, 4), topounits=(1, 4), kinds=ALL, zero_topounit=0.3, zero_landunit=0.4,
                   zero_patch=0.3, pfts=(1, 2, 12, 13), cfts=(17, 19, 23), glcmask=0.5, icemask=0.3, ties=0.3)
    a, b = list(d.domains(7)), list(d.domains(7))
    assert a == b and a != list(d.domains(8))
    assert list(DomainDraw(**{**d.__dict__, "count": 3}).domains(7)) == a[:3]  # domain k does not depend on count
    kinds, zero_lu, zero_t, ties = set(), 0, 0, 0
    for cells in a:
        lay = Layout(cells, nsets=2)  # weights sum to 1, one landunit per type
        for g in cells:
            elev = [t.elevation for t in g.topounits]
            ties += len(elev) - len(set(elev))
            for t in g.topounits:
                zero_t += t.weight == 0.0
                assert any(lu.kind is LandunitKind.SOIL for lu in t.landunits)  # ELM always has one
                assert any(lu.weight != 0.0 for lu in t.landunits)
                for lu in t.landunits:
                    kinds.add(lu.kind)
                    if lu.weight == 0.0:
                        zero_lu += 1
                        assert lu.kind not in (LandunitKind.LAKE, LandunitKind.WETLAND, LandunitKind.GLACIER)
                        assert lu.kind is not LandunitKind.GLACIER_MEC or g.glcmask
        assert lay.active("g").all()
    assert kinds == set(LandunitKind) and zero_lu > 10 and zero_t > 3 and ties > 0


def test_draw_table_round_trip_and_errors():
    d = DomainDraw(count=3, kinds={"soil": 1.0, "wetland": 0.5}, pfts=(1, 13), zero_landunit=0.2, elevation=(10.0, 20.0))
    assert DomainDraw.from_table(d.to_table()) == d
    spec = dom.DomainSpec.from_table({"draw": d.to_table(), "nsets": 2}, Path("."))
    assert spec.draw == d and spec.gridcells == next(d.domains(0))  # the sample: domain 1 of seed 0
    assert "gridcell" not in spec.to_table() and spec.to_table()["draw"] == d.to_table()
    assert spec.with_gridcells(spec.gridcells).draw is None
    with pytest.raises(DrawError, match="not landunit kinds"):
        DomainDraw(kinds={"forest": 1.0})
    with pytest.raises(DrawError, match="cfts"):
        DomainDraw(kinds={"soil": 1.0, "crop": 0.5})
    with pytest.raises(dom.DomainError, match="unknown keys"):
        dom.DomainSpec.from_table({"draw": {"gridcels": [1, 2]}}, Path("."))
    with pytest.raises(dom.DomainError, match="not both"):
        dom.DomainSpec.from_table({"draw": {}, "gridcell": [{"landunits": [{"kind": "lake"}]}]}, Path("."))


def test_shrink_candidates_are_smaller_and_valid():
    g = Gridcell((Topounit((Soil(1, 13, weight=0.6), Lake(weight=0.4)), weight=0.7, elevation=10.0),
                  Topounit((Lake(weight=1.0), Soil(1, weight=0.0)), weight=0.3, elevation=5.0)), name="a")
    h = Gridcell.of(Soil(2), name="b")
    cands = list(shrink_candidates([g, h]))
    assert cands[0] == [h] and cands[1] == [g]  # whole gridcells first
    for c in cands:
        Layout(c)
        assert size(c) < size([g, h]) or c != [g, h]
    # dropping the soil landunit of a topounit puts ELM's virtual one back
    assert all(any(lu.kind is LandunitKind.SOIL for lu in t.landunits) for c in cands for gc in c for t in gc.topounits)
    assert not list(shrink_candidates([Gridcell.of(Soil(1))]))  # nothing smaller


def test_runner_draws_domains_and_shrinks_a_failure(tmp_path):
    from spel.proptest import runner
    from spel.proptest import spec as sp
    from spel.proptest.tests import fake

    pytest.importorskip("hypothesis")
    case = tmp_path / "case"
    inv = fake.inventory(case)
    draw = {"count": 2, "gridcells": [2, 3], "topounits": [1, 3], "kinds": {"soil": 1.0, "lake": 0.5},
            "pfts": [1, 2, 13], "max_pfts": 2, "elevation": [0.0, 100.0]}
    data = fake.spec_data(case, property=[{"name": "small", "expr": "y(p) < -10.0"}])  # fails on any domain
    data["domain"] = {"draw": draw, "nsets": 1}
    data["settings"]["max_examples"] = 3
    s = sp.parse(data, tmp_path / "t.proptest.toml", case, inv)
    out = io.StringIO()
    res = runner.run(s, inv, settings={"seed": 0}, out=out, backend=fake.backends())
    text = out.getvalue()
    assert len(res) == 2 and text.count("== domain") == 2
    failed = [o for o in res if not o.passed]
    assert failed and all("shrunk" in o.domain for o in failed), text
    saved = Path(failed[0].failure)
    assert saved.name.endswith("__d" + failed[0].domain.split("/")[0].split()[-1])
    cells = tomllib.loads((saved / runner.DOMAIN_FILE).read_text())["gridcell"]
    shrunk = [Gridcell.from_table(c, "c") for c in cells]
    assert Layout(shrunk).sizes() == {"g": 1, "t": 1, "l": 1, "c": 1, "p": 1}  # any patch fails: the smallest domain
    assert "shrunk (" in text and "saved:" in text
