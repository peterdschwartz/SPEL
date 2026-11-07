import pytest

from spel.transforms.nutrient import groups
from spel.transforms.nutrient.common import Family, Variant, match_fields
from spel.transforms.nutrient.extract import ExtractError


def family(n, p):
    v = lambda x: Variant(x, "real", 1, "begp:endp", "pointer", 0)
    fam = Family("vegetation_nutrient_state", {"c": "c_t", "n": "n_t", "p": "p_t"})
    fam.fields = match_fields({"c": {}, "n": {x: v(x) for x in n}, "p": {x: v(x) for x in p}})
    return fam


def test_group_edits(monkeypatch):
    monkeypatch.setattr(groups, "GROUPS", {"retrans": {"vegetation_nutrient_state": ["retrans"]}})
    (fe,) = groups.group_edits([family(["retransn"], ["retransp"])])
    assert (fe.type_name, fe.component, fe.elements) == ("vegetation_retrans_state", "retrans", ("n", "p"))
    assert {t.type_name: t.moved for t in fe.types} == {
        "n_t": {"retransn": "retrans%pool"},
        "p_t": {"retransp": "retrans%pool"},
    }


def test_group_edits_missing_field(monkeypatch):
    monkeypatch.setattr(groups, "GROUPS", {"retrans": {"vegetation_nutrient_state": ["retrans"]}})
    with pytest.raises(ExtractError, match="no N/P-only fields"):
        groups.group_edits([family(["retransn"], [])])


def test_load_groups_replaces_builtin(tmp_path):
    f = tmp_path / "g.toml"
    f.write_text('[retrans]\nvegetation_nutrient_state = ["retrans"]\n')
    (fe,) = groups.group_edits([family(["retransn"], ["retransp"])], groups.load_groups(f))
    assert fe.component == "retrans"


def test_load_groups_rejects_bad_shape(tmp_path):
    f = tmp_path / "g.toml"
    f.write_text('[retrans]\nvegetation_nutrient_state = "retrans"\n')
    with pytest.raises(ExtractError, match="lists of base names"):
        groups.load_groups(f)
