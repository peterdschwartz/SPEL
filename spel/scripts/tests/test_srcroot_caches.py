import json
from pathlib import Path

import pytest

import spel.scripts.analysis_cache as ac
from spel.scripts.config import CASE_META_FILENAME, REFERENCE_META_FILENAME
from spel.scripts.validate_access import provenance
from spel.srcroot_override import ENV_VAR, apply_srcroot, find_srcroot


def make_checkout(path: Path) -> Path:
    (path / "components/elm/src").mkdir(parents=True)
    return path


@pytest.mark.parametrize(
    "argv, expected",
    [
        (["create", "-s", "x"], None),
        (["create", "--srcroot", "/a", "-s", "x"], "/a"),
        (["--srcroot=/a", "create"], "/a"),
        (["create", "--srcroot=/a", "--srcroot", "/b"], "/b"),
        (["run", "case", "--", "--srcroot", "/a"], None),
    ],
)
def test_find_srcroot(argv, expected):
    assert find_srcroot(argv) == expected


def test_apply_srcroot_expands_and_validates(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv(ENV_VAR, raising=False)
    root = make_checkout(tmp_path / "E3SM")
    assert apply_srcroot(["create", "--srcroot=~/E3SM/"]) == root
    import os

    assert os.environ[ENV_VAR] == str(root)
    with pytest.raises(SystemExit):
        apply_srcroot(["--srcroot", str(tmp_path / "missing")])


def test_cache_dir_per_srcroot(tmp_path, monkeypatch):
    monkeypatch.setattr(ac, "unittests_dir", tmp_path)
    ref, dev = tmp_path / "E3SM", tmp_path / "dev_E3SM"
    assert ac.cache_dir(ref) != ac.cache_dir(dev)
    assert ac.cache_dir(ref) == ac.cache_dir(Path(str(ref) + "/"))
    assert ac.cache_dir(ref).parent.name.startswith("E3SM-")
    # same directory name, different checkouts
    assert ac.cache_dir(tmp_path / "a/E3SM") != ac.cache_dir(tmp_path / "b/E3SM")


def write_legacy(tmp_path, srcroot):
    old = tmp_path / ".spel-cache" / ac.DRIVER_ROUTINE
    old.mkdir(parents=True)
    (old / ac.META_FILE).write_text(json.dumps({"e3sm_srcroot": str(srcroot)}))
    return old


def test_migrate_legacy_cache(tmp_path, monkeypatch):
    dev = tmp_path / "dev_E3SM"
    monkeypatch.setattr(ac, "unittests_dir", tmp_path)
    monkeypatch.setattr(ac, "E3SM_SRCROOT", dev)
    old = write_legacy(tmp_path, dev)
    assert ac.migrate_legacy_cache()
    assert not old.exists()
    assert (ac.cache_dir() / ac.META_FILE).is_file()
    assert [p for p, _ in ac.list_caches()] == [ac.cache_dir()]


def test_legacy_cache_of_other_checkout_is_left_alone(tmp_path, monkeypatch):
    monkeypatch.setattr(ac, "unittests_dir", tmp_path)
    monkeypatch.setattr(ac, "E3SM_SRCROOT", tmp_path / "E3SM")
    old = write_legacy(tmp_path, tmp_path / "dev_E3SM")
    assert not ac.migrate_legacy_cache()
    assert old.exists()


def test_provenance_flags_cross_checkout_comparison(tmp_path):
    case, data = tmp_path / "case", tmp_path / "data"
    case.mkdir()
    data.mkdir()
    (case / CASE_META_FILENAME).write_text(json.dumps({"e3sm_srcroot": "/x/dev_E3SM"}))
    assert any("origin not recorded" in line for line in provenance(case, data))
    ref = {"e3sm_srcroot": "/x/E3SM", "e3sm": {"branch": "master", "commit": "34d78535b7aa"}}
    (data / REFERENCE_META_FILENAME).write_text(json.dumps(ref))
    lines = provenance(case, data)
    assert "master@34d78535b7" in lines[1]
    assert lines[-1].startswith("NOTE: different E3SM checkouts")
    ref["e3sm_srcroot"] = "/x/dev_E3SM"
    (data / REFERENCE_META_FILENAME).write_text(json.dumps(ref))
    assert not provenance(case, data)[-1].startswith("NOTE")
