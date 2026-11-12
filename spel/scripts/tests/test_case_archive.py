"""`spel archive`: pack cases with their data, extract them elsewhere (spel/scripts/case_archive.py)."""

import io
import json
import tarfile

import pytest

from spel.scripts import case_archive as ca


def make_case(unittests, name, srcroot="/src/E3SM"):
    case = unittests / name
    (case / "build" / "CMakeFiles").mkdir(parents=True)
    (case / "build" / "libx.a").write_bytes(b"x" * 100)
    (case / "build-py-gfortran-OFF").mkdir()
    (case / "proptest" / "domain-abc").mkdir(parents=True)
    (case / ".spel-orig").mkdir()
    (case / ".spel-orig" / "Mod.F90").write_text("module m\nend module m\n")
    (case / "Mod.F90").write_text("module m\n! edited\nend module m\n")
    (case / "fut.pkl").write_bytes(b"pickle")
    (case / "check_config.sh").write_text("#!/bin/sh\n")
    (case / "check_config.sh").chmod(0o755)
    (case / ca.CASE_META_FILENAME).write_text(json.dumps({"case_name": name, "e3sm_srcroot": srcroot}))
    data = unittests / "input-data" / name
    data.mkdir(parents=True)
    (data / "spel-inputs0001.nc").write_bytes(b"nc-inputs")
    shared = unittests / "shared"
    shared.mkdir(exist_ok=True)
    (shared / "spel-outputs0001.nc").write_bytes(b"nc-outputs")
    (data / "spel-outputs0001.nc").symlink_to(shared / "spel-outputs0001.nc")  # as --link-inputs makes
    pt = unittests / "proptest-data" / name / "failures" / "s__c"
    pt.mkdir(parents=True)
    (pt / "example.json").write_text("{}")
    return case


def test_pack_cleans_and_extract_restores(tmp_path):
    src = tmp_path / "a" / "unit-tests"
    make_case(src, "canflux")
    out = io.StringIO()
    arc = ca.pack(["canflux"], tmp_path / "canflux.spel.tar.gz", src, proptest_data=True, yes=True, out=out)
    case = src / "canflux"
    assert not (case / "build").exists() and not (case / "build-py-gfortran-OFF").exists()
    assert not (case / "proptest").exists() and (case / ".spel-orig" / "Mod.F90").is_file()
    with tarfile.open(arc) as tar:
        names = tar.getnames()
        assert all(tar.getmember(n).isfile() or tar.getmember(n).isdir() for n in names)  # no links
    assert ca.MANIFEST in names and "cases/canflux/fut.pkl" in names
    assert "input-data/canflux/spel-outputs0001.nc" in names  # the symlink's target, as a file
    assert not any("build" in n.split("/")[2:3] for n in names if n.startswith("cases/"))
    manifest = ca.list_archive(arc, out=out)
    assert manifest["cases"]["canflux"]["e3sm_srcroot"] == "/src/E3SM"

    dest = tmp_path / "b" / "unit-tests"
    ca.extract(arc, dest, set_srcroot=tmp_path / "E3SM", out=out)
    assert (dest / "canflux" / "Mod.F90").read_text() == (case / "Mod.F90").read_text()
    assert (dest / "canflux" / ".spel-orig" / "Mod.F90").is_file()
    assert (dest / "canflux" / "check_config.sh").stat().st_mode & 0o100  # still executable
    out_nc = dest / "input-data" / "canflux" / "spel-outputs0001.nc"
    assert not out_nc.is_symlink() and out_nc.read_bytes() == b"nc-outputs"
    assert (dest / "proptest-data" / "canflux" / "failures" / "s__c" / "example.json").is_file()
    meta = json.loads((dest / "canflux" / ca.CASE_META_FILENAME).read_text())
    assert meta["e3sm_srcroot"] == str((tmp_path / "E3SM").resolve())
    assert meta["e3sm_srcroot_original"] == "/src/E3SM"

    with pytest.raises(ca.ArchiveError, match="exist already"):
        ca.extract(arc, dest, out=out)
    (dest / "canflux" / "stale.F90").write_text("x")
    ca.extract(arc, dest, force=True, out=out)
    assert not (dest / "canflux" / "stale.F90").exists()


def test_keep_builds_dry_run_and_several_cases(tmp_path):
    src = tmp_path / "unit-tests"
    make_case(src, "a")
    make_case(src, "b")
    out = io.StringIO()
    assert ca.pack(["a", "b"], tmp_path / "x.tar", src, dry_run=True, out=out) is None
    assert (src / "a" / "build").is_dir() and "delete" in out.getvalue()
    arc = ca.pack(["a", "b"], tmp_path / "x.tar", src, keep_builds=True, out=out)
    assert (src / "a" / "build").is_dir()  # kept, but not archived
    with tarfile.open(arc) as tar:
        assert not any(n.startswith("cases/a/build") for n in tar.getnames())
        assert not any(n.startswith("proptest-data/") for n in tar.getnames())  # not asked for
    dest = tmp_path / "dest"
    assert ca.extract(arc, dest, cases=["b"], out=out) == ["b"]
    assert (dest / "b" / "fut.pkl").is_file() and not (dest / "a").exists()
    with pytest.raises(ca.ArchiveError, match="no case c"):
        ca.extract(arc, dest, cases=["c"], out=out)


def test_pack_refuses_without_confirmation_and_bad_names(tmp_path, monkeypatch):
    src = tmp_path / "unit-tests"
    make_case(src, "a")
    monkeypatch.setattr("sys.stdin", io.StringIO())  # not a terminal
    with pytest.raises(ca.ArchiveError, match="confirmation"):
        ca.pack(["a"], tmp_path / "a.tar.gz", src, out=io.StringIO())
    assert (src / "a" / "build").is_dir()
    with pytest.raises(ca.ArchiveError, match="not a unit-test case"):
        ca.pack(["input-data"], tmp_path / "x.tar.gz", src, yes=True, out=io.StringIO())
    with pytest.raises(ca.ArchiveError, match=".tar.gz"):
        ca.pack(["a"], tmp_path / "a.zip", src, yes=True, out=io.StringIO())


@pytest.mark.parametrize("name,link", [("../evil", False), ("/etc/evil", False), ("cases/a/../../evil", False),
                                       ("other/a/x", False), ("cases/a/link", True)])
def test_extract_refuses_unsafe_members(tmp_path, name, link):
    arc = tmp_path / "bad.tar"
    with tarfile.open(arc, "w") as tar:
        data = json.dumps({"format": ca.FORMAT, "created": "now", "cases": {"a": {}}}).encode()
        info = tarfile.TarInfo(ca.MANIFEST)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
        info = tarfile.TarInfo(name)
        if link:
            info.type, info.linkname = tarfile.SYMTYPE, "/etc/passwd"
            tar.addfile(info)
        else:
            info.size = 1
            tar.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(ca.ArchiveError, match="refusing"):
        ca.extract(arc, tmp_path / "dest", out=io.StringIO())
    assert not (tmp_path / "evil").exists() and not (tmp_path / "dest").exists()
