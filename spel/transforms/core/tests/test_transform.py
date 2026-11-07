"""The transform protocol, driver and registry on a toy transform (no analysis needed)."""
import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from spel.transforms.core import driver, registry
from spel.transforms.core.edits import EditConflict, insert, replace
from spel.transforms.core.source import Sources
from spel.transforms.core.transform import Change, Context, Refusal, Transform

SRC = "subroutine s\n  x = 1\n  y = 2\nend subroutine s"


@dataclass
class TagOptions:
    names: list[str] = field(default_factory=lambda: ["x"], metadata={"help": "variables to tag"})
    loud: bool = False
    limit: int = 0


class Tag(Transform):
    """Matches: assignments to `names`. Changes: a comment above. Refuses: `y`."""

    name = "test-tag"
    summary = "tag assignments"
    Options = TagOptions

    def select(self, ctx, opts):
        lines = ctx.lines(ctx.srcroot / "m.F90")
        for ln, l in enumerate(lines):
            if "=" in l:
                yield ln, l.split("=")[0].strip()

    def plan(self, ctx, opts, match):
        ln, var = match
        if var == "y":
            return Refusal(f"line {ln + 1}", "y is never tagged")
        if var in opts.names:
            text = "  ! TAGGED" + ("!" if opts.loud else "")
            return Change(ctx.srcroot / "m.F90", [insert(ln, text)], where=f"line {ln + 1}", why=var)


@pytest.fixture
def ctx(tmp_path):
    (tmp_path / "m.F90").write_text(SRC)
    return Context(None, tmp_path, Sources())


def test_run_plans_reports_and_writes(ctx, tmp_path):
    out = driver.Memory()
    res = driver.run(Tag(), ctx, TagOptions(loud=True), sink=out)
    assert out.files[tmp_path / "m.F90"].split("\n")[1] == "  ! TAGGED!"
    assert [str(r) for r in res.plan.refusals] == ["line 3: y is never tagged"]
    assert "1 changes in 1 files, 1 refused" in res.plan.report()
    assert "+  ! TAGGED!" in res.plan.diff(ctx.sources)
    assert (tmp_path / "m.F90").read_text() == SRC  # Memory sink: the source is untouched


def test_dry_run_writes_nothing_and_outdir_flattens(ctx, tmp_path):
    assert driver.run(Tag(), ctx, sink=driver.InPlace(), dry_run=True).written == []
    assert (tmp_path / "m.F90").read_text() == SRC
    res = driver.run(Tag(), ctx, sink=driver.OutDir(tmp_path / "out"), manifest=True)
    assert res.written == [tmp_path / "out" / "m.F90"]
    runs = json.loads((tmp_path / "out" / driver.MANIFEST).read_text())["runs"]
    assert runs[0]["transform"] == "test-tag" and runs[0]["options"]["names"] == ["x"]
    assert runs[0]["refused"] == ["line 3: y is never tagged"]


def test_conflicting_changes_are_an_error(ctx, tmp_path):
    class Clash(Tag):
        def plan(self, ctx, opts, match):
            return Change(tmp_path / "m.F90", [replace(1, f"  x = {match[0]}")])  # differ per match

    with pytest.raises(EditConflict):
        driver.run(Clash(), ctx, dry_run=True)


def test_options_become_flags_and_spec_keys(tmp_path, monkeypatch):
    monkeypatch.setitem(registry.REGISTRY, "test-tag", Tag)
    monkeypatch.setattr(registry, "_loaded", True)
    args, steps = registry._steps_from_args(["test-tag", "--names", "a", "b", "--loud", "--limit", "3", "--diff"])
    assert args.diff and steps == [(Tag, TagOptions(names=["a", "b"], loud=True, limit=3))]
    _, steps = registry._steps_from_args(["test-tag"])
    assert steps[0][1] == TagOptions()
    spec = tmp_path / "s.toml"
    spec.write_text('[[step]]\ntransform = "test-tag"\noptions = { names = ["z"] }\n'
                    '[[step]]\ntransform = "test-tag"\n')
    assert registry.load_spec(spec) == [(Tag, TagOptions(names=["z"])), (Tag, TagOptions())]
    spec.write_text('transform = "test-tag"\n[options]\nbogus = 1\n')
    with pytest.raises(ValueError, match="unknown option 'bogus'"):
        registry.load_spec(spec)


def test_builtins_register_and_describe():
    reg = registry.load_builtins()
    for name in ("collapse-bounds", "unblock", "section-loops", "defer-checks", "acc-locals"):
        assert name in reg
        assert reg[name].describe().startswith(f"{name}: ")
