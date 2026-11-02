"""Pipelines (step n+1 plans on step n's output) and the snippet harness (core/testing.py)."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from spel.scripts.fortran_parser.scope_walk import AssignEvent
from spel.transforms.collapse_bounds import CollapseBounds
from spel.transforms.core.capture import VirtualFS
from spel.transforms.core.edits import replace
from spel.transforms.core.pipeline import AnalysisOverlay, PipelineError, line_map, preprocessed_lines
from spel.transforms.core.testing import check_pipeline, check_transform, snippet_context
from spel.transforms.core.transform import Change, NoOptions, Plan, Transform

BEFORE = """\
module m
contains
subroutine s(x, n)
  real(r8), intent(inout) :: x(:,:)
  integer, intent(in) :: n
  integer :: i, j
  !$acc parallel loop collapse(2) default(present)
  do j = 1, n
     do i = 1, size(x, 1)
        x(i, j) = 0._r8
     end do
  end do
end subroutine s
end module m
"""

HOISTED = """\
module m
contains
subroutine s(x, n)
  real(r8), intent(inout) :: x(:,:)
  integer, intent(in) :: n
  integer :: i, j
  integer :: spel_lb1, spel_lb2, spel_ub1, spel_ub2  ! spel collapse-bounds: host copies of collapsed-loop inquiry bounds
  spel_ub2 = size(x,1)  ! spel collapse-bounds
  !$acc parallel loop collapse(2) default(present)
  do j = 1, n
     do i = 1, spel_ub2
        x(i, j) = 0._r8
     end do
  end do
end subroutine s
end module m
"""


class TagAssignments(Transform):
    """Matches: the assignments of the record. Changes: `! <ln>` (0-based line) appended."""

    name = "test-tag-assignments"
    summary = "tag assignments with their recorded line"
    Options = NoOptions

    def build(self, ctx, opts):
        plan = Plan(self.name)
        for sub in ctx.routines():
            lines = ctx.lines(sub)
            edits = [replace(e.ln, f"{lines[e.ln]}  ! {e.ln}") for e in sub.record.events
                     if isinstance(e, AssignEvent)]
            plan.add(Change(ctx.path(sub), edits, where=sub.id))
        return plan


def test_check_transform_runs_reparses_and_replans(tmp_path):
    res = check_transform(CollapseBounds, BEFORE, HOISTED, tmp_path)
    assert "1 changes in 1 files" in res.plan.report()
    with pytest.raises(AssertionError, match="collapse-bounds gave"):
        check_transform(CollapseBounds, BEFORE, BEFORE, tmp_path / "wrong")


def test_snippet_context_analyzes_like_the_analysis(tmp_path):
    ctx = snippet_context(BEFORE, tmp_path)
    (sub,) = ctx.fut.subroutine_dict.values()
    assert (sub.id, sub.startline, sub.endline, sub.last_decl_ln) == ("m::s", 2, 12, 5)
    assert sub.sub_lines[0].line == "subroutine s(x, n)" and sub.sub_lines[-1].ln == 12
    assert sub.record_access.args["x"]  # access maps built
    assert [l.node.lineno for l in sub.record.loop_nest] == [7]


def test_pipeline_step_plans_on_previous_output(tmp_path):
    # step 2 sees step 1's lines: the hoisted assignment is new and the old one moved by two
    tagged = HOISTED.replace("size(x,1)  ! spel collapse-bounds", "size(x,1)  ! spel collapse-bounds  ! 7")
    tagged = tagged.replace("x(i, j) = 0._r8", "x(i, j) = 0._r8  ! 11")
    res = check_pipeline([CollapseBounds, TagAssignments], BEFORE, tagged, tmp_path)
    assert [s.transform.name for s in res.steps] == ["collapse-bounds", "test-tag-assignments"]
    # the snippet on disk is untouched; each step's context holds the text it planned on
    assert (tmp_path / "snippet_mod.F90").read_text() == BEFORE
    assert res.steps[1].ctx.lines(res.steps[1].ctx.fut.snippet)[7].startswith("  spel_ub2 =")


def test_pipeline_stops_when_a_step_breaks_the_routine(tmp_path):
    class Break(TagAssignments):
        name = "test-break"

        def build(self, ctx, opts):
            sub = next(iter(ctx.routines()))
            return Plan(self.name).add(Change(ctx.path(sub), [replace(sub.end_stmt_ln, "end do")]))

    with pytest.raises(PipelineError, match="after test-break"):
        check_pipeline([Break, TagAssignments], BEFORE, BEFORE, tmp_path)


def test_overlay_refuses_edits_of_a_module_head(tmp_path):
    path = tmp_path / "m.F90"
    path.write_text(BEFORE)
    mod = SimpleNamespace(name="m", filepath=path, end_of_head_ln=1, get_mod_lines=lambda: [])
    fut = SimpleNamespace(subroutine_dict={}, module_dict={"m": mod})
    new = BEFORE.replace("module m\n", "module m\n  use foo\n", 1)
    with pytest.raises(PipelineError, match="edits the head of module m"):
        AnalysisOverlay(fut, tmp_path).update({path: BEFORE}, {path: new}, "test-step")
    # directives and comments are fine (no routines here, so nothing else is refreshed)
    new = BEFORE.replace("module m\n", "module m\n  !$acc declare create(a)\n", 1)
    ov = AnalysisOverlay(fut, tmp_path)
    ov.update({path: BEFORE}, {path: new}, "test-step")
    assert ov._head["m"] == 2


def test_line_map():
    old = ["a", "b", "c", "d"]
    m = line_map(old, ["a", "x", "y", "b", "C", "d"])
    assert (m.strict(0), m.strict(1), m.strict(2), m.strict(3)) == (0, 3, 4, 5)
    assert 4 not in m.same and 3 in m.same
    m = line_map(old, ["a", "d"])
    assert m.strict(1) is None and m.near(1) == 1 and m.near(3) == 1


def test_preprocessor_reads_the_overridden_text(tmp_path):
    path = tmp_path / "p.F90"
    disk = "module p\n#ifdef NOPE\ninteger :: a\n#endif\nend module p\n"
    path.write_text(disk)
    edited = "module p\n#ifndef NOPE\ninteger :: b\n#endif\nend module p\n"
    assert [lt.line for lt in preprocessed_lines(path, 5)] == ["module p", "end module p"]
    with VirtualFS({path.resolve(): edited}, capture=False).active():
        lines = preprocessed_lines(path, 5)
    assert [(lt.ln, lt.line) for lt in lines] == [(0, "module p"), (2, "integer :: b"), (4, "end module p")]
    assert path.read_text() == disk


def test_spec_pipeline_writes_the_net_result_once(tmp_path, monkeypatch, capsys):
    import json

    from spel.scripts import config
    from spel.transforms.core import driver, pipeline, registry
    from spel.transforms.core.testing import SnippetOverlay

    ctx = snippet_context(BEFORE, tmp_path / "src")
    monkeypatch.setattr(config, "E3SM_SRCROOT", str(ctx.srcroot))
    monkeypatch.setattr(driver, "context", lambda fut=None, srcroot=None, reanalyze=False, config=None: ctx)
    monkeypatch.setattr(pipeline, "AnalysisOverlay", SnippetOverlay)
    monkeypatch.setitem(registry.REGISTRY, "test-tag-assignments", TagAssignments)
    spec = tmp_path / "s.toml"
    spec.write_text('[[step]]\ntransform = "collapse-bounds"\n[[step]]\ntransform = "test-tag-assignments"\n')
    out = tmp_path / "out"
    assert registry.main(["--spec", str(spec), "-o", str(out), "--verify"]) == 0
    assert "  ! 11" in (out / "snippet_mod.F90").read_text()
    assert (tmp_path / "src" / "snippet_mod.F90").read_text() == BEFORE
    runs = json.loads((out / driver.MANIFEST).read_text())["runs"]
    assert [r["transform"] for r in runs] == ["collapse-bounds", "test-tag-assignments"]
    printed = capsys.readouterr().out
    assert "collapse-bounds: 1 changes" in printed and printed.count("wrote ") == 1


@pytest.mark.parametrize("argv, new", [
    (["compact", "rewrite", "-o", "x"], "spel transform compact-rewrite"),
    (["nutrient", "merge", "--procedure", "init"], "spel transform nutrient-merge"),
    (["acc2omp", "--case", "c"], "spel transform acc2omp"),
])
def test_removed_commands_name_their_transform(monkeypatch, capsys, argv, new):
    from spel import cli

    monkeypatch.setattr("sys.argv", ["spel", *argv])
    with pytest.raises(SystemExit) as err:
        cli.main()
    assert err.value.code == 2 and new in capsys.readouterr().err
