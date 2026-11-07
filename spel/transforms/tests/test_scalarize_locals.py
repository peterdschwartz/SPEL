"""Per-iteration local arrays become scalars (scalarize_locals)."""
from spel.scripts.fortran_parser.spel_parser import Parser
from spel.scripts.tests.test_record_access import StubSub
from spel.scripts.types import LineTuple
from spel.transforms.core.edits import EditSet
from spel.transforms.local_forms import check_routine
from spel.transforms.scalarize_locals import plan_routine
from spel.transforms.tests.test_local_forms import SRC


def _apply(src):
    lines = src.split("\n")
    sub = StubSub(src)
    sub.startline, sub.endline = 0, len(lines) - 1
    tree = Parser(lines=[LineTuple(line=l, ln=i) for i, l in enumerate(lines)],
                  keep_spans=True).parse_program().statements[0]
    findings = [f for f in check_routine(sub) if f.kind == "scalar"]
    edits, done, why = plan_routine(sub, lines, tree, findings)
    es = EditSet("x")
    for e in edits:
        es.add(e)
    return es.apply(lines), done, why


def test_scalarize():
    out, done, why = _apply(SRC)
    text = "\n".join(out)
    assert done == ["tmp"], why
    assert "  real(r8) :: tmp\n" in text + "\n"
    assert "tmp = x(c,j) * rho" in text and "y(c,j) = tmp + ph(fc) * half + lg" in text
    # the others stay arrays
    assert "cond(fc,j)" in text and "carried(fc,j)" in text and "kept(fc)" in text


def test_second_run_changes_nothing():
    out, _, _ = _apply(SRC)
    again, done, _ = _apply("\n".join(out))
    assert done == [] and again == out


def test_directive_blocks():
    src = SRC.replace("  do j = 1, 10\n", "  !$acc enter data create(tmp)\n  do j = 1, 10\n")
    _, done, why = _apply(src)
    assert done == [] and any("directive" in w for w in why)
