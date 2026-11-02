"""BOZ literal constants (z'7ff4...', o'17', b'101') lex as one INT token and print back as written."""
from spel.scripts.fortran_parser import spel_ast as A
from spel.scripts.fortran_parser.tests.test_scope_walk import gvar, parse_statements, walk

SRC = """\
subroutine s(x, n)
  implicit none
  real(r8), intent(out) :: x
  integer, intent(out) :: n
  real(r8), parameter :: nan = transfer(int(Z'7FF4000000000000', i8), 1._r8)
  x = nan
  n = o'17' + b'101' + z"ff"
end subroutine s
"""


def _literals(node, out):
    if isinstance(node, A.IntegerLiteral):
        out.append(node)
    for v in vars(node).values() if hasattr(node, "__dict__") else ():
        for c in (v if isinstance(v, list) else [v]):
            if isinstance(c, A.Node) and c is not node:
                _literals(c, out)
    return out


def test_boz_literals_parse_and_walk():
    from spel.scripts.fortran_parser.scope_walk import AssignEvent

    rec = walk(SRC, globals=[gvar("r8"), gvar("i8")])
    assert {e.lhs.path for e in rec.events if isinstance(e, AssignEvent)} == {"x", "n"}
    tree = parse_statements(SRC).statements[0]
    lits = [l for l in _literals(tree, []) if l.boz is not None]
    assert sorted((l.boz.lower(), l.value) for l in lits) == sorted([
        ("z'7ff4000000000000'", 0x7FF4000000000000), ("o'17'", 0o17), ("b'101'", 0b101), ('z"ff"', 0xFF)])
    # printed as written (a decimal would change the meaning of int(..., i8) at the edge of the range)
    assert all(str(l) == l.boz for l in lits)
