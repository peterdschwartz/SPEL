"""Source spans of tokens and expressions (`keep_spans`)."""

from spel.scripts.fortran_parser.spel_ast import FieldAccessExpression, FuncExpression, KeywordArgument, SubCallStatement
from spel.scripts.fortran_parser.spel_parser import Parser
from spel.scripts.registrations import iter_statements
from spel.scripts.types import LineTuple
from spel.transforms.compact.rewrite import iter_expressions
from spel.transforms.core.edits import span_text

SRC = """\
subroutine foo(this, bounds, &   ! dummies
     n)
  integer :: n
  this % x( c, j) = this%y(c) + &
     ! a comment line inside the statement
     & this%z( c) * (a + b) ; w(c) = 'a !&b' // s(1)
  call bar(n, data=this%q, &
       k = 2)
  !$acc parallel loop &
  !$acc& async(1)
  do c = 1, n
     q(c) = 1
  end do
end subroutine foo"""


def parse(src=SRC, **kw):
    lines = src.split("\n")
    p = Parser(lines=[LineTuple(line=l, ln=i) for i, l in enumerate(lines)], keep_spans=True, **kw)
    return lines, p.parse_program().statements[0]


def exprs(sub):
    out = []
    for stmt, _ in iter_statements(sub.body):
        out += [(stmt, e) for e in iter_expressions(stmt)]
    return out


def texts(lines, sub, cls=object):
    return [span_text(lines, e.span) for _, e in exprs(sub) if isinstance(e, cls) and e.span]


def test_spans_of_array_references_across_continuations_and_semicolons():
    lines, sub = parse()
    funcs = texts(lines, sub, FuncExpression)
    assert "this % x( c, j)" in texts(lines, sub, FieldAccessExpression)
    assert "this%y(c)" in funcs
    assert "this%z( c)" in funcs
    assert "w(c)" in funcs
    assert "s(1)" in funcs
    assert "q(c)" in funcs
    # an expression across a continuation (and a comment line) spans the lines
    sums = [t for t in texts(lines, sub) if t.startswith("this%y(c) + &")]
    assert sums and sums[0].endswith("this%z( c) * (a + b)")
    # the first subscript of this%x(c, j): columns of `c` on line 3
    (x,) = [e for _, e in exprs(sub) if isinstance(e, FuncExpression) and str(e.function) == "x"]
    assert x.function.span == (3, 9, 3, 10)
    assert x.args[0].span == (3, 12, 3, 13)


def test_keyword_arguments_strings_and_header():
    lines, sub = parse()
    call = next(s for s, _ in iter_statements(sub.body) if isinstance(s, SubCallStatement))
    assert span_text(lines, call.function.span) == "bar(n, data=this%q, &\n       k = 2)"
    assert [span_text(lines, a.span) for a in call.function.args] == ["n", "data=this%q", "k = 2"]
    assert lines[call.function.span[2]][call.function.span[3] - 1] == ")"
    assert span_text(lines, sub.args_close) == ")" and sub.args_close[0] == 1
    strings = [e for _, e in exprs(sub) if type(e).__name__ == "StringLiteral"]
    assert span_text(lines, strings[0].span) == "'a !&b'"


def test_classified_nodes_keep_spans():
    from spel.scripts.fortran_parser.spel_ast import InfixExpression, to_actual_arg

    lines, sub = parse()
    call = next(s for s, _ in iter_statements(sub.body) if isinstance(s, SubCallStatement))
    kw = to_actual_arg(call.function.args[1])
    assert isinstance(kw, KeywordArgument) and isinstance(call.function.args[1], InfixExpression)
    assert span_text(lines, kw.span) == "data=this%q"
    assert span_text(lines, kw.value.span) == "this%q"


def test_directive_tokens_and_no_spans_by_default():
    lines, sub = parse(keep_directives=True)
    directive = next(s for s, _ in iter_statements(sub.body) if type(s).__name__ == "DirectiveStatement")
    assert directive.token.span[:2] == (8, 2)
    plain = Parser(lines=[LineTuple(line=l, ln=i) for i, l in enumerate(SRC.split("\n"))])
    sub = plain.parse_program().statements[0]
    assert all(e.span is None for _, e in exprs(sub)) and sub.args_close is None
