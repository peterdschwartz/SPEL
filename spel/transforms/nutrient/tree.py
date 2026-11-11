"""
A routine as SPEL's parser sees it, tied back to its source lines.

procedures.py compares statements as source text (string case matters) and
merge.py rewrites routines line by line. Both take the routine's structure
from the parse tree (registrations.parse_routine): which statements
are declarations, plain statements, loops, ifs and their branches, associate
blocks, with their first and last lines (`lineno`, `end_ln`), and the
conditions guarding each statement. The source is read only for a
statement's own text: its physical lines, continuations joined, statements
sharing a line (`a = 1 ; b = 2`) taken in order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

from spel.scripts.fortran_parser.scope_walk import flatten_designator
from spel.scripts.fortran_parser.spel_ast import (
    AllocateStatement,
    ArrayInit,
    AssociateConstruct,
    BoundsExpression,
    DirectiveStatement,
    FieldAccessExpression,
    FuncExpression,
    KeywordArgument,
    PrefixExpression,
    DoLoop,
    DoWhile,
    ExpressionStatement,
    GotoStatement,
    IfConstruct,
    ImplicitNoneStatement,
    InfixExpression,
    MacroIf,
    ReturnStatement,
    SelectCaseConstruct,
    Statement,
    SubCallStatement,
    UseStatement,
    VariableDecl,
    WhereConstruct,
)
from spel.scripts.types import directive_sentinel
from spel.scripts.fortran_parser.spel_ast import Identifier
from spel.scripts.registrations import _norm, parse_routine

SPECS = (VariableDecl, UseStatement, ImplicitNoneStatement)
LOOPS = (DoLoop, DoWhile, AssociateConstruct)


def split_code(line: str) -> tuple[str, str]:
    """(code, comment) of a source line, `!` inside strings kept."""
    out, quote = [], None
    for i, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch == "!":
            return "".join(out), line[i:]
        out.append(ch)
    return "".join(out), ""


def split_semicolons(text: str) -> list[str]:
    parts, cur, quote = [], [], None
    for ch in text:
        if quote:
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch == ";":
            parts.append("".join(cur))
            cur = []
            continue
        cur.append(ch)
    parts.append("".join(cur))
    return [p.strip() for p in parts if p.strip()]


def statement_at(lines: list[str], ln: int) -> tuple[int, list[str]]:
    """(last physical line, statements) of the source statement (or directive) starting on line `ln`."""
    if sentinel := directive_sentinel(lines[ln]):
        parts, i = [], ln
        while True:
            text = split_code(lines[i].lstrip()[len(sentinel):])[0].strip()
            if text.startswith("&"):
                text = text[1:].strip()
            if not text.endswith("&") or i + 1 >= len(lines):
                parts.append(text)
                break
            parts.append(text[:-1].strip())
            i += 1
            while i + 1 < len(lines) and not directive_sentinel(lines[i]):
                i += 1
        return i, [f"{lines[ln].lstrip()[: len(sentinel)]} {' '.join(parts)}"]
    buf, i = "", ln
    while True:
        code = split_code(lines[i])[0].strip()
        if buf and code.startswith("&"):
            code = code[1:].lstrip()
        if not code.endswith("&"):
            return i, split_semicolons(buf + code)
        buf += code[:-1].rstrip() + " "
        i += 1
        while i < len(lines) and not split_code(lines[i])[0].strip():
            i += 1  # comment lines between continuation lines


@dataclass
class Span:
    first: int  # physical lines first..last (0-based)
    last: int
    text: str  # the statement's source, continuations joined


@dataclass
class Leaf:
    """A statement that is not a construct."""

    span: Span
    node: Statement
    guards: tuple[str, ...] = ()  # conditions of the enclosing ifs/cases, normalized
    guard_exprs: tuple = ()  # the same conditions as parsed (not negated), one per guard
    # set by procedures.compare_procedure
    key: str = ""  # normalized guards + text
    body: str = ""  # normalized text
    plain: str = ""  # normalized guards + text without long_name=
    fields: frozenset[str] = frozenset()  # base names of the components it touches
    iso: bool = False  # under a carbon_type guard (dropped from the key)
    kind: str = ""  # generic | guards | longname | differs | own | other

    @property
    def ln(self) -> int:
        return self.span.first

    @property
    def text(self) -> str:
        return self.span.text

    @property
    def is_call(self) -> bool:
        return isinstance(self.node, SubCallStatement)

    @property
    def is_directive(self) -> bool:
        return isinstance(self.node, DirectiveStatement)

    @property
    def closes(self) -> bool:
        """A directive that ends the construct before it (`!$acc end parallel`)."""
        return self.is_directive and self.node.construct == "end"


@dataclass
class Block:
    """A loop, if (with its else-if/else branches) or associate construct."""

    node: Statement
    head: Optional[Span]  # None: a construct merge.py doesn't rewrite (select, where, #if)
    segments: list[tuple[Optional[Span], list]]  # (else-if/else header, children)
    tail: Optional[Span] = None

    def leaves(self) -> Iterator[Leaf]:
        for _, children in self.segments:
            yield from iter_leaves(children)

    def blocks(self) -> Iterator["Block"]:
        yield self
        for _, children in self.segments:
            for c in children:
                if isinstance(c, Block):
                    yield from c.blocks()


def iter_leaves(nodes: list) -> Iterator[Leaf]:
    for n in nodes:
        if isinstance(n, Leaf):
            yield n
        else:
            yield from n.leaves()


@dataclass
class Routine:
    path: Path
    name: str
    start: int  # the subroutine statement's line
    end: int  # the end subroutine line
    header: list[tuple[Span, Statement]]  # the subroutine statement, then its declarations
    body: list  # Leaf | Block
    tail: Span
    dummies: list[str]
    unsupported: list[str] = field(default_factory=list)  # what merge.py can't rewrite

    def leaves(self) -> Iterator[Leaf]:
        return iter_leaves(self.body)

    def blocks(self) -> Iterator[Block]:
        for n in self.body:
            if isinstance(n, Block):
                yield from n.blocks()

    @property
    def last_header_line(self) -> int:
        return self.header[-1][0].last

    @property
    def local_names(self) -> set[str]:
        """Variables the routine declares that aren't dummies."""
        dummies = {d.lower() for d in self.dummies}
        return {n for _, node in self.header[1:] for n in declared_names(node)} - dummies

    def requirements(self, names: set[str]) -> tuple[list[int], list[tuple[Span, VariableDecl, list[str]]], list[tuple[str, list[str]]]]:
        """
        What a routine made from this one's statements needs for `names`: the positions
        of the dummies among them (`this` excluded), the declarations of the dummies and
        locals among them as (span, declaration, the entities needed from it), and
        (module, names) for the routine's own `use` imports among them. Other names are
        visible from the module.
        """
        names = {n.lower() for n in names} - {self.dummies[0].lower()} if self.dummies else set(names)
        dummies = [i for i, d in enumerate(self.dummies) if i > 0 and d.lower() in names]
        decls, uses = [], []
        for sp, node in self.header[1:]:
            if isinstance(node, UseStatement):
                got = [str(o).lower() for o in node.objs if str(o).lower() in names]
                if got:
                    uses.append((str(node.module).lower(), got))
                continue
            declared = declared_names(node)
            needed = [n for n in declared if n in names]
            if needed:
                decls.append((sp, node, needed))
        return dummies, decls, uses


def read_routine(sub) -> Routine:
    """The routine `sub` (an analyzed Subroutine: filepath, startline, endline) from its parse tree."""
    lines = Path(sub.filepath).read_text(errors="replace").split("\n")
    tree = parse_routine(sub.filepath, sub.startline, sub.endline, keep_directives=True, keep_string_case=True)
    taken: dict[int, int] = {}
    unsupported: list[str] = []

    def span(ln: int) -> Span:
        last, parts = statement_at(lines, ln)
        k = taken.get(ln, 0)
        taken[ln] = k + 1
        text = parts[k] if k < len(parts) else ""
        if not text:
            unsupported.append(f"line {ln + 1}: no source statement for the parsed one")
        return Span(ln, last, text)

    def build(stmts: list[Statement], guards: tuple[str, ...], gexprs: tuple = ()) -> list:
        out: list = []
        for s in stmts:
            if s.label is not None:
                unsupported.append(f"line {s.lineno + 1}: statement label {s.label}")
            if isinstance(s, IfConstruct):
                close = s.else_.end_ln if s.else_ else s.else_ifs[-1].end_ln if s.else_ifs else s.end_ln
                if close < 0 or close == s.lineno:  # `if (c) stmt`: one statement (no end_ln)
                    out.append(Leaf(span(s.lineno), s, guards, gexprs))
                    continue
                cond = _norm(s.condition)
                head = span(s.lineno)
                segs = [(None, build(s.consequence.statements, guards + (cond,), gexprs + (s.condition,)))]
                prev, prev_e = [cond], [s.condition]
                for e in s.else_ifs:
                    c = _norm(e.condition)
                    hdr = span(e.lineno)
                    g = guards + tuple(f"not({p})" for p in prev) + (c,)
                    segs.append((hdr, build(e.consequence.statements, g, gexprs + tuple(prev_e) + (e.condition,))))
                    prev.append(c)
                    prev_e.append(e.condition)
                if s.else_ is not None:
                    hdr = span(s.else_.lineno)
                    g = guards + tuple(f"not({p})" for p in prev)
                    segs.append((hdr, build(s.else_.alternative.statements, g, gexprs + tuple(prev_e))))
                out.append(Block(s, head, segs, span(close)))
            elif isinstance(s, LOOPS):
                head = span(s.lineno)
                out.append(Block(s, head, [(None, build(s.body.statements, guards, gexprs))], span(s.end_ln)))
            elif isinstance(s, SelectCaseConstruct):
                unsupported.append(f"line {s.lineno + 1}: select case")
                sel = _norm(s.selector)
                segs = []
                for case in s.cases:
                    vals = ",".join(_norm(v) for v in case.values or []) or "default"
                    segs.append((None, build(case.body.statements, guards + (f"{sel}=={vals}",), gexprs + (s.selector,))))
                out.append(Block(s, None, segs))
            elif isinstance(s, WhereConstruct):
                unsupported.append(f"line {s.lineno + 1}: where")
                segs = [(None, build(s.body.statements, guards, gexprs))]
                segs += [(None, build(ew.body.statements, guards, gexprs)) for ew in s.elsewheres]
                out.append(Block(s, None, segs))
            elif isinstance(s, MacroIf):
                unsupported.append(f"line {s.lineno + 1}: preprocessor #if")
                segs = [(None, build(s.body.statements, guards, gexprs))]
                segs += [(None, build(b.body.statements, guards, gexprs)) for b in s.branches]
                out.append(Block(s, None, segs))
            else:
                if isinstance(s, (ReturnStatement, GotoStatement)):
                    unsupported.append(f"line {s.lineno + 1}: {type(s).__name__}")
                out.append(Leaf(span(s.lineno), s, guards, gexprs))
        return out

    header = [(span(tree.lineno), tree)]
    stmts = list(tree.body.statements)
    i = 0
    while i < len(stmts) and (isinstance(stmts[i], SPECS) or is_spec_directive(stmts[i])):
        header.append((span(stmts[i].lineno), stmts[i]))
        i += 1
    body = build(stmts[i:], ())
    return Routine(
        path=Path(sub.filepath),
        name=str(tree.name).lower(),
        start=tree.lineno,
        end=tree.end_ln,
        header=header,
        body=body,
        tail=Span(tree.end_ln, tree.end_ln, split_code(lines[tree.end_ln])[0].strip()),
        dummies=[str(a) for a in tree.args],
        unsupported=unsupported,
    )


def is_spec_directive(node: Statement) -> bool:
    """`!$acc routine`, `!$acc declare`: directives of the specification part."""
    return isinstance(node, DirectiveStatement) and node.construct in ("routine", "declare")


def designator_path(expr) -> Optional[list[str]]:
    """`this%core%leaf(i)` -> ['this', 'core', 'leaf']; None for anything but a designator."""
    try:
        return [n.lower() for n, _ in flatten_designator(expr)]
    except Exception:
        return None


def assigned_path(node: Statement, pointer: bool = False) -> Optional[list[str]]:
    """
    The designator an assignment statement writes (['this', 'core', 'species']), else
    None; with `pointer`, also what a pointer assignment (`p => x`) associates.
    """
    if isinstance(node, ExpressionStatement):
        expr = node.expression
        if isinstance(expr, InfixExpression) and (expr.operator == "=" or pointer and expr.operator == "=>"):
            return designator_path(expr.left_expr)
    return None


def written_paths(node: Statement) -> Optional[list[list[str]]]:
    """
    What a statement writes, when the parse tree says it plainly: an assignment's or
    pointer assignment's target, an allocate's objects; None for anything else
    (calls, I/O, constructs).
    """
    from spel.scripts.fortran_parser.spel_ast import AllocateStatement

    if isinstance(node, AllocateStatement):
        paths = [designator_path(o) for o in node.objects]
        return paths if all(paths) else None
    path = assigned_path(node, pointer=True)
    return [path] if path else None


def associations(r: Routine) -> dict[str, list[str]]:
    """Associate name -> the designator it names (['this', 'core', 'leaf']), for the whole routine."""
    out = {}
    for blk in r.blocks():
        if isinstance(blk.node, AssociateConstruct):
            for name, sel in blk.node.associations.items():
                path = designator_path(sel)
                if path:
                    out[str(name).lower()] = path
    return out


def declared_names(node: Statement) -> list[str]:
    """The entities a declaration statement declares."""
    if isinstance(node, VariableDecl):
        return [e.token.literal.lower() for e in node.entities]
    return []


def call_keywords(leaf: Leaf) -> Optional[tuple[str, dict[str, object]]]:
    """(called name, {keyword: value expression}) of a call whose arguments are all keywords, else None."""
    if not isinstance(leaf.node, SubCallStatement):
        return None
    func = leaf.node.function
    if not isinstance(func, FuncExpression):  # `call x` without arguments
        return None
    args = func.args
    out = {}
    for a in args:
        if not (isinstance(a, InfixExpression) and a.operator == "="):
            return None
        out[str(a.left_expr).lower()] = a.right_expr
    return str(func.function).lower(), out


LITERAL_TYPES = ("StringLiteral", "IntegerLiteral", "FloatLiteral", "LogicalLiteral")


def names_in(expr) -> set[str]:
    """The base names an expression refers to (`this%core%x(i) + f(n)` -> this, i, f, n)."""
    if expr is None or type(expr).__name__ in LITERAL_TYPES:
        return set()
    match expr:
        case Identifier():
            return {expr.value.split("%")[0].lower()}
        case FuncExpression():
            out = names_in(expr.function)
            for a in expr.args:
                if isinstance(a, InfixExpression) and a.operator == "=":
                    a = a.right_expr  # keyword argument: the keyword isn't a name in scope
                out |= names_in(a)
            return out
        case FieldAccessExpression():
            return names_in(expr.left) | names_in_subscripts(expr.field)
        case InfixExpression():
            return names_in(expr.left_expr) | names_in(expr.right_expr)
        case PrefixExpression():
            return names_in(expr.right_expr)
        case BoundsExpression():
            return names_in(expr.start) | names_in(expr.end)
        case KeywordArgument():
            return names_in(expr.value)
        case ArrayInit():
            out = set()
            for e in expr.elements:
                out |= names_in(e)
            return out
    return set()


def names_in_subscripts(expr) -> set[str]:
    """Names in the subscripts of a component reference (`%y(i)` -> i), not the component."""
    if isinstance(expr, FuncExpression):
        out = set()
        for a in expr.args:
            out |= names_in(a)
        return out | names_in_subscripts(expr.function)
    if isinstance(expr, FieldAccessExpression):
        return names_in_subscripts(expr.left) | names_in_subscripts(expr.field)
    return set()


def statement_names(node) -> set[str]:
    """The base names a statement refers to."""
    match node:
        case ExpressionStatement():
            return names_in(node.expression)
        case SubCallStatement():
            return names_in(node.function)
        case AllocateStatement():
            out = set()
            for o in node.objects:
                out |= names_in(o)
            return out
        case IfConstruct():
            out = names_in(node.condition)
            for e in node.else_ifs:
                out |= names_in(e.condition)
            return out
        case DoLoop():
            out = {node.index.lower()} if node.index else set()
            for e in (node.start, node.end, node.step):
                out |= names_in(e)
            return out
        case DoWhile():
            return names_in(node.condition)
        case AssociateConstruct():
            out = set()
            for sel in node.associations.values():
                out |= names_in(sel)
            return out
    return set()


def declaration_lines(lines: list[str], decls) -> list[str]:
    """
    Source lines for (span, declaration, needed entities): the declaration as written when
    it declares only those, else its type and attributes with just the needed entities.
    """
    out = []
    for sp, node, needed in decls:
        if needed == declared_names(node):
            out.extend(lines[sp.first : sp.last + 1])
            continue
        spec = sp.text.split("::", 1)[0].rstrip()
        indent = lines[sp.first][: len(lines[sp.first]) - len(lines[sp.first].lstrip())]
        ents = [str(e) for e in node.entities if e.token.literal.lower() in needed]
        out.append(f"{indent}{spec} :: {', '.join(ents)}")
    return out


def use_lines(uses: list[tuple[str, list[str]]], indent: str = "    ") -> list[str]:
    return [f"{indent}use {module}, only : {', '.join(names)}" for module, names in uses]
