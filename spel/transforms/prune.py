"""
Delete derived-type members nothing accesses.

A member (data component) of any type in the analysis' type_dict is deleted
when every statement anywhere under the scan root that names it (`%name`)
only declares, allocates, deallocates, nullifies or resets it:

  * its declaration in the type definition,
  * `allocate(x%name(...))`, `deallocate(x%name)`, `nullify(x%name)`,
    `x%name => null()`, `if (associated(x%name)) deallocate(x%name)`,
  * an assignment to it whose right-hand side reads no component and calls no
    function but intrinsics (`= spval`, `= nan`, `= 0._r8`, `= value_patch`).

Any other reference keeps it: a read, a call argument, a pointer association,
an associate, a history or restart registration, an OpenACC/OpenMP directive,
a statement that doesn't parse. Members are matched by name, not by type (the
scan covers code SPEL hasn't analyzed: init, restart, history, FATES, EMI), so
a name used through any type keeps every member of that name: conservative.

Statements and names come from SPEL's parser and lexer; the scan only uses text
to find statements and their line spans. `spel transform prune`.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from spel.scripts.fortran_parser.lexer import Lexer
from spel.scripts.fortran_parser.spel_ast import (
    AllocateStatement,
    AssignmentStatement,
    ExpressionStatement,
    FuncExpression,
    IfConstruct,
    InfixExpression,
    NullifyStatement,
    PointerAssignment,
    VariableDecl,
)
from spel.scripts.fortran_parser.spel_parser import Parser
from spel.scripts.fortran_parser.symbols import FORTRAN_INTRINSICS
from spel.scripts.fortran_parser.tokens import TokenTypes as Tok
from spel.scripts.types import LineTuple, LogicalLineIterator, directive_sentinel
from spel.transforms.core.edits import diff_edits
from spel.transforms.core.registry import transform
from spel.transforms.core.transform import Change, Plan, Transform
from spel.transforms.nutrient.tree import designator_path, split_code, statement_at

FORTRAN_SUFFIXES = (".F90", ".f90", ".F", ".f")
# searched for references but never edited: standalone EMI builds with their own same-named types
NO_EDIT = ("external_models/emi/src/elm_stub", "external_models/emi/src/demo")


def component_refs(text: str) -> list[str]:
    """Component names in a statement (`x%y(i)%z` -> y, z), from SPEL's lexer."""
    lex = Lexer(LogicalLineIterator([LineTuple(line=text, ln=0)], log_name="prune"))
    out, prev = [], None
    while True:
        try:
            tok = lex.next_token()
        except Exception:
            return out + ["?"]  # unlexable: treated as using everything it names
        if tok.token == Tok.EOF:
            return out
        if tok.token == Tok.IDENT:
            # the lexer keeps `this%core%leaf` as one identifier; after `x(i)` the `%` is its own token
            parts = tok.literal.lower().split("%")
            out += [q for q in (parts if prev == Tok.PERCENT else parts[1:]) if q]
        prev = tok.token


def parse_statement(text: str):
    """The statement's node; None when it doesn't parse alone (e.g. split by #ifdef): a use."""
    logging.disable(logging.ERROR)
    try:
        stmts = Parser(lines=[LineTuple(line=text, ln=0)]).parse_program().statements
    except (Exception, SystemExit):
        return None
    finally:
        logging.disable(logging.NOTSET)
    return stmts[0] if len(stmts) == 1 else None


def is_constant(expr) -> bool:
    """No component read, no function call but intrinsics (array refs look like calls: not constant)."""
    if component_refs(str(expr)):
        return False
    stack = [expr]
    while stack:
        e = stack.pop()
        if isinstance(e, FuncExpression):
            if str(e.function).lower() not in FORTRAN_INTRINSICS:
                return False
            stack.extend(e.args)
        elif isinstance(e, InfixExpression):
            stack += [e.left_expr, e.right_expr]
        elif hasattr(e, "right_expr"):
            stack.append(e.right_expr)
    return True


def last_is(expr, name: str) -> bool:
    path = designator_path(expr)
    return bool(path) and len(path) > 1 and path[-1] == name


@dataclass
class Ref:
    """A statement naming member(s): where it is and what it does with each name."""

    path: Path
    first: int
    last: int
    part: int  # index among the statement's `;`-separated parts
    nparts: int
    text: str
    removable: dict[str, str] = field(default_factory=dict)  # name -> 'delete' | 'object'
    uses: set[str] = field(default_factory=set)


def classify(text: str, names: set[str]) -> tuple[dict[str, str], set[str]]:
    """({name: how it can go}, names it uses) for the candidate names a statement mentions."""
    refs = [n for n in component_refs(text) if n in names or n == "?"]
    if not refs:
        return {}, set()
    if "?" in refs:
        return {}, set(refs) - {"?"} | (names & set(component_refs(text)))
    node = parse_statement(text)
    counts = defaultdict(int)
    for n in refs:
        counts[n] += 1
    removable: dict[str, str] = {}

    def assignment(lhs, rhs) -> Optional[str]:
        for n in counts:
            if last_is(lhs, n) and counts[n] == 1 and is_constant(rhs) and not component_refs(
                    ",".join(str(a) for a in getattr(lhs, "args", []) or [])):
                return n
        return None

    match node:
        case AllocateStatement():
            hits = defaultdict(int)
            for obj in node.objects:
                for n in counts:
                    if last_is(obj, n):
                        hits[n] += 1
            for n, k in hits.items():
                if k == counts[n] and not any(n in component_refs(str(v)) for v in node.options.values()):
                    removable[n] = "object" if len(node.objects) > 1 else "delete"
        case NullifyStatement():
            objs = getattr(node, "objects", None) or getattr(node, "pointers", None) or []
            for n in counts:
                if sum(last_is(o, n) for o in objs) == counts[n]:
                    removable[n] = "delete" if len(objs) == 1 else "keep"
        case PointerAssignment():
            if str(node.target).lower().replace(" ", "") == "null()":
                for n in counts:
                    if last_is(node.pointer, n) and counts[n] == 1:
                        removable[n] = "delete"
        case AssignmentStatement():
            n = assignment(node.left, node.right)
            if n:
                removable[n] = "delete"
        case ExpressionStatement() if isinstance(node.expression, InfixExpression) and node.expression.operator in ("=", "=>"):
            e = node.expression
            if e.operator == "=":
                n = assignment(e.left_expr, e.right_expr)
                if n:
                    removable[n] = "delete"
            elif str(e.right_expr).lower().replace(" ", "") == "null()":
                for n in counts:
                    if last_is(e.left_expr, n) and counts[n] == 1:
                        removable[n] = "delete"
        case IfConstruct() if not node.else_ifs and node.else_ is None and len(node.consequence.statements) == 1:
            # if (associated(x%n)) deallocate(x%n) / if (allocated(...)) ...
            cond = node.condition
            body = node.consequence.statements[0]
            if isinstance(cond, FuncExpression) and str(cond.function).lower() in ("associated", "allocated") \
                    and len(cond.args) == 1:
                inner, _ = classify(str(body), names)
                for n, how in inner.items():
                    if how == "delete" and last_is(cond.args[0], n) and counts[n] == 2:
                        removable[n] = "delete"
    removable = {n: h for n, h in removable.items() if h != "keep"}
    return removable, set(counts) - set(removable)


class Pruner:
    """
    srcroot:   E3SM checkout
    type_dirs: directories whose modules' types are pruned (default: components/elm/src/data_types)
    scan_root: where references are searched (default: components/elm)
    """

    def __init__(self, srcroot: Path, type_dict: dict, scan_root=None, read=None):
        self.srcroot = Path(srcroot)
        self.read = read  # path -> lines (the transform's ctx.lines); default: the file
        self.type_dict = type_dict
        self.scan_root = Path(scan_root) if scan_root else self.srcroot / "components/elm"
        self.lines: dict[Path, list[str]] = {}
        # name -> (type, file, declaration statement, its line)
        self.members: dict[str, list[tuple[str, Path, VariableDecl, int]]] = defaultdict(list)
        self.refs: dict[str, list[Ref]] = defaultdict(list)
        self.kept: dict[str, str] = {}
        self.dead: list[str] = []
        self.no_edit = list(NO_EDIT) + self.submodules()

    def submodules(self) -> list[str]:
        """Paths of git submodules (FATES, ...): searched, never edited (another repository)."""
        import subprocess

        r = subprocess.run(["git", "-C", str(self.srcroot), "submodule", "status"], capture_output=True, text=True)
        return [line.split()[1] + "/" for line in r.stdout.splitlines() if len(line.split()) > 1]

    def file_lines(self, path: Path) -> list[str]:
        if path not in self.lines:
            # universal newlines, as the transform driver reads (it writes CRLF files back as CRLF)
            self.lines[path] = (list(self.read(path)) if self.read
                                else path.read_text(errors="replace").split("\n"))
        return self.lines[path]

    def editable_path(self, path: Path) -> bool:
        rel = path.relative_to(self.srcroot).as_posix() if path.is_relative_to(self.srcroot) else path.as_posix()
        return not any(rel.startswith(d) or d in rel for d in self.no_edit)

    def collect_members(self) -> None:
        """
        The data members of every type in the analysis' type_dict, at the declaration the
        analysis records (checked against the parse); inherited and flattened nested
        copies (`core%leaf`) are their declaring type's.
        """
        seen = set()
        for tname, dtype in sorted(self.type_dict.items()):
            path = Path(dtype.filepath)
            path = path if path.is_absolute() else self.srcroot / path
            if not path.exists():
                continue
            lines = self.file_lines(path)
            for cname, var in dtype.components.items():
                if "%" in cname or var.ln is None or not (0 <= var.ln < len(lines)):
                    continue
                key = (path, var.ln, cname)
                if key in seen:
                    continue
                last, parts = statement_at(lines, var.ln)
                decl = parse_statement(parts[0]) if len(parts) == 1 else None
                names = [e.token.literal.lower() for e in decl.entities] if isinstance(decl, VariableDecl) else []
                if cname.lower() not in names:
                    continue  # a copy (inherited from a type in another file) or not at that line
                seen.add(key)
                self.members[cname.lower()].append((tname, path, decl, var.ln))

    def scan(self) -> None:
        names = set(self.members)
        for path in sorted(self.scan_root.rglob("*")):
            if path.suffix not in FORTRAN_SUFFIXES or not path.is_file():
                continue
            lines = self.file_lines(path)
            i = 0
            while i < len(lines):
                code = split_code(lines[i])[0]
                if directive_sentinel(lines[i]):
                    last, parts = statement_at(lines, i)
                    for n in set(component_refs(parts[0].split(None, 1)[-1])) & names:
                        self.refs[n].append(Ref(path, i, last, 0, 1, parts[0], uses={n}))
                    i = last + 1
                    continue
                if not code.strip():
                    i += 1
                    continue
                last, parts = statement_at(lines, i)
                for k, part in enumerate(parts):
                    if "%" not in part:
                        continue
                    removable, uses = classify(part, names)
                    for n in set(removable) | uses:
                        self.refs[n].append(Ref(path, i, last, k, len(parts), part,
                                                {n: removable[n]} if n in removable else {}, {n} & uses))
                i = last + 1

    def decide(self) -> None:
        for n in sorted(self.members):
            refs = self.refs.get(n, [])
            used = [r for r in refs if n in r.uses]
            if used:
                r = used[0]
                self.kept[n] = f"used: {r.path.relative_to(self.srcroot)}:{r.first + 1}: {r.text[:70]}"
                continue
            if any(not self.editable_path(path) for _, path, _, _ in self.members[n]):
                self.kept[n] = "declared in a submodule or the EMI stub (not edited)"
                continue
            bad = [r for r in refs if not self.editable(r, n)]
            if bad:
                r = bad[0]
                self.kept[n] = f"can't edit: {r.path.relative_to(self.srcroot)}:{r.first + 1}: {r.text[:70]}"
                continue
            self.dead.append(n)

    def editable(self, r: Ref, n: str) -> bool:
        if r.removable.get(n) == "object":
            return r.first == r.last and r.nparts == 1
        return r.nparts == 1 or r.first == r.last

    def build(self) -> dict[Path, list[str]]:
        self.collect_members()
        self.scan()
        self.decide()
        dead = set(self.dead)
        # deletions: (path) -> {line: replacement lines or None}
        edits: dict[Path, dict[int, Optional[list[str]]]] = defaultdict(dict)
        parts_drop: dict[tuple[Path, int], set[int]] = defaultdict(set)
        objects_drop: dict[tuple[Path, int], set[str]] = defaultdict(set)
        for n in dead:
            for r in self.refs.get(n, []):
                if not self.editable_path(r.path):
                    continue
                if r.removable.get(n) == "object":
                    objects_drop[(r.path, r.first)].add(n)
                elif r.nparts == 1:
                    for ln in range(r.first, r.last + 1):
                        edits[r.path][ln] = None
                else:
                    parts_drop[(r.path, r.first)].add(r.part)
        for (path, ln), drop in parts_drop.items():
            if edits[path].get(ln, "") is None:
                continue
            lines = self.file_lines(path)
            _, parts = statement_at(lines, ln)
            keep = [p for k, p in enumerate(parts) if k not in drop]
            indent = lines[ln][: len(lines[ln]) - len(lines[ln].lstrip())]
            comment = split_code(lines[ln])[1]
            edits[path][ln] = [indent + " ; ".join(keep) + (" " + comment if comment else "")] if keep else None
        for (path, ln), names_ in objects_drop.items():
            lines = self.file_lines(path)
            node = parse_statement(statement_at(lines, ln)[1][0])
            objs = [str(o) for o in node.objects if not any(last_is(o, n) for n in names_)]
            opts = "".join(f", {k}={v}" for k, v in node.options.items())
            indent = lines[ln][: len(lines[ln]) - len(lines[ln].lstrip())]
            comment = split_code(lines[ln])[1]
            new = f"{indent}{node.token.literal}({', '.join(objs)}{opts})" if objs else None
            edits[path][ln] = [new + (" " + comment if comment else "")] if new else None
        # declarations
        for n in dead:
            for _, path, decl, at in self.members[n]:
                lines = self.file_lines(path)
                last, _ = statement_at(lines, at)
                ents = [e for e in decl.entities if e.token.literal.lower() not in dead]
                if not ents:
                    for ln in range(at, last + 1):
                        edits[path][ln] = None
                else:
                    spec = split_code(lines[at])[0].split("::", 1)[0].rstrip()
                    edits[path][at] = [f"{spec} :: {', '.join(str(e) for e in ents)}"]
                    for ln in range(at + 1, last + 1):
                        edits[path][ln] = None
        self.files = {}
        for path, by_line in edits.items():
            lines = self.file_lines(path)
            out = []
            for i, l in enumerate(lines):
                if i in by_line:
                    if by_line[i] is not None:
                        out += by_line[i]
                else:
                    out.append(l)
            self.files[path] = out
        return self.files

    def report(self, verbose: bool = False) -> str:
        by_type = defaultdict(list)
        for n in self.dead:
            for t, _, _, _ in self.members[n]:
                by_type[t].append(n)
        out = [f"{len(self.dead)} unused member names ({sum(len(v) for v in by_type.values())} members) "
               f"of {sum(len(v) for v in self.members.values())}; {len(self.files)} files edited"]
        for t in sorted(by_type):
            out.append(f"  {t}: {', '.join(sorted(by_type[t]))}")
        if verbose:
            out.append("kept:")
            out += [f"  {n}: {why}" for n, why in sorted(self.kept.items())]
        return "\n".join(out)


@dataclass
class Options:
    verbose: bool = field(default=False, metadata={"help": "also report why each kept member is kept"})


@transform
class Prune(Transform):
    """
    Matches: data members of the analyzed types (type_dict) that every statement under
             components/elm naming them (`%name`) only declares, allocates, deallocates,
             nullifies or resets to a constant.
    Changes: deletes those statements (or the member's part of a `;` line or an allocate list)
             and the member's declaration.
    Refuses: nothing; members it can't edit safely are kept and listed with --verbose
             (any read, call argument, pointer association, associate, registration,
             directive, unparsable statement; declarations in submodules or the EMI stub).
    Members are matched by name across types (conservative).
    """

    name = "prune"
    summary = "delete derived-type members nothing accesses (only declared, allocated, reset)"
    Options = Options

    def build(self, ctx, opts) -> Plan:
        p = Pruner(ctx.srcroot, ctx.fut.type_dict, read=ctx.lines)
        files = p.build()
        plan = Plan(self.name)
        plan.notes += p.report(verbose=opts.verbose).split("\n")
        for path, lines in sorted(files.items()):
            plan.add(Change(path, diff_edits(p.file_lines(path), lines, why=self.name), where=path.name))
        return plan
