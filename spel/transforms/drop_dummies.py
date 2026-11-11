"""
Remove dummy arguments a routine no longer uses (`spel transform drop-dummies`).

After filter-locals many routines keep a `bounds` dummy that nothing reads any more (their arrays
are sized by the filter count). This pass removes such a dummy from the routine's header and
declarations and the matching actual from every call:

    subroutine SolveTemperature(bounds, num_filter, filter, ...)   ->  (num_filter, filter, ...)
    call SolveTemperature(bounds, num_nolakec, ...)                 ->  (num_nolakec, ...)

Only when it is safe:
- the dummy is in `--names` (default: bounds) and nothing in the routine's text uses it: no
  statement, declaration (`x(bounds%begc:)`), I/O statement or directive; the routine is a
  subroutine without internal procedures;
- every call is known: the analyzed call sites are all the `call NAME` statements of the ELM
  sources, the name is defined once (no interface body, no same-named routine elsewhere) and is not
  bound as a procedure (generic interface, type-bound procedure);
- the declaration declares only the dummy, on one line; each argument to remove sits on one line
  with its separating comma.
Anything else is refused and reported. Values and results are unchanged.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from spel.transforms.core.edits import delete, replace_cols
from spel.transforms.core.registry import transform
from spel.transforms.core.source import statement_start
from spel.transforms.core.transform import Change, Plan, Refusal, Transform
from spel.transforms.filter_locals import TextUses, _word_in
from spel.transforms.filter_locals_ip import text_call_count

if TYPE_CHECKING:
    from spel.scripts.analyze_subroutines import Subroutine
    from spel.scripts.fortran_parser.spel_ast import SubroutineDefinitionConstruct


def remove_arg(lines: list[str], spans: list, i: int):
    """The edit that removes argument `i` of a list whose items have parser `spans`, with its
    comma; None when the layout is not one this handles (argument over several lines, comma on
    another line)."""
    s = spans[i]
    if s is None or s[0] != s[2]:
        return None
    ln, c0, _, c1 = s
    if i + 1 < len(spans):
        nxt = spans[i + 1]
        if nxt is None:
            return None
        if nxt[0] == ln:
            return replace_cols(ln, c0, nxt[1], "")
        rest = lines[ln][c1:]
        k = len(rest) - len(rest.lstrip())
        if not rest[k:].startswith(","):
            return None
        end = c1 + k + 1
        while end < len(lines[ln]) and lines[ln][end] == " " and not lines[ln][end:].lstrip().startswith("&"):
            end += 1
        return replace_cols(ln, c0, end, "")
    if i == 0:
        return replace_cols(ln, c0, c1, "")
    prev = spans[i - 1]
    if prev is None:
        return None
    if prev[2] == ln:
        return replace_cols(ln, prev[3], c1, "")
    # the last argument on a continuation line: its comma ends the previous argument's line
    pl, pc = prev[2], prev[3]
    rest = lines[pl][pc:]
    k = len(rest) - len(rest.lstrip())
    if not rest[k:].startswith(","):
        return None
    return [replace_cols(pl, pc + k, pc + k + 1, ""), replace_cols(ln, c0, c1, "")]


def _definitions(srcroot: Path, subs, name: str) -> int:
    """Lines defining a routine `name` (or an interface body of it) in the ELM sources' text."""
    def_re = re.compile(rf"^\s*(?:(?:pure|elemental|recursive|impure)\s+)*subroutine\s+{re.escape(name)}\b", re.I)
    elm = srcroot / "components" / "elm" / "src"
    files = (sorted(elm.rglob("*.F90")) if elm.is_dir()
             else sorted({Path(s.filepath) if Path(s.filepath).is_absolute() else srcroot / s.filepath for s in subs}))
    return sum(1 for f in files if "external_models" not in f.parts
               for line in f.read_text(errors="replace").split("\n") if def_re.match(line.split("!", 1)[0]))


@dataclass
class Options:
    names: list[str] = field(default_factory=lambda: ["bounds"])  # dummies to remove when unused
    routines: list[str] = field(default_factory=list)  # module::name (default: the analyzed/active ones)


@transform
class DropDummies(Transform):
    """
    Matches: dummies named in `--names` (default: bounds) that nothing in their routine uses.
    Changes: removes them from the routine's header and declarations and the matching actual from
             every call (positional or keyword).
    Refuses: routines with an unknown caller, an interface body or a same-named routine elsewhere,
             functions, internal procedures, declarations shared with other names, arguments laid
             out over several lines.
    """

    name = "drop-dummies"
    summary = "remove unused dummies (default: bounds) from routines and their calls"
    Options = Options

    def build(self, ctx, opts) -> Plan:
        from spel.scripts.fortran_parser.spel_ast import (KeywordArgument, SubCallStatement, VariableDecl,
                                                          to_actual_arg)
        from spel.scripts.registrations import iter_statements, parse_routine

        out = Plan(self.name)
        names = {n.lower() for n in opts.names}
        trees: dict[str, Optional[SubroutineDefinitionConstruct]] = {}

        def tree_of(sub: Subroutine) -> Optional[SubroutineDefinitionConstruct]:
            if sub.id not in trees:
                try:
                    trees[sub.id] = parse_routine(ctx.path(sub), sub.startline, sub.endline, keep_spans=True,
                                                  skip_io=True, cpp=True)
                except (Exception, SystemExit):  # noqa: BLE001 - the parser exits on errors
                    trees[sub.id] = None
            return trees[sub.id]

        calls: dict[str, dict] = defaultdict(dict)  # callee -> (caller, ln) -> dummy -> argn
        for s in ctx.fut.subroutine_dict.values():
            ra = s.record_access
            for b in ra.bindings if ra is not None else []:
                calls[b.callee].setdefault((b.caller, b.ln), {})[b.dummy.lower()] = b.argn
        all_subs = list(ctx.fut.subroutine_dict.values())
        edits: dict[Path, list] = defaultdict(list)
        for sub in ctx.routines(opts.routines or None):
            if sub.record is None:
                continue
            tree = None
            for d in sorted(names):
                decl = sub.record.declaration(d)
                if decl is None or decl.intent is None:
                    continue
                tree = tree or tree_of(sub)
                why = self._unused(ctx, sub, tree, d)
                if why:
                    out.add(Refusal(f"{sub.id}:{d}", why))
                    continue
                name = sub.id.split("::")[-1]
                sites = calls.get(sub.id, {})
                ntext = text_call_count(ctx.srcroot, all_subs, name)
                if ntext != len(sites) or _definitions(ctx.srcroot, all_subs, name) != 1:
                    out.add(Refusal(f"{sub.id}:{d}", f"{name}: {len(sites)} analyzed calls, {ntext} in the sources "
                                                     f"(or defined more than once / bound as a procedure)"))
                    continue
                mine: list[tuple[Path, object]] = []
                ok = self._header_edits(ctx, sub, tree, d, mine, VariableDecl, iter_statements)
                for (caller_id, ln), dummies in sorted(sites.items()):
                    if not ok:
                        break
                    caller = ctx.fut.subroutine_dict.get(caller_id)
                    ctree = tree_of(caller) if caller is not None else None
                    if ctree is None:
                        ok = False
                        break
                    ok = self._call_edit(ctx, caller, ctree, ln, name, d, sub, mine, SubCallStatement,
                                         KeywordArgument, to_actual_arg, iter_statements)
                if not ok:
                    out.add(Refusal(f"{sub.id}:{d}", "an argument or declaration layout this pass does not handle"))
                    continue
                for path, e in mine:
                    edits[path].append(e)
                out.notes.append(f"{sub.id}: {d} removed ({len(sites)} calls)")
        for path, es in sorted(edits.items()):
            out.add(Change(path, es, where=path.name, why="unused dummies removed"))
        return out

    @staticmethod
    def _unused(ctx, sub: Subroutine, tree: Optional[SubroutineDefinitionConstruct], d: str) -> str:
        from spel.scripts.fortran_parser.spel_ast import Identifier, VariableDecl
        from spel.scripts.registrations import iter_statements
        from spel.transforms.compact.rewrite import iter_expressions

        if tree is None:
            return "does not parse with spans"
        if tree.contains:
            return "has internal procedures"
        if tree.is_function:
            return "a function"
        if any(a.base.lower() == d for a in sub.record.accesses()):
            return "used (analysis)"
        lines = ctx.lines(ctx.path(sub))
        uses = TextUses(sub, lines, tree)
        if uses.bare.get(d) or uses.subscripted.get(d):
            return "used"
        if _word_in(d, uses.io_text) or _word_in(d, uses.directive_text):
            return "used in an I/O statement or a directive"
        for stmt, _ in iter_statements(tree.body):
            if isinstance(stmt, VariableDecl):
                if any(isinstance(e, Identifier) and e.value.lower() == d for e in iter_expressions(stmt)):
                    return "used in a declaration"
                for ent in stmt.entities:
                    if ent.token.literal.lower() != d and any(
                            isinstance(e, Identifier) and e.value.lower() == d for e in iter_expressions(ent.bounds)):
                        return "used in a declaration"
        return ""

    @staticmethod
    def _header_edits(ctx, sub: Subroutine, tree: SubroutineDefinitionConstruct, d: str, mine: list,
                      VariableDecl, iter_statements) -> bool:
        lines = ctx.lines(ctx.path(sub))
        path = ctx.path(sub)
        args = [a.lower() for a in tree.args]
        spans = tree.arg_spans
        if d not in args or not spans or len(spans) != len(args):
            return False
        e = remove_arg(lines, spans, args.index(d))
        if e is None:
            return False
        mine += [(path, x) for x in (e if isinstance(e, list) else [e])]
        for stmt, _ in iter_statements(tree.body):
            if isinstance(stmt, VariableDecl) and any(ent.token.literal.lower() == d for ent in stmt.entities):
                ent = next(ent for ent in stmt.entities if ent.token.literal.lower() == d)
                span = ent.span
                if len(stmt.entities) != 1 or span is None:
                    return False
                ln = span[0]
                if statement_start(lines, ln) != ln or lines[ln].split("!", 1)[0].rstrip().endswith("&"):
                    return False
                mine.append((path, delete(ln, why=f"drop-dummies {d}")))
                return True
        return False

    @staticmethod
    def _call_edit(ctx, caller: Subroutine, ctree: SubroutineDefinitionConstruct, ln: int, name: str, d: str,
                   callee: Subroutine, mine: list, SubCallStatement, KeywordArgument, to_actual_arg,
                   iter_statements) -> bool:
        from spel.scripts.fortran_parser.spel_ast import Identifier

        lines = ctx.lines(ctx.path(caller))
        hits = []
        for stmt, _ in iter_statements(ctree.body):
            if isinstance(stmt, SubCallStatement):
                fn = stmt.function
                fname = fn.function.value if isinstance(fn.function, Identifier) else ""
                if fname.lower() == name.lower() and fn.span is not None and statement_start(lines, fn.span[0]) == ln:
                    hits.append(fn)
        if len(hits) != 1:
            return False
        fn = hits[0]
        actuals = [to_actual_arg(a) for a in fn.args]
        spans = [a.span for a in actuals]
        idx = next((i for i, a in enumerate(actuals) if isinstance(a, KeywordArgument) and a.keyword.lower() == d), None)
        if idx is None:
            pos = _dummy_position(ctx, callee, d)
            if pos is None or pos >= len(actuals) or any(isinstance(a, KeywordArgument) for a in actuals[:pos + 1]):
                return False
            idx = pos
        e = remove_arg(lines, spans, idx)
        if e is None:
            return False
        mine += [(ctx.path(caller), x) for x in (e if isinstance(e, list) else [e])]
        return True


def _dummy_position(ctx, sub, d: str) -> Optional[int]:
    from spel.scripts.registrations import parse_routine

    try:
        tree = parse_routine(ctx.path(sub), sub.startline, sub.endline, skip_io=True, cpp=True)
    except (Exception, SystemExit):  # noqa: BLE001
        return None
    args = [a.lower() for a in tree.args]
    return args.index(d) if d in args else None
