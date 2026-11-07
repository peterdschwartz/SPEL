"""
Split large Fortran modules: move type definitions (with their procedures)
into new modules, keep the module variables (the instances) where they are,
and rewrite every `use` that names a moved entity.

Inputs are SPEL's analysis and parse tree, never the text:

  * the module's ModuleConstruct (spel_parser) gives its pieces and their
    line spans: head `use` statements, type definitions, module variables,
    directives, subprograms; comment lines before a piece belong to it,
  * each subprogram's SubroutineRecord (scope walk, built on demand for
    routines outside elm_drv's tree) gives the module-level names it
    references: globals, calls, functions, the types of its declarations;
    the parse of its declarations adds kinds, bounds and initializers,
  * ModuleScopes (module_resolver) says which `use` provides a name.

Placement (a Plan): a type goes where the plan says; a subprogram goes with
the type it is bound to, else with the type of its first dummy (the
nutrient base routines), else stays; a module variable that isn't a
type's instance goes with its users when they all move to one module.

Refused: a moved piece that references something staying in its source
module (the source will use the new module: a cycle), a module procedure
without analysis, preprocessor conditionals in the specification part.

Every new module and every rewritten source gets the `use` lines its code
needs: an original statement whose names are all needed is kept as
written, others are rebuilt with just the needed names. Client `use`
statements of a split module (anywhere under the source root) are split by
where each name now lives; names the source re-exports from elsewhere
stay with it.
"""

from __future__ import annotations

import difflib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from spel.scripts.fortran_modules import use_items
from spel.scripts.fortran_parser.lexer import Lexer
from spel.scripts.fortran_parser.spel_ast import (
    AccessStatement,
    DirectiveStatement,
    ImplicitNoneStatement,
    MacroIf,
    ModuleConstruct,
    SaveStatement,
    SubroutineDefinitionConstruct,
    TypeDef,
    UseStatement,
    VariableDecl,
)
from spel.scripts.fortran_parser.spel_parser import Parser
from spel.scripts.fortran_parser.symbols import FORTRAN_INTRINSICS, Origin
from spel.scripts.fortran_parser.tokens import TokenTypes as Tok
from spel.scripts.module_resolver import ModuleResolver, ModuleScopes, module_head
from spel.scripts.types import LineTuple, LogicalLineIterator
from spel.transforms.nutrient.tree import names_in, split_code, statement_at, statement_names
from spel.transforms.core.edits import EditSet, changed_lines, diff_edits, replace
from spel.transforms.core.fortran import wrap_items
from spel.transforms.core.registry import transform
from spel.transforms.core.transform import Refusal, Transform

INTRINSIC_TYPES = {"integer", "real", "logical", "character", "complex", "double", "doubleprecision"}
FORTRAN_SUFFIXES = (".F90", ".f90", ".F", ".f")


class SplitError(Exception):
    pass


@dataclass
class Piece:
    """A top-level piece of a module: its node, the names it defines, its lines."""

    kind: str  # use | type | var | directive | sub | other
    node: object
    first: int  # first line, its leading comments included
    start: int  # the statement's own first line
    last: int
    defines: list[str] = field(default_factory=list)
    home: Optional[str] = None  # new module, None: stays
    passed: list[tuple[str, str, str]] = field(default_factory=list)  # (instance, type, intent) made dummies
    copies: set[str] = field(default_factory=set)  # new modules that get their own copy (a parameter)


@dataclass
class Source:
    name: str  # module name as written
    path: Path
    lines: list[str]
    tree: ModuleConstruct
    head: list[Piece]
    subs: list[Piece]

    @property
    def key(self) -> str:
        return self.name.lower()

    def pieces(self):
        return self.head + self.subs


Plan = Callable[[str, str], Optional[str]]  # (source module, type name) -> new module or None


def parse_module_file(path: Path) -> tuple[list[str], ModuleConstruct]:
    lines = path.read_text(errors="replace").split("\n")
    # `#include` lines carry no Fortran (CH4Mod's shr_assert.h): blank them for the parser only
    prog = Parser(
        lines=[LineTuple(line="" if l.lstrip().startswith("#include") else l, ln=i) for i, l in enumerate(lines)],
        keep_directives=True,
        keep_string_case=True,
    ).parse_program()
    mods = [s for s in prog.statements if isinstance(s, ModuleConstruct)]
    if len(mods) != 1:
        raise SplitError(f"{path}: expected one module, found {len(mods)}")
    return lines, mods[0]


def is_comment(line: str) -> bool:
    code, _ = split_code(line)
    return not code.strip()


def _pieces(lines: list[str], nodes: list, floor: int) -> list[Piece]:
    out: list[Piece] = []
    prev_last = floor
    for n in nodes:
        start = n.lineno
        match n:
            case TypeDef():
                kind, last, defines = "type", n.end_ln, [n.name.lower()]
            case SubroutineDefinitionConstruct():
                kind, last, defines = "sub", n.end_ln, [str(n.name).lower()]
            case UseStatement():
                kind, last, defines = "use", statement_at(lines, start)[0], []
            case VariableDecl():
                kind, last = "var", statement_at(lines, start)[0]
                defines = [e.token.literal.lower() for e in n.entities]
            case DirectiveStatement():
                kind, last, defines = "directive", start, []
            case MacroIf():
                raise SplitError(f"line {start + 1}: preprocessor conditional in the module's specification part")
            case _:
                kind, last, defines = "other", statement_at(lines, start)[0], []
        first = start
        while first - 1 > prev_last and is_comment(lines[first - 1]):
            first -= 1
        # blank lines right after the previous piece stay with it
        while first < start and not lines[first].strip():
            first += 1
        out.append(Piece(kind, n, first, start, last, defines))
        prev_last = last
    return out


def load_source(path: Path) -> Source:
    lines, tree = parse_module_file(path)
    head = _pieces(lines, tree.spec.statements, tree.lineno)
    floor = tree.contains_ln if tree.contains_ln is not None else tree.end_ln
    subs = _pieces(lines, tree.subprograms, floor)
    return Source(str(tree.name), path, lines, tree, head, subs)


def spelled(lines: list[str], name: str) -> str:
    """`name` as the code in `lines` writes it (Fortran is case-insensitive; keep the author's case)."""
    low = name.lower()
    for line in lines:
        code = split_code(line)[0]
        i = code.lower().find(low)
        while i >= 0:
            before = code[i - 1] if i else " "
            after = code[i + len(low)] if i + len(low) < len(code) else " "
            if not (before.isalnum() or before == "_") and not (after.isalnum() or after == "_"):
                return code[i : i + len(low)]
            i = code.lower().find(low, i + 1)
    return name


def wrap(head: str, items: list[str], width: int = 100) -> list[str]:
    """`head` followed by the comma-separated items, continued with `&` past `width`."""
    return wrap_items(head, items, width)


def squeeze(lines: list[str]) -> list[str]:
    """Runs of blank lines (left where pieces were removed) as one."""
    out = []
    for l in lines:
        if not l.strip() and out and not out[-1].strip():
            continue
        out.append(l)
    return out


def idents(text: str) -> set[str]:
    """Names in a bit of Fortran the parser keeps as text (a kind, a directive clause)."""
    if not text:
        return set()
    lex = Lexer(LogicalLineIterator([LineTuple(line=text, ln=0)], log_name="split"))
    out = set()
    while True:
        tok = lex.next_token()
        if tok.token == Tok.EOF:
            return out
        if tok.token == Tok.IDENT:
            out.add(tok.literal.lower())


def _decl_text(src: Source, p: Piece) -> str:
    return " ".join("".join(src.lines[p.start : p.last + 1]).split()).lower()


def twin_source(sp: "Splitter", home: str, piece: Piece) -> str:
    return next(k for k, q in sp.new_modules[home] if q is piece)


def is_parameter(node) -> bool:
    attrs = getattr(getattr(node, "attrs", None), "attrs", None) or []
    return any(str(a).replace(" ", "").lower().startswith("parameter") for a in attrs)


def decl_names(node: VariableDecl) -> set[str]:
    """Module-level names a declaration needs: its derived type, kind, bounds, initializers."""
    out: set[str] = set()
    t = node.var_type
    lit = t.token.literal.lower()
    if lit not in INTRINSIC_TYPES and lit not in ("type", "class"):
        out.add(lit)  # type(name) / class(name): the parser keeps the type name
    out |= idents(t.kind or "") - {"kind", "len"}
    if t.len_ is not None:
        out |= names_in(t.len_)
    for a in node.attrs.attrs if node.attrs else []:
        out |= names_in(a) - {"dimension", "intent", "in", "out", "inout", "pointer", "target",
                              "allocatable", "public", "private", "save", "parameter", "optional"}
    for e in node.entities:
        for b in e.bounds:
            out |= names_in(b)
        if e.init is not None:
            out |= names_in(e.init) - {"null"}
    return out


def tree_names(node) -> set[str]:
    """
    Base names in every statement under `node` (parse tree). The scope walk doesn't record
    the arguments of calls to external procedures (`xtype=ncd_double` in a restartvar call).
    """
    out: set[str] = set()
    stack = [node]
    seen = set()
    while stack:
        n = stack.pop()
        if id(n) in seen:
            continue
        seen.add(id(n))
        if isinstance(n, list):
            stack.extend(n)
            continue
        if not hasattr(n, "__dict__"):
            continue
        out |= statement_names(n)
        for attr in ("exprs", "unit", "fmt"):  # write/print
            v = getattr(n, attr, None)
            for e in v if isinstance(v, list) else [v]:
                if e is not None and not isinstance(e, str):
                    out |= names_in(e)
        for v in vars(n).values():
            if isinstance(v, list) or type(v).__name__ in ("BlockStatement", "ElseIf", "Else", "CaseBlock", "ElseWhere") \
                    or (hasattr(v, "statement_node") and not isinstance(v, str)):
                stack.append(v)
    return out


class Splitter:
    """
    sources:  module names to split (lower case)
    plan:     where each type goes
    """

    def __init__(self, fut, sources: list[str], plan: Plan, srcroot: Path, scan_root: Optional[Path] = None):
        self.fut = fut
        self.md, self.sd = fut.module_dict, fut.subroutine_dict
        self.plan = plan
        self.srcroot = Path(srcroot)
        self.scan_root = Path(scan_root) if scan_root else self.srcroot / "components"
        self.scopes = ModuleScopes(self.md, self.sd)
        self.sources: dict[str, Source] = {}
        for m in sources:
            if m not in self.md:
                raise SplitError(f"module {m} is not in the analysis")
            path = Path(self.md[m].filepath)
            self.sources[m] = load_source(path if path.is_absolute() else self.srcroot / path)
        self.spelling: dict[str, str] = {}  # lower-case name -> as the sources write it
        for src in self.sources.values():
            self.spelling[src.key] = spelled([src.lines[src.tree.lineno]], src.key)
            for p in src.pieces():
                text = src.lines[p.start : p.last + 1]
                for d in p.defines:
                    self.spelling.setdefault(d, spelled(text, d))
                if p.kind == "use":
                    self.spelling.setdefault(p.node.module.lower(), spelled(text, p.node.module))
                    for local, remote in use_items(p.node):
                        self.spelling.setdefault(local, spelled(text, local))
                        self.spelling.setdefault(remote, spelled(text, remote))
        self.home: dict[str, str] = {}  # entity -> module it lives in after the split (lower case)
        self.where: dict[str, tuple[str, Piece]] = {}  # entity -> (source module, piece)
        self.new_modules: dict[str, list[tuple[str, Piece]]] = {}  # new name -> pieces in order
        self.notes: list[str] = []

    # ---- placement -------------------------------------------------------
    def assign(self) -> None:
        bound: dict[str, str] = {}  # procedure -> type it is bound to
        for key, src in self.sources.items():
            for p in src.pieces():
                for d in p.defines:
                    self.where[d] = (key, p)
            for p in src.head:
                if p.kind == "type":
                    p.home = self.plan(key, p.node.name.lower())
                    for proc in p.node.methods.statements if p.node.methods else []:
                        if getattr(proc, "name", None):
                            bound[str(proc.name).lower()] = p.node.name.lower()
        types = {p.node.name.lower(): p for s in self.sources.values() for p in s.head if p.kind == "type"}
        for key, src in self.sources.items():
            for p in src.subs:
                name = p.defines[0]
                t = bound.get(name) or self.first_dummy_type(key, name)
                if t in types:
                    p.home = types[t].home
        self.place_variables()
        self.place_directives()
        self.place_arguments()
        for key, src in self.sources.items():
            for p in src.pieces():
                for d in p.defines:
                    self.home[d] = (p.home or src.name).lower()
                if p.home:
                    self.new_modules.setdefault(p.home, []).append((key, p))
                for h in sorted(p.copies):
                    # one copy per module: several sources may define the same parameter
                    twin = next(
                        (q for k2, q in self.new_modules.get(h, []) if q.copies and set(q.defines) == set(p.defines)),
                        None,
                    )
                    if twin is not None:
                        if _decl_text(self.sources[key], p) != _decl_text(self.sources[twin_source(self, h, twin)], twin):
                            raise SplitError(f"parameter {p.defines} differs between the sources copying it to {h}")
                        continue
                    self.new_modules.setdefault(h, []).append((key, p))
        directory = next(iter(self.sources.values())).path.parent
        clash = [n for n in self.new_modules if n.lower() in self.md or (directory / f"{n}.F90").exists()]
        if clash:
            raise SplitError(f"modules {clash} exist already (split before? run on the unsplit sources)")

    def record(self, mod: str, name: str):
        sid = f"{mod}::{name}"
        sub = self.sd.get(sid)
        if sub is None:
            raise SplitError(f"{sid}: not in the analysis")
        if sub.record is None:
            if sub.syntax_tree is None:
                sub.parse()
            sub.walk_syntax_tree(self.scopes)
        return sub.record

    def first_dummy_type(self, mod: str, name: str) -> Optional[str]:
        rec = self.record(mod, name)
        if not rec.dummy_args:
            return None
        return (rec.dummy_args[0].type or "").lower() or None

    def place_arguments(self) -> None:
        """
        A moved routine that reads an instance staying in its source (which will use the
        routine's new module) gets the instance as a dummy argument of the same name: the
        body is unchanged, and every call passes the instance (`col_ws_init` reads `col_es`).
        """
        for key, src in self.sources.items():
            for p in src.subs:
                if p.home is None:
                    continue
                rec = self.record(key, p.defines[0])
                for n in sorted(self.sub_names(key, p)):
                    w = self.where.get(n)
                    if not w or w[0] != key or w[1].home is not None or w[1].kind != "var":
                        continue
                    if w[1].node.var_type.token.literal.lower() not in self.where:
                        continue  # not an instance of a split module's type: the cycle check reports it
                    if p.node.result and p.node.result != p.node.name:
                        raise SplitError(f"{p.defines[0]}: would pass {n}, but has a result clause")
                    statuses = {e.status for e in rec.events if type(e).__name__ == "Access" and e.base == n}
                    intent = "in" if statuses <= {"r"} else "inout"
                    p.passed.append((n, w[1].node.var_type.token.literal, intent))
                    self.notes.append(f"{p.defines[0]}: {n} passed as an argument (intent({intent}))")

    def place_variables(self) -> None:
        """
        Module variables of no defined type follow their users when those all move to one
        module. A `parameter` whose users spread over several modules is copied into each new
        module that uses it (a constant has no state); the source keeps its own.
        """
        users: dict[str, set] = defaultdict(set)
        for key, src in self.sources.items():
            for p in src.subs:
                for n in self.sub_names(key, p) & {d for q in src.head if q.kind == "var" for d in q.defines}:
                    users[n].add(p.home)
        self.copied: dict[str, set[str]] = {}
        for key, src in self.sources.items():
            types = {q.node.name.lower() for q in src.head if q.kind == "type"}
            for p in src.head:
                if p.kind != "var" or p.node.var_type.token.literal.lower() in types:
                    continue  # instances stay
                homes = set.union(*(users.get(d, set()) for d in p.defines)) if p.defines else set()
                if len(homes) == 1 and None not in homes:
                    p.home = homes.pop()
                elif len(homes) > 1 and is_parameter(p.node):
                    p.copies = {h for h in homes if h is not None}
                    for d in p.defines:
                        self.copied[d] = {h.lower() for h in p.copies}
                    self.notes.append(f"{src.name}: parameter {', '.join(p.defines)} copied to {sorted(p.copies)}")

    def place_directives(self) -> None:
        """A directive naming only moved variables goes with them."""
        for key, src in self.sources.items():
            for p in src.head:
                if p.kind != "directive":
                    continue
                named = idents(p.node.text.split("!", 2)[-1]) & set(self.where)
                homes = {self.where[n][1].home for n in named if self.where[n][0] == key}
                if len(homes) > 1:
                    raise SplitError(f"{src.name} line {p.start + 1}: directive names variables going to {homes}")
                if homes:
                    p.home = homes.pop()

    # ---- names -----------------------------------------------------------
    def resolver(self, mod: str) -> ModuleResolver:
        return ModuleResolver(mod, self.md, self.sd, scopes=self.scopes)

    def sub_names(self, mod: str, p: Piece) -> set[str]:
        """Module-level names a subprogram references (from its record and declarations)."""
        rec = self.record(mod, p.defines[0])
        own = set(rec.declarations) | {p.defines[0]}
        own |= {str(s.name).lower() for s in p.node.contains}
        local_uses = set()
        for st in p.node.body.statements:
            if isinstance(st, UseStatement):
                local_uses |= {l for l, _ in use_items(st)} if st.has_only else set()
        out: set[str] = set()
        for e in rec.events:
            kind = type(e).__name__
            if kind == "Access" and e.origin in (Origin.GLOBAL, Origin.EXTERNAL):
                out.add(e.base.lower())
            elif kind == "CallEvent" and "%" not in e.name:
                out.add(e.name.lower())
            elif kind == "FunctionRef" and not e.intrinsic:
                out.add(e.name.lower())
        for _, d in rec.declarations.items():
            t = (d.type or "").lower()
            if t and t.split("(")[0] not in INTRINSIC_TYPES:
                out.add(t)
        for st in p.node.body.statements:
            if isinstance(st, VariableDecl):
                out |= decl_names(st)
        res = self.resolver(mod)
        out |= {n for n in tree_names(p.node.body) - own - FORTRAN_INTRINSICS
                if res.lookup(n) is not None or res.fallback(n) is not None}
        return out - own - local_uses

    def piece_names(self, mod: str, p: Piece) -> set[str]:
        match p.kind:
            case "sub":
                return self.sub_names(mod, p) - {n for n, _, _ in p.passed}
            case "type":
                out = set()
                for m in p.node.members.statements:
                    if isinstance(m, VariableDecl):
                        out |= decl_names(m)
                for proc in p.node.methods.statements if p.node.methods else []:
                    if getattr(proc, "name", None):
                        out.add(str(proc.name).lower())
                return out
            case "var":
                return decl_names(p.node)
            case "directive":
                return idents(p.node.text.split("!", 2)[-1]) & set(self.where)
        return set()

    def provider(self, mod: str, name: str) -> Optional[tuple[str, Optional[UseStatement], str]]:
        """
        Where `name`, referenced in source module `mod`, comes from: (module, the head use
        statement providing it or None for one of the split modules' own entities, remote name).
        None: not a module-level name (local, intrinsic).
        """
        if name in self.where and self.where[name][0] == mod:
            return self.home[name], None, name
        for st in self.head_uses(mod):
            if st.has_only and st.module.lower() in self.sources:
                for local, remote in use_items(st):
                    if local == name and self.where.get(remote, (None,))[0] == st.module.lower():
                        return self.home[remote], None, remote
        if name in FORTRAN_INTRINSICS:
            return None
        res = self.resolver(mod)
        sym = res.lookup(name)
        fallback = sym is None and res.fallback(name) is not None
        if sym is None and not fallback:
            return None
        for st in self.head_uses(mod):
            if st.has_only:
                for local, remote in use_items(st):
                    if local == name:
                        return st.module.lower(), st, remote
            elif not fallback:
                visible, _ = self.scopes.use_symbols(st)
                if name in visible:
                    return st.module.lower(), st, name
        if fallback or (sym is not None and sym.origin == Origin.EXTERNAL):
            whole = [st for st in self.head_uses(mod) if not st.has_only and not self.scopes.is_available(st.module)]
            if whole:
                return whole[0].module.lower(), whole[0], name
        return None

    def head_uses(self, mod: str) -> list[UseStatement]:
        return [p.node for p in self.sources[mod].head if p.kind == "use"]

    def requirements(self, pieces: list[tuple[str, Piece]], here: str) -> dict:
        """
        For pieces that will live in module `here`: {(module, whole use statement or None):
        [(local, remote)]}, in the order the names are first needed.
        """
        need: dict = {}
        for mod, p in pieces:
            for name in sorted(self.piece_names(mod, p)):
                if here.lower() in getattr(self, "copied", {}).get(name, ()):
                    continue  # a parameter copied into this module
                prov = self.provider(mod, name)
                if prov is None:
                    continue
                module, stmt, remote = prov
                if stmt is None and module == here.lower():
                    continue
                key = (module, stmt if stmt is not None and not stmt.has_only else None)
                items = need.setdefault(key, [])
                if stmt is not None and not stmt.has_only:
                    continue  # a whole use
                if (name, remote) not in items:
                    items.append((name, remote))
                # defined assignments/operators are used without their names (`x = nan`):
                # they come along with any name of their statement
                if stmt is not None:
                    for g in use_items(stmt):
                        if g[0].startswith(("assignment(", "operator(")) and g not in items:
                            items.append(g)
        return need

    # ---- output ----------------------------------------------------------
    def module_name(self, key: str) -> str:
        for n in self.new_modules:
            if n.lower() == key:
                return n
        return self.spelling.get(key, key)

    def item(self, local: str, remote: str) -> str:
        sp = lambda n: self.spelling.get(n, n)
        return sp(local) if local == remote else f"{sp(local)} => {sp(remote)}"

    def use_text(self, need: dict, indent: str = "  ") -> list[str]:
        """The `use` lines for `need` (generated)."""
        out, whole_done = [], set()
        for (module, whole), items in need.items():
            if whole is not None:
                if module not in whole_done:
                    out.append(f"{indent}use {self.module_name(module)}")
                    whole_done.add(module)
                continue
            out += wrap(f"{indent}use {self.module_name(module)}, only : ", [self.item(l, r) for l, r in items])
        return out

    def build(self) -> dict[Path, list[str]]:
        self.assign()
        files: dict[Path, list[str]] = {}
        self.deps: dict[str, set[str]] = {}
        directory = next(iter(self.sources.values())).path.parent
        # every module's needs first: a moved entity another module needs must be public
        needs = {new.lower(): self.requirements(pieces, new) for new, pieces in self.new_modules.items()}
        needs |= {key: self.source_need(key, src) for key, src in self.sources.items()}
        self.deps = {m: {d for d, _ in need} for m, need in needs.items()}
        self.exported: dict[str, set[str]] = defaultdict(set)
        for need in needs.values():
            for (m, whole), items in need.items():
                if whole is None:
                    self.exported[m] |= {r for _, r in items}
        for new, pieces in self.new_modules.items():
            files[directory / f"{new}.F90"] = self.new_module(new, pieces, needs[new.lower()])
        for key, src in self.sources.items():
            files[src.path] = self.rewrite_source(key, src, needs[key])
        self.check_cycles()
        self.files = files
        self.client_edits = self.rewrite_clients(files)
        return files

    def sub_text(self, mod: str, p: Piece) -> list[str]:
        """The subprogram's lines, with the instances it is passed added as dummies."""
        lines = self.sources[mod].lines[p.first : p.last + 1]
        if not p.passed:
            return lines
        off = p.first
        names = ", ".join(n for n, _, _ in p.passed)
        # the subprogram statement: before the `)` closing its dummy list (no result clause)
        last, _ = statement_at(self.sources[mod].lines, p.start)
        code, comment = split_code(lines[last - off])
        close = code.rstrip().rfind(")")
        if close < 0:
            raise SplitError(f"{p.defines[0]}: no dummy list to extend")
        sep = ", " if p.node.args else ""
        lines[last - off] = code[:close] + sep + names + code[close:] + comment
        # declarations after the last dummy's
        rec = self.record(mod, p.defines[0])
        decl_ln = max(d.ln for d in rec.dummy_args) if rec.dummy_args else last
        decl_last, _ = statement_at(self.sources[mod].lines, decl_ln)
        ref = lines[decl_last - off] if rec.dummy_args else lines[last - off]
        indent = ref[: len(ref) - len(ref.lstrip())] or "    "
        decls = [f"{indent}type({t}), intent({i}) :: {n}  ! the instance, passed in (spel split)" for n, t, i in p.passed]
        at = decl_last - off + 1
        return lines[:at] + decls + lines[at:]

    def check_cycles(self) -> None:
        """
        A use cycle through a new module: its dependencies are its `use` lines, a rewritten
        source's too, other modules' from the analysis (one that uses a split module may
        end up using any of its new modules).
        """
        from_source = defaultdict(set)
        for n, pieces in self.new_modules.items():
            for m, _ in pieces:
                from_source[m].add(n.lower())

        def deps(m: str) -> set[str]:
            if m in self.deps:
                return self.deps[m]
            out = set(getattr(self.md.get(m), "modules", {}) or {})
            for s in out & set(self.sources):
                out |= from_source[s]
            return {x.lower() for x in out}

        new = {n.lower() for n in self.new_modules}
        for start in new:
            stack, seen = [(start, [start])], set()
            while stack:
                m, path = stack.pop()
                for d in deps(m):
                    if d == start:
                        raise SplitError("use cycle: " + " -> ".join(path + [d]))
                    if d not in seen:
                        seen.add(d)
                        stack.append((d, path + [d]))

    def new_module(self, name: str, pieces: list[tuple[str, Piece]], need: dict) -> list[str]:
        origins = sorted({self.module_name(m) for m, _ in pieces})
        heads = {m: module_head(self.md[m]) for m, _ in pieces}
        for m, head in heads.items():
            stated = {n for p in self.sources[m].head if isinstance(p.node, AccessStatement) for n in p.node.names}
            moved = {d for mm, p in pieces if mm == m for d in p.defines}
            if stated & moved:
                raise SplitError(f"{self.sources[m].name}: access statements name moved entities {sorted(stated & moved)}")
        private = any(h.default_private for h in heads.values())
        # with mixed defaults: private, and listed public the names public in their source or
        # needed by another module (not those whose definition has the attribute: access is
        # given once)
        public = []
        if private:
            wanted = self.exported.get(name.lower(), set())
            for m, p in pieces:
                h = heads[m]
                public += [d for d in p.defines if (h.is_public(d) or d in wanted) and d not in h.access]
        # `#include`s of the sources (shr_assert.h) whose macros the moved text uses
        includes: list[str] = []
        for m in sorted({m for m, _ in pieces}):
            src = self.sources[m]
            moved_text = "\n".join(l for mm, pc in pieces if mm == m for l in src.lines[pc.first:pc.last + 1])
            for l in src.lines:
                if l.lstrip().startswith("#include") and l not in includes:
                    header = l.split("#include", 1)[1].strip().strip('"<>').split(".")[0].upper()
                    if header.startswith("SHR_ASSERT") and "SHR_ASSERT" in moved_text.upper() or header not in ("SHR_ASSERT",):
                        includes.append(l.strip())
        # the includes go inside the module, as in CH4Mod: shr_assert.h can expand to `use` statements
        out = [
            f"module {name}",
            *includes,
            "  !",
            "  ! !DESCRIPTION:",
            f"  ! Generated by `spel split` from {', '.join(origins)}.",
            "  !",
            *self.use_text(need),
            "  !",
            "  implicit none",
            "  save",
            "  private" if private else "  public",
        ]
        if public:
            out += wrap("  public :: ", [self.spelling.get(d, d) for d in public])
        for m, p in pieces:
            if p.kind != "sub":
                text = self.sources[m].lines[p.first : p.last + 1]
                if p.kind == "type" and out[-1].strip():
                    out.append("")
                out += text
        out += ["", "contains", ""]
        for m, p in pieces:
            if p.kind == "sub":
                out += self.sub_text(m, p) + [""]
        out.append(f"end module {name}")
        return squeeze(out)

    def source_need(self, key: str, src: Source) -> dict:
        staying = [(key, p) for p in src.pieces() if p.home is None and p.kind != "use"]
        need = self.requirements(staying, src.name)
        # names clients import from this module that it re-exports from elsewhere stay available
        for name in sorted(self.reexported(key)):
            prov = self.provider(key, name)
            if prov and prov[1] is not None:
                module, stmt, remote = prov
                items = need.setdefault((module, None if stmt.has_only else stmt), [])
                if stmt.has_only and (name, remote) not in items:
                    items.append((name, remote))
        return need

    def rewrite_source(self, key: str, src: Source, need: dict) -> list[str]:
        moved = [p for p in src.pieces() if p.home is not None]
        uses = [p for p in src.head if p.kind == "use"]
        new_uses = self.source_use_lines(src, need)
        drop = {ln for p in moved for ln in range(p.first, p.last + 1)}
        drop |= {ln for p in uses for ln in range(p.start, p.last + 1)}
        if src.tree.contains_ln is not None and all(p.home for p in src.subs):
            drop.add(src.tree.contains_ln)
        first_use = min((p.start for p in uses), default=src.tree.lineno + 1)
        out = []
        for i, line in enumerate(src.lines):
            if i == first_use:
                out += new_uses
            if i not in drop:
                out.append(line)
        return squeeze(out)

    def source_use_lines(self, src: Source, need: dict) -> list[str]:
        """
        A rewritten source's head uses: each original kept as written when all its names
        are still needed, trimmed when some are, dropped when none; then the rest.
        """
        out = []
        covered = set()
        for p in src.head:
            if p.kind != "use":
                continue
            st, m = p.node, p.node.module.lower()
            if not st.has_only:
                if (m, st) in need:
                    out += src.lines[p.start : p.last + 1]  # its leading comments stay in place
                    covered.add((m, st))
                continue
            wanted = need.get((m, None), [])
            mine = use_items(st)
            got = [i for i in mine if i in wanted]
            covered |= {(m, i) for i in got}
            if got == mine:
                out += src.lines[p.start : p.last + 1]
            elif got:
                indent = src.lines[p.start][: len(src.lines[p.start]) - len(src.lines[p.start].lstrip())]
                out += wrap(f"{indent}use {self.module_name(m)}, only : ", [self.item(l, r) for l, r in got])
        for (module, whole), items in need.items():
            if whole is not None:
                continue
            rest = [i for i in items if (module, i) not in covered]
            if rest:
                out += self.use_text({(module, None): rest})
        return out

    def reexported(self, key: str) -> set[str]:
        """Names clients import from module `key` that aren't its own entities."""
        own = {d for p in self.sources[key].pieces() for d in p.defines}
        return {r for _, _, st in self.client_uses() if st.module.lower() == key and st.has_only
                for _, r in use_items(st) if r not in own}

    # ---- clients ---------------------------------------------------------
    def client_uses(self):
        """(path, first line, UseStatement) of every use of a split module under the scan root."""
        if hasattr(self, "_client_uses"):
            return self._client_uses
        out = []
        targets = set(self.sources)
        for path in sorted(self.scan_root.rglob("*")):
            if path.suffix not in FORTRAN_SUFFIXES or not path.is_file():
                continue
            try:
                text = path.read_text(errors="replace")
            except OSError:
                continue
            if not any(t in text.lower() for t in targets):
                continue
            lines = text.split("\n")
            it = LogicalLineIterator([LineTuple(line=l, ln=i) for i, l in enumerate(lines)], log_name="split")
            for lt in it:
                code = lt.line.strip().lower()
                if not code.startswith("use") or not any(t in code for t in targets):
                    continue
                stmts = Parser(lines=[LineTuple(line=lt.line, ln=lt.ln)]).parse_program().statements
                for st in stmts:
                    if isinstance(st, UseStatement) and st.module.lower() in targets:
                        out.append((path, lt.ln, st))
        self._client_uses = out
        return out

    def client_lines(self, st: UseStatement, indent: str, comment: str, original: list[str] = None) -> list[str]:
        """A client's use of a split module, as uses of the modules its names now live in."""
        key = st.module.lower()
        if not st.has_only:
            mods = [self.sources[key].name] + [n for n in self.new_modules
                                               if any(m == key for m, _ in self.new_modules[n])]
            return [f"{indent}use {m}" for m in mods]
        groups: dict[str, list[tuple[str, str]]] = {}
        for local, remote in use_items(st):
            module = self.home.get(remote, key) if self.where.get(remote, (None,))[0] == key else key
            groups.setdefault(module, []).append((local, remote))
        if len(groups) == 1 and original is not None:
            (module,) = groups
            first = original[0]
            name = spelled([first], key)
            at = split_code(first)[0].lower().find(key, split_code(first)[0].lower().find("use") + 3)
            return [first[:at] + self.module_name(module) + first[at + len(name):]] + original[1:]
        sp = lambda n: spelled(original or [], n)  # the client's spelling
        out = []
        for m, items in groups.items():
            out += wrap(f"{indent}use {self.module_name(m)}, only : ",
                        [sp(l) if l == r else f"{sp(l)} => {sp(r)}" for l, r in items])
        if comment and out:
            out[-1] += f" {comment}"
        return out

    def rewrite_clients(self, files: dict[Path, list[str]]) -> dict[Path, list[str]]:
        """The final text of every file the split writes. `self.client_edit_sets[path]` holds
        the edits of each client (and new or rewritten file) against `files[path]`, or against
        the file on disk for a client the split does not otherwise write."""
        edits: dict[Path, list[str]] = {}
        self.client_edit_sets: dict[Path, EditSet] = {}
        by_path = defaultdict(list)
        for path, ln, st in self.client_uses():
            by_path[path].append((ln, st))
        for path in files:
            by_path.setdefault(path, [])
        for path, uses in by_path.items():
            original = list(files.get(path) or path.read_text(errors="replace").split("\n"))
            lines = list(original)
            if path in files:
                uses = self.reparse_uses(lines)
            self.pass_instances(path, lines, uses)  # edits call lines in place, one line each
            es = EditSet(path).extend(changed_lines(original, lines))
            for ln, st in uses:
                last, _ = statement_at(lines, ln)
                code, comment = split_code(lines[last])
                indent = lines[ln][: len(lines[ln]) - len(lines[ln].lstrip())]
                old = lines[ln : last + 1]
                new = self.client_lines(st, indent, comment.strip(), old)
                if [l.rstrip() for l in old] != [l.rstrip() for l in new] and self.changes(st):
                    es.add(replace(ln, new, n=last + 1 - ln, expect=original[ln : last + 1],
                                   why=f"use {st.module}"))
            if es:
                self.client_edit_sets[path] = es
                edits[path] = es.apply(original)
        for path, lines in files.items():
            if path not in edits:
                edits[path] = lines
        return edits

    def core_plan(self, name: str = "split"):
        """The core Plan of a built split: new modules as new files, the rewritten sources and
        the clients as edits of the files on disk."""
        from spel.transforms.core.transform import Change, Plan

        out = Plan(name, notes=list(self.notes) + self.report().split("\n"))
        for path, lines in sorted(self.client_edits.items()):
            if path in self.files and not path.exists():
                out.new_files[path] = "\n".join(lines)
            elif path in self.files:
                disk = path.read_text(errors="replace").split("\n")
                out.add(Change(path, diff_edits(disk, lines, why="split source"), where=path.name,
                               why="moved definitions out, uses rewritten"))
            else:
                out.add(Change(path, self.client_edit_sets[path].edits, where=path.name,
                               why="uses of moved names rewritten"))
        return out

    def pass_instances(self, path: Path, lines: list[str], uses) -> bool:
        """
        Calls of routines given instance dummies get `name=name` for each: direct calls, and
        type-bound calls through an instance of the type this file imports from a split module.
        """
        targets: dict[str, list] = {}  # routine -> passed
        bindings: dict[str, dict[str, str]] = {}  # type -> binding -> routine
        for key, src in self.sources.items():
            for p in src.head:
                if p.kind == "type" and p.node.methods:
                    for proc in p.node.methods.statements:
                        if getattr(proc, "name", None):
                            alias = str(proc.alias or proc.name).lower()
                            bindings.setdefault(p.node.name.lower(), {})[alias] = str(proc.name).lower()
            for p in src.subs:
                if p.passed:
                    targets[p.defines[0]] = p.passed
        if not targets:
            return False
        imported = {}  # local name -> remote, for names this file imports from a split module
        for _, st in uses:
            if st.has_only:
                for local, remote in use_items(st):
                    if self.where.get(remote, (None,))[0] == st.module.lower():
                        imported[local] = remote
        changed = False
        it = LogicalLineIterator([LineTuple(line=l, ln=i) for i, l in enumerate(lines)], log_name="split")
        for lt in list(it):
            code = lt.line.strip().lower()
            if not code.startswith("call"):
                continue
            try:
                stmts = Parser(lines=[LineTuple(line=lt.line, ln=lt.ln)]).parse_program().statements
            except Exception:
                continue
            for st in stmts:
                func = getattr(st, "function", None)
                callee = str(getattr(func, "function", "")).lower()
                routine = None
                if "%" in callee:
                    obj, _, binding = callee.rpartition("%")
                    remote = imported.get(obj)
                    w = self.where.get(remote) if remote else None
                    if w and w[1].kind == "var":
                        routine = bindings.get(w[1].node.var_type.token.literal.lower(), {}).get(binding)
                elif callee in targets:
                    routine = callee
                if routine not in targets:
                    continue
                missing = [n for n, _, _ in targets[routine] if n not in imported]
                if missing:
                    raise SplitError(f"{path}:{lt.ln + 1}: calls {routine} but doesn't import {missing}")
                last, _ = statement_at(lines, lt.ln)
                c, comment = split_code(lines[last])
                close = c.rstrip().rfind(")")
                extra = ", ".join(f"{n}={n}" for n, _, _ in targets[routine])
                lines[last] = c[:close] + ", " + extra + c[close:] + comment
                self.notes.append(f"{path.name}:{lt.ln + 1}: call {callee} passes {extra}")
                changed = True
        return changed

    def changes(self, st: UseStatement) -> bool:
        key = st.module.lower()
        if not st.has_only:
            return any(m == key for ps in self.new_modules.values() for m, _ in ps)
        return any(self.where.get(r, (None,))[0] == key and self.home.get(r) != key for _, r in use_items(st))

    def reparse_uses(self, lines: list[str]) -> list[tuple[int, UseStatement]]:
        out = []
        it = LogicalLineIterator([LineTuple(line=l, ln=i) for i, l in enumerate(lines)], log_name="split")
        for lt in it:
            code = lt.line.strip().lower()
            if code.startswith("use") and any(t in code for t in self.sources):
                for st in Parser(lines=[LineTuple(line=lt.line, ln=lt.ln)]).parse_program().statements:
                    if isinstance(st, UseStatement) and st.module.lower() in self.sources:
                        out.append((lt.ln, st))
        return out

    # ---- report ----------------------------------------------------------
    def report(self) -> str:
        out = []
        for path, lines in sorted(self.client_edits.items()):
            if path in self.files and not path.exists():
                out.append(f"  new     {len(lines):6d} lines  {path.relative_to(self.srcroot)}")
        for key, src in self.sources.items():
            out.append(f"  source  {len(self.client_edits[src.path]):6d} lines  {src.path.relative_to(self.srcroot)}"
                       f" (was {len(src.lines)})")
        clients = [p for p in self.client_edits if p not in self.files]
        out.append(f"  {len(clients)} other files: use statements rewritten")
        return "\n".join(out)

    def diff(self) -> str:
        out = []
        for path, lines in sorted(self.client_edits.items()):
            old = path.read_text(errors="replace").split("\n") if path.exists() else []
            out += difflib.unified_diff(old, lines, str(path), str(path), lineterm="")
        return "\n".join(out)

    def apply(self) -> list[Path]:
        for path, lines in self.client_edits.items():
            path.write_text("\n".join(lines))
        return sorted(self.client_edits)


# ---- the plan for ELM's subgrid data types -------------------------------
NUTRIENT_ROLES = ("_nutrient_", "_mineral_", "_uptake_", "_retrans_")
LEVELS = {"vegetation": "Vegetation", "column": "Column"}


def elm_data_types_plan(mod: str, type_name: str) -> Optional[str]:
    """
    Nutrient base types and role components (all levels) -> NutrientStateType /
    NutrientFluxType; other vegetation/column types -> <Level>StateTypes /
    <Level>FluxTypes; the rest stay.
    """
    kind = "State" if type_name.endswith("_state") else "Flux" if type_name.endswith("_flux") else None
    if kind is None:
        return None
    if any(r in type_name for r in NUTRIENT_ROLES):
        return f"Nutrient{kind}Type"
    level = type_name.split("_")[0]
    if level in LEVELS:
        return f"{LEVELS[level]}{kind}Types"
    return None


def ch4_type_plan(mod: str, type_name: str) -> Optional[str]:
    """ch4_type (and its procedures) -> Ch4TypeMod, so biogeophysics can use the type without
    depending on CH4Mod (the GPU branch's L1 layer, made by hand before)."""
    return "Ch4TypeMod" if type_name.lower() == "ch4_type" else None


PLANS = {"elm-data-types": (["columndatatype", "vegetationdatatype", "gridcelldatatype"], elm_data_types_plan),
         "ch4-type": (["ch4mod"], ch4_type_plan)}


@dataclass
class Options:
    plan: Optional[str] = field(default=None, metadata={"help": "the split plan (default: elm-data-types)"})


@transform
class Split(Transform):
    """
    Matches: the type definitions (and their procedures) the plan assigns to new modules.
    Changes: moves them out of large modules into new ones (instances stay); every `use` of a
             moved name is rewritten; calls of procedures given instance dummies pass them.
    Refuses: a plan that would make a use cycle, or a module it cannot split (SplitError).
    The plans are PLANS.
    """

    name = "split"
    summary = "move type definitions out of large modules into new ones"
    Options = Options

    def build(self, ctx, opts):
        from spel.transforms.core.transform import Plan

        key = opts.plan or "elm-data-types"
        if key not in PLANS:
            return Plan(self.name).add(Refusal(key, f"no such plan; known: {', '.join(sorted(PLANS))}"))
        mods, plan_fn = PLANS[key]
        try:
            sp = Splitter(ctx.fut, mods, plan_fn, Path(ctx.srcroot))
            sp.build()
        except SplitError as err:
            return Plan(self.name).add(Refusal(key, str(err)))
        return sp.core_plan(self.name)
