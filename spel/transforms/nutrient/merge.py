"""
Merge the code a family's C/N/P procedures share into a base-type routine.

For a binding (``SetValues``, ``ZeroDWT``, ``Init``) of a family whose types
hold the base fields as their ``core`` component (extract.py),
procedures.compare_procedure classifies each statement. The statements that
are generic and touch only base fields move:

  * the base routine ``<base>_<binding>`` is a plain module procedure built
    from one of the routines (the dummy names most of them use): its
    declarations, the moved statements and the loops/ifs around them, with
    ``type(<base>), intent(inout) :: this`` and ``this%core%x`` -> ``this%x``.
    No type-bound procedure and no polymorphic dummy: device code can call it.
    Its dummies are `this` and those its body uses, which every routine must
    have under the same name (dummies correspond by name, never by position;
    Init's carbon_type, ratios to other types, ... stay out),
  * each concrete routine keeps its name, dummies and own statements (and the
    loops/ifs they need) and calls ``<base>_<binding>(this%core, ...)`` first,
    after its ``core%species`` assignment (species.py).

A carbon statement under ``if (.not. use_fates)`` that N and P have
unguarded moves too: the base routine is then built from the carbon one, with
that ``if`` as ``if (.not. (is_carbon(this%species) .and. use_fates))``,
so FATES runs skip exactly what they skipped.

Only statements that write fields and nothing else move (an assignment or
pointer assignment to `this%...`, an allocate of `this%...`): one writing a
local could feed a kept statement. Calls and other statements without a
field stay, once, in the concrete routine; only assignments to a routine's
own locals (loop indices) are copied into both as scaffolding for a loop or
if that both keep.

The base call runs the moved statements before everything the concrete
routine keeps, so a statement moves only if, in every routine, no kept
statement on the same fields and no kept call comes before it, and no copy of
it is under a carbon_type guard (moved unguarded, it would run for C12 too);
otherwise it stays in every routine.

The structure comes from SPEL's parse tree (tree.py): what is a loop, an if
and its branches, a declaration, a call, a directive. OpenACC/OpenMP
directives go with the construct they annotate, and `!$acc routine` is kept
on the base routine. A routine with constructs this doesn't rewrite (select,
where, #if, return, labels) is refused.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from spel.scripts.fortran_parser.spel_ast import AssociateConstruct, Identifier, IfConstruct, VariableDecl
from spel.scripts.registrations import _norm
from spel.transforms.nutrient.common import ORDER, Family
from spel.transforms.nutrient.extract import CORE, ExtractError
from spel.transforms.nutrient.procedures import compare_procedure
from spel.transforms.nutrient.tree import (
    Block,
    Leaf,
    Routine,
    Span,
    assigned_path,
    associations,
    declared_names,
    is_spec_directive,
    declaration_lines,
    designator_path,
    iter_leaves,
    names_in,
    read_routine,
    split_code,
    statement_names,
    use_lines,
    written_paths,
)

DEFAULT_BINDINGS = ("setvalues", "zerodwt")
# carbon's guard for fields FATES doesn't allocate (as it appears in a key), and its base-routine form
FATES_GUARD = "(.not.use_fates)"
FATES_IF_BASE = "if (.not. (is_carbon(this%species) .and. use_fates)) then"
_CORE_RE = re.compile(rf"\bthis(\s*)%(\s*){CORE}\s*%", re.I)


def drop_fates_guard(key: str) -> tuple[str, bool]:
    """(the key without carbon's `.not. use_fates` guard, whether it had it)."""
    guards, _, body = key[1:].partition("] ")
    terms = [t for t in guards.split(" && ") if t]
    kept = [t for t in terms if t != FATES_GUARD]
    return f"[{' && '.join(kept)}] {body}", len(kept) != len(terms)


def is_fates_if(node) -> bool:
    return isinstance(node, IfConstruct) and _norm(node.condition).strip("()") == ".not.use_fates"


def _first_span(node) -> Span:
    return node.span if isinstance(node, Leaf) else node.head


def _spans(nodes):
    for n in nodes:
        if isinstance(n, Leaf):
            yield n.span
        else:
            if n.head is not None:
                yield n.head
            for hdr, kids in n.segments:
                if hdr is not None:
                    yield hdr
                yield from _spans(kids)
            if n.tail is not None:
                yield n.tail


def emit(
    lines: list[str], r: Routine, keep, essential, rewrite=None, scaffold=None, untouched=None, kept_nodes=None
) -> list[tuple[int, str]]:
    """
    The routine's lines (original line number, text) with the leaves `keep`
    accepts. A loop/if/associate stays if it holds a kept `essential` leaf; the
    `scaffold` leaves (assignments to the routine's own locals: loop indices)
    stay wherever their block stays, and at the top level. A directive stays with the node it
    annotates (the next one; an `end` directive, the one before). A comment or
    blank line stays if the next code line does. `rewrite` = {id(span): text,
    or None to drop it} for the spans whose text changes. A block `untouched`
    accepts stays whole (in a concrete routine: one nothing moved out of).
    `kept_nodes`, a list, receives the leaves and blocks kept.
    """
    rewrite = rewrite or {}
    scaffold = scaffold or (lambda l: False)
    untouched = untouched or (lambda b: False)
    retained: set[int] = set()

    def walk(nodes) -> None:
        for n in nodes:
            if isinstance(n, Leaf):
                if not n.is_directive and (keep(n) or scaffold(n)):
                    retained.add(id(n.span))
            elif untouched(n) or any(keep(l) and essential(l) for l in n.leaves()):
                retained.update({id(n.head), id(n.tail)})
                for hdr, kids in n.segments:
                    if hdr is not None:
                        retained.add(id(hdr))
                    walk(kids)
        for i, n in enumerate(nodes):
            if isinstance(n, Leaf) and n.is_directive:
                nb = nodes[i - 1] if n.closes and i > 0 else nodes[i + 1] if i + 1 < len(nodes) else None
                if nb is not None and id(_first_span(nb)) in retained:
                    retained.add(id(n.span))

    walk(r.body)
    if kept_nodes is not None:
        def gather(nodes):
            for n in nodes:
                if id(_first_span(n)) in retained:
                    kept_nodes.append(n)
                    if isinstance(n, Block):
                        for _, kids in n.segments:
                            gather(kids)
        gather(r.body)
    header = [sp for sp, _ in r.header]
    retained.update(id(sp) for sp in header + [r.tail])
    on_line: dict[int, list[Span]] = {}
    for sp in header + list(_spans(r.body)) + [r.tail]:
        for ln in range(sp.first, sp.last + 1):
            on_line.setdefault(ln, []).append(sp)
    code_lines = sorted(on_line)
    last_header = r.last_header_line
    out: list[tuple[int, str]] = []

    def kept_line(ln: int) -> bool:
        return any(id(sp) in retained and rewrite.get(id(sp), "") is not None for sp in on_line[ln])

    for ln in range(r.start, r.end + 1):
        here = on_line.get(ln, [])
        if not here:
            nxt = next((c for c in code_lines if c > ln), None)
            if ln <= last_header or (nxt is not None and kept_line(nxt)):
                out.append((ln, lines[ln]))
            continue
        kept = [sp for sp in here if id(sp) in retained]
        indent = lines[ln][: len(lines[ln]) - len(lines[ln].lstrip())]
        if any(id(sp) in rewrite for sp in kept):
            sp = kept[0]
            if len(kept) > 1 or len(here) > 1:
                raise ExtractError(f"line {ln + 1}: a rewritten statement shares its line")
            if ln == sp.first and rewrite[id(sp)] is not None:
                out.append((ln, indent + rewrite[id(sp)]))
            continue
        if not kept:
            continue
        if len(kept) < len(here):
            if not all(sp.first == sp.last == ln for sp in here):
                raise ExtractError(f"line {ln + 1} mixes kept and removed statements")
            out.append((ln, indent + " ; ".join(sp.text for sp in kept)))
        else:
            out.append((ln, lines[ln]))
    return out


@dataclass
class FileEdits:
    replace: dict[int, tuple[int, list[str]]] = field(default_factory=dict)  # start -> (end, lines)
    insert: dict[int, list[str]] = field(default_factory=dict)  # before line

    def add_replace(self, start: int, end: int, new: list[str]) -> None:
        for s, (e, _) in self.replace.items():
            if not (end < s or start > e):
                raise ExtractError(f"overlapping edits at lines {start + 1}-{end + 1}")
        self.replace[start] = (end, new)

    def apply(self, lines: list[str]) -> list[str]:
        out, i = [], 0
        while i < len(lines):
            out.extend(self.insert.get(i, []))
            if i in self.replace:
                end, new = self.replace[i]
                out.extend(new)
                i = end + 1
                continue
            out.append(lines[i])
            i += 1
        return out


@dataclass
class MergeResult:
    family: str
    binding: str
    base_routine: str
    moved: dict[str, int]  # element -> statements moved
    fates: int = 0  # moved statements carbon has under `.not. use_fates`
    blocked: int = 0  # shared statements kept for order or isotope guards


class Merger:
    def __init__(self, families: list[Family], fut, bindings=DEFAULT_BINDINGS):
        self.fut = fut
        self.families = [f for f in families if f.name in fut.type_dict]
        self.bindings = [b.lower() for b in bindings]
        self.edits: dict[Path, FileEdits] = {}
        self.results: list[MergeResult] = []
        self.refused: list[str] = []
        self._lines: dict[Path, list[str]] = {}
        # dummy names of each binding over all families (`build`); empty for a lone `merge`
        self.names_used: dict[str, Counter] = {}

    def lines(self, path: Path) -> list[str]:
        path = Path(path)
        if path not in self._lines:
            self._lines[path] = path.read_text(errors="replace").split("\n")
        return self._lines[path]

    def fe(self, path: Path) -> FileEdits:
        return self.edits.setdefault(Path(path), FileEdits())

    def build(self) -> None:
        # dummy names of each binding in all families: breaks ties in a family
        td, sd = self.fut.type_dict, self.fut.subroutine_dict
        self.names_used = {}
        for fam in self.families:
            for binding in self.bindings:
                for tname in fam.types.values():
                    sid = td[tname].procedures.get(binding)
                    if isinstance(sid, str) and sid in sd:
                        dummies = tuple(d.lower() for d in read_routine(sd[sid]).dummies)
                        self.names_used.setdefault(binding, Counter())[dummies] += 1
        for fam in self.families:
            for binding in self.bindings:
                try:
                    res = self.merge(fam, binding)
                except ExtractError as err:
                    self.refused.append(f"{fam.name} {binding}: {err}")
                    continue
                if res:
                    self.results.append(res)

    def merge(self, fam: Family, binding: str) -> MergeResult | None:
        td, sd = self.fut.type_dict, self.fut.subroutine_dict
        if any(CORE not in td[t].components for t in fam.types.values()):
            raise ExtractError(f"run `spel transform nutrient-extract` first (no {CORE} component)")
        rep = compare_procedure(fam, binding, sd, td)
        if rep is None:
            return None
        base_fields = {f"{CORE}%{c}" for c in td[fam.name].components}
        routines = rep.trees
        for e, r in routines.items():
            if r.unsupported:
                raise ExtractError(f"{rep.routines[e]}: {'; '.join(r.unsupported)}")
        paths = {r.path for r in routines.values()}
        if len(paths) != 1:
            raise ExtractError("routines in several files")
        path = paths.pop()
        lines = self.lines(path)

        def generic(leaf: Leaf) -> bool:
            return leaf.kind == "generic" and bool(leaf.fields) and leaf.fields <= base_fields

        # carbon statements under `if (.not. use_fates)` that N/P have unguarded:
        # in the base routine under `.not. (is_carbon(species) .and. use_fates)`
        fates_keys: set[str] = set()
        if "c" in routines and len(routines) > 1:
            others = {e: {l.key for l in r.leaves()} for e, r in routines.items() if e != "c"}
            for l in routines["c"].leaves():
                k, had = drop_fates_guard(l.key)
                if had and l.kind == "guards" and l.fields and l.fields <= base_fields:
                    if all(k in ks for ks in others.values()):
                        fates_keys.add(k)

        # associate names of each routine, by the leaves they're used in
        alias_of = {}
        for r in routines.values():
            names = associations(r)
            for l in r.leaves():
                alias_of[id(l)] = names

        def writes_fields_only(leaf: Leaf) -> bool:
            """It writes `this%...` (directly or through an associate name) and nothing else."""
            paths = written_paths(leaf.node)
            if not paths:
                return False
            aliases = alias_of.get(id(leaf), {})
            return all((aliases.get(p[0]) or p)[0] == "this" for p in paths)

        def text_key(leaf: Leaf) -> str | None:
            """The statement's text as every routine has it (carbon's FATES guard dropped)."""
            if leaf.is_directive or not writes_fields_only(leaf):
                return None  # directives go with what they annotate
            if generic(leaf):
                return leaf.key
            if leaf.kind == "guards" and drop_fates_guard(leaf.key)[0] in fates_keys:
                return drop_fates_guard(leaf.key)[0]
            return None

        # statements correspond by text and occurrence: Init's `x(:) = spval` after the
        # allocation is the 1st, the history section's `x(begp:endp) = spval` (the same
        # text once normalized) the 2nd, and only an isotope-guarded copy blocks its own
        occurrence: dict[int, str] = {}
        counts: dict[str, Counter] = {}
        for e, r in routines.items():
            n = counts.setdefault(e, Counter())
            for l in r.leaves():
                t = text_key(l)
                if t is not None:
                    occurrence[id(l)] = f"{t} #{n[t]}"
                    n[t] += 1
        def mkey(leaf: Leaf) -> str | None:
            """The key a moving statement is known by in every routine that has its text, else None."""
            k = occurrence.get(id(leaf))
            if k is None:
                return None
            t, _, i = k.rpartition(" #")
            return k if all(n[t] > int(i) for n in counts.values() if n[t]) else None

        def scaffold_of(r: Routine):
            """Statements that may be copied into both routines: assignments to r's own locals."""
            local = r.local_names

            def scaffold(leaf: Leaf) -> bool:
                if leaf.kind != "other":
                    return False
                paths = written_paths(leaf.node)
                return bool(paths) and all(p[0] in local and len(p) == 1 for p in paths)

            return scaffold

        # a moved statement may only read locals that copied scaffolding sets (loop indices,
        # plain local assignments): one set by a kept statement or passed to a call isn't set
        # yet when the base routine runs
        def unsafe_locals(r: Routine) -> set[str]:
            scaffold, local = scaffold_of(r), r.local_names
            out = set()
            for l in r.leaves():
                if l.is_call:
                    out |= statement_names(l.node) & local
                elif not scaffold(l):
                    out |= {p[0] for p in (written_paths(l.node) or []) if p[0] in local}
            return out

        # a key moves in every routine or in none: blocked if a copy is isotope-guarded,
        # reads a local the base routine wouldn't have set, or follows a kept statement on
        # its fields or a kept call
        blocked: set[str] = {mkey(l) for r in routines.values() for l in r.leaves() if l.iso and mkey(l)}
        for r in routines.values():
            unsafe = unsafe_locals(r)
            blocked |= {mkey(l) for l in r.leaves() if mkey(l) and statement_names(l.node) & unsafe}
        while True:
            before = len(blocked)
            for r in routines.values():
                seen: set[str] = set()
                barrier = False
                for l in r.leaves():
                    k = mkey(l)
                    if k is not None and k not in blocked:
                        if barrier or l.fields & seen:
                            blocked.add(k)
                    else:
                        seen |= {f for f in l.fields if not f.startswith("?")}
                        barrier |= l.is_call
            if len(blocked) == before:
                break

        def moves(leaf: Leaf) -> bool:
            k = mkey(leaf)
            return k is not None and k not in blocked

        # the base routine is built from the carbon one when it holds the FATES guard
        if fates_keys and any(moves(l) and not generic(l) for l in routines["c"].leaves()):
            ref_e = "c"
        else:
            common = Counter(tuple(d.lower() for d in r.dummies) for r in routines.values())
            overall = self.names_used.get(binding, Counter())
            best = max(common, key=lambda k: (common[k], overall[k]))
            ref_e = next(e for e in ORDER if e in routines and tuple(d.lower() for d in routines[e].dummies) == best)
        ref = routines[ref_e]
        if not any(moves(l) for l in ref.leaves()):
            return None
        base_name = f"{fam.name}_{binding}"
        if any(k.split("::")[-1] == base_name for k in sd):
            raise ExtractError(f"{base_name} exists already (merged before)")

        self._scaffold_of = scaffold_of
        out = self.base_routine(fam, binding, ref, ref_e, routines, moves, fates_keys, lines)
        first = min(r.start for r in routines.values())
        self.fe(path).insert.setdefault(first, []).extend(out + [""])

        # concrete routines: their own statements, after a call of the base routine
        dummy_names = self._dummy_names
        moved = {}
        for e, r in routines.items():
            scaffold = scaffold_of(r)
            keep = lambda l: not moves(l)
            essential = lambda l: not moves(l) and not scaffold(l)
            moved[e] = sum(moves(l) for l in r.leaves())
            untouched = lambda b: not any(moves(l) for l in b.leaves())
            pairs = emit(lines, r, keep=keep, essential=essential, scaffold=scaffold, untouched=untouched)
            args = ", ".join([f"this%{CORE}"] + dummy_names)
            at_line = self.call_line(r, keep)
            at = next((i for i, (ln, _) in enumerate(pairs) if ln >= at_line), len(pairs) - 1)
            ref_line = pairs[at][1] if at < len(pairs) - 1 else "    "
            indent = ref_line[: len(ref_line) - len(ref_line.lstrip())] or "    "
            new = [t for _, t in pairs]
            new.insert(at, f"{indent}call {base_name}({args})")
            self.fe(path).add_replace(r.start, r.end, new)
        return MergeResult(
            fam.name, binding, base_name, moved,
            fates=sum(1 for l in ref.leaves() if moves(l) and not generic(l)),
            blocked=len(blocked),
        )

    def base_routine(self, fam, binding, ref: Routine, ref_e: str, routines, moves, fates_keys, lines) -> list[str]:
        base_name = f"{fam.name}_{binding}"
        rewrite: dict[int, str | None] = {}
        for blk in ref.blocks():
            if fates_keys and is_fates_if(blk.node):
                if blk.head.first != blk.head.last:
                    raise ExtractError(f"line {blk.head.first + 1}: the use_fates if spans lines")
                rewrite[id(blk.head)] = FATES_IF_BASE
        scaffold = self._scaffold_of(ref)
        kept: list = []
        emit(lines, ref, keep=moves, essential=moves, rewrite=rewrite, scaffold=scaffold, kept_nodes=kept)
        # what the kept statements refer to (parse tree)
        names = set()
        for n in kept:
            if isinstance(n, Leaf):
                names |= statement_names(n.node)
        # associate constructs keep the names those statements use, pointed at the core
        for n in kept:
            if isinstance(n, Block) and isinstance(n.node, AssociateConstruct):
                text = self.associate_head(n, names, ref.dummies[0].lower())
                rewrite[id(n.head)] = text
                if text is None:
                    rewrite[id(n.tail)] = None
                else:
                    names |= {x for a, e in n.node.associations.items() if str(a).lower() in names for x in names_in(e)}
            elif isinstance(n, Block):
                names |= statement_names(n.node)
        pairs = emit(lines, ref, keep=moves, essential=moves, rewrite=rewrite, scaffold=scaffold)
        positions, decls, uses = ref.requirements(names)
        self._dummy_names = [ref.dummies[i] for i in positions]
        for r in routines.values():
            missing = [d for d in self._dummy_names if d.lower() not in {x.lower() for x in r.dummies}]
            if missing:
                raise ExtractError(f"{r.name} has no dummy {missing}: dummies correspond by name")
        dummies = ", ".join([ref.dummies[0]] + self._dummy_names)
        body = [_CORE_RE.sub(r"this\1%\2", t) for ln, t in pairs if ref.last_header_line < ln < ref.tail.first]
        device = [
            lines[sp.first] for r in routines.values() for sp, n in r.header
            if is_spec_directive(n) and n.construct == "routine"
        ][:1]
        return [
            f"  subroutine {base_name}({dummies})",
            *device,
            *use_lines(uses),
            "    !",
            "    ! !ARGUMENTS:",
            f"    type({fam.name}), intent(inout) :: this",
            *declaration_lines(lines, decls),
            "    !-----------------------------------------------------------------------",
            *body,
            f"  end subroutine {base_name}",
        ]

    @staticmethod
    def associate_head(blk: Block, used: set[str], this: str) -> str | None:
        """The associate statement with only the used names, `this%core%x` as `this%x`; None if none is used."""
        items = []
        for name, sel in blk.node.associations.items():
            if str(name).lower() not in used:
                continue
            path = designator_path(sel)
            if path and path[0] == this:
                if len(path) != 3 or path[1] != CORE or not isinstance(sel, Identifier):
                    raise ExtractError(f"line {blk.head.first + 1}: {name} names {'%'.join(path)}, not a core field")
                items.append(f"{name} => {this}%{path[2]}")
            else:
                items.append(f"{name} => {sel}")
        if not items:
            return None
        return "associate( &\n" + ", &\n".join(f"         {i}" for i in items) + " )"

    @staticmethod
    def call_line(r: Routine, keep) -> int:
        """
        The line before which the base call goes: the first kept statement after the
        species assignment (or the directive annotating it).
        """
        for i, n in enumerate(r.body):
            if isinstance(n, Leaf):
                if n.is_directive or not keep(n) or assigned_path(n.node) == ["this", CORE, "species"]:
                    continue
            elif not any(keep(l) for l in n.leaves()):
                continue
            prev = r.body[i - 1] if i > 0 else None
            if isinstance(prev, Leaf) and prev.is_directive and not prev.closes:
                return prev.span.first
            return _first_span(n).first
        return r.tail.first

    def apply(self) -> dict[Path, list[str]]:
        return {p: fe.apply(self.lines(p)) for p, fe in self.edits.items()}

    def report(self) -> str:
        out = []
        for r in self.results:
            moved = ", ".join(f"{e}:{n}" for e, n in r.moved.items())
            notes = f"; {r.fates} of carbon's under .not. use_fates" if r.fates else ""
            notes += f"; {r.blocked} kept for order or isotope guards" if r.blocked else ""
            out.append(f"{r.family} {r.binding}: {r.base_routine} ({moved} statements moved{notes})")
        for msg in self.refused:
            out.append(f"not merged: {msg}")
        return "\n".join(out)
