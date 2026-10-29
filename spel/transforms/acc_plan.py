"""
Propose an OpenACC loop structure for the loops of a routine, from the analysis.

For every outermost loop that no compute directive covers yet, the record's
accesses decide the shape (`spel accplan MODULE`):

- elementwise:  every array written in the nest is indexed by every loop index
  (directly or through an injective alias such as `c = filter(fc)`): the perfect
  nest can be collapsed.
- column-parallel: the outer loop (e.g. over columns) carries no dependence; the
  inner loops (levels, species) carry one or write per-column values: parallel
  over the outer loop, `loop seq` inside. Bit-for-bit with the serial code since
  each element keeps its summation order.
- interchange: a (level, column) nest whose column arrays accumulate over levels
  (`tot(c) = tot(c) + x(c,j)` under `do j` / `do fc`): swap to column-outer and
  run the level loop `seq`. Same arithmetic order per column. More generally a
  rectangular perfect nest whose carried loops sit between parallel ones (`do j /
  do pi / do fc` summing patches into `x(c,j)`) is reordered: the loops that carry
  nothing outermost and collapsed (`order`, `npar`), the carried ones inside in
  their original order. Parallelizing only the outer level loop would run every
  column in one thread.
- scatter: a loop over a finer level writes arrays of a coarser one through a
  many-to-one alias (`c = veg_pp%column(p)` in a patch loop): parallelize over
  the coarser level with an inner loop over its members in filter order instead.
- carried: a dependence on the outer index that none of the above resolves
  (needs a reduction, atomics, or a sequential loop).

The private/firstprivate clauses come from the OpenACC-to-OpenMP planner
(`plan_compute_loop`) applied to a synthetic `!$acc parallel loop`, so they follow
the same rules as `spel acc2omp`. Statements a device cannot run (`write`,
`print`, `call endrun`) are listed per loop: replace them by a flag set in the
loop and tested on the host after it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from spel.scripts.fortran_parser.scope_walk import AssignEvent, LoopInfo, SubroutineRecord
from spel.transforms.acc_directives import find_directives, parse_directive
from spel.transforms.acc_to_omp import LoopPlan, Note, format_directive, plan_compute_loop

# alias targets that map many fine elements to one coarse one
COARSER = {"column", "gridcell", "landunit", "topounit"}   # x%column(p): many fine elements -> one coarse


# first member of a coarser element -> its member count (`p = col_pp%pfti(c) + pi - 1`)
FIRST_MEMBER = {"pfti": "npfts", "coli": "ncolumns", "luni": "nlandunits", "topi": "ntopounits"}


@dataclass
class Alias:
    index: str          # the loop index this scalar derives from
    injective: bool     # one element per index value (filters, index arithmetic)
    via: str            # what it was read through
    also: tuple = ()    # further loop indices it covers injectively, jointly with `index`


# the device data layer's flag: a kernel runs on the host while it is false (initialization, host
# segments of elm_drv in targeted mode); every compute construct of the port carries it
DEVICE_FLAG = "elm_device_ready"


@dataclass
class LoopProposal:
    routine: str
    file: str
    loop: LoopInfo
    nest: list[LoopInfo]
    kind: str                                   # elementwise | column-parallel | interchange | scatter | carried | has-directive
    reason: str = ""
    carried: dict[str, list[str]] = field(default_factory=dict)   # index -> arrays carried over it
    scatter: dict[str, list[str]] = field(default_factory=dict)   # index -> arrays scattered through it
    same_value: dict[str, list[str]] = field(default_factory=dict)  # index -> arrays written per element but not per index value
    plan: Optional[LoopPlan] = None
    notes: list[Note] = field(default_factory=list)
    unsafe: list[tuple[int, str]] = field(default_factory=list)   # (1-based line, statement)
    private_arrays: list[str] = field(default_factory=list)       # local arrays written and read per iteration
    copyin: list[str] = field(default_factory=list)               # `bounds`: a dummy the loop reads (copyin of present data is a no-op)
    children: list["LoopProposal"] = field(default_factory=list)  # proposals for the inner loops of a host loop
    order: list[str] = field(default_factory=list)  # interchange: the nest's indices in their new order
    npar: int = 1  # interchange: how many of `order` are parallel (collapsed)

    @property
    def lines(self) -> tuple[int, int]:
        return self.loop.start_ln + 1, self.loop.end_ln + 1

    @property
    def indices(self) -> str:
        return ">".join(l.index or "?" for l in self.loop.walk() if l.index) if self.loop.children else (self.loop.index or "?")

    def directive_lines(self) -> list[str]:
        """The proposed directive(s), as source lines (indent of the loop)."""
        if self.plan is None or self.kind in ("has-directive", "scatter", "carried"):
            return []
        indent = self.plan.directive.indent
        clauses = ["gang", "vector"]
        if self.kind == "elementwise" and len(self.nest) > 1:
            clauses.append(f"collapse({len(self.nest)})")
        elif self.kind == "interchange" and self.npar > 1:
            clauses.append(f"collapse({self.npar})")
        clauses.append("default(present)")
        clauses.extend(c for c in self.plan.clauses() if not c.startswith("collapse"))
        if self.private_arrays:
            clauses.append(f"private({', '.join(self.private_arrays)})")
        if self.copyin:
            clauses.append(f"copyin({', '.join(self.copyin)})")
        clauses.append(f"if({DEVICE_FLAG})")  # runs on the host until the device data layer is up (initialization)
        out = [l.replace("!$omp", "!$acc") for l in format_directive(indent, "parallel loop", clauses)]
        if self.kind in ("column-parallel", "interchange"):
            out.append(f"{indent}!$acc loop seq   ! on each inner loop")
        return out

    def __str__(self) -> str:
        a, b = self.lines
        head = f"{self.file}:{a}-{b} do {self.indices}: {self.kind.upper()}"
        if self.reason:
            head += f" {self.reason}"
        parts = [head]
        for text in self.directive_lines():
            parts.append("    " + text.strip())
        for idx, arrs in self.carried.items():
            parts.append(f"    carried over {idx}: {', '.join(arrs)}")
        for idx, arrs in self.scatter.items():
            parts.append(f"    scatter via {idx}: {', '.join(arrs)}")
        for idx, arrs in self.same_value.items():
            parts.append(f"    same-value writes over {idx}: {', '.join(arrs)}")
        for ln, stmt in self.unsafe:
            parts.append(f"    device-unsafe {ln}: {stmt}")
        for n in self.notes:
            if n.level != "info":
                parts.append(f"    {n.level}: {n.msg}")
        for ch in self.children:
            parts.append("  | " + str(ch).replace("\n", "\n  | "))
        return "\n".join(parts)


def _terms(e, sign: int = 1):
    """(sign, node) of the terms of a +/- expression."""
    from spel.scripts.fortran_parser import spel_ast as A

    if isinstance(e, A.InfixExpression) and e.operator.strip() in ("+", "-"):
        yield from _terms(e.left_expr, sign)
        yield from _terms(e.right_expr, sign if e.operator.strip() == "+" else -sign)
    else:
        yield sign, e


def _member_offset(rhs, guards, indices: set[str], aliases: dict[str, "Alias"]) -> Optional[tuple[str, str]]:
    """
    `x%pfti(c) + pi - 1` under a guard `pi <= x%npfts(c)` (and the coli/luni/topi forms):
    the members of distinct coarse elements are disjoint ranges, so the result is one
    element per (c, pi). Returns (c, pi), or None.
    """
    from spel.scripts.fortran_parser import spel_ast as A

    terms = list(_terms(rhs))
    maps = [t for sg, t in terms if sg > 0 and isinstance(t, A.FuncExpression) and len(t.args) == 1
            and str(t.function).lower().split("%")[-1] in FIRST_MEMBER]
    idx = [t.value.lower() for sg, t in terms if sg > 0 and isinstance(t, A.Identifier)
           and t.value.lower() in indices]
    rest = [t for sg, t in terms if not (isinstance(t, A.FuncExpression) and any(t is m for m in maps))
            and not (isinstance(t, A.Identifier) and t.value.lower() in idx)]
    if len(maps) != 1 or len(idx) != 1 or not all(isinstance(t, A.IntegerLiteral) for t in rest):
        return None
    m, pi = maps[0], idx[0]
    coarse = str(m.args[0]).lower()
    if coarse not in aliases and coarse not in indices:
        return None
    holder, first = str(m.function).lower().rsplit("%", 1) if "%" in str(m.function) else ("", str(m.function).lower())
    count = f"{holder}%{FIRST_MEMBER[first]}" if holder else FIRST_MEMBER[first]
    for g in guards:
        n = g.node
        if g.negated or not isinstance(n, A.InfixExpression) or n.operator.strip() != "<=":
            continue
        left, right = n.left_expr, n.right_expr
        if isinstance(left, A.Identifier) and left.value.lower() == pi and isinstance(right, A.FuncExpression) \
                and str(right.function).lower() == count and len(right.args) == 1 \
                and str(right.args[0]).lower() == coarse:
            return coarse, pi
    return None


def _parallel_first(nest, carried, scatter, same_value, aliases=None) -> Optional[tuple[list[str], int]]:
    """
    Order of a perfect nest with carried loops: the indices no dependence is carried over (and
    that scatter or rewrite nothing) first and collapsed, then the carried ones in their
    original order. A loop whose bounds use another nest index (directly or through an alias:
    `do p = pfti(c), pftf(c)` with `c = filter(fc)`) must stay inside it, and the collapsed
    loops must not depend on each other. None when fewer than two loops would be parallel and
    the order does not change, or when the order is not valid.
    """
    aliases = aliases or {}
    idx = [lp.index for lp in nest]
    if len(nest) < 2 or any(lp.index is None for lp in nest):
        return None
    par = [i for i in idx if i not in carried and i not in scatter and i not in same_value]
    seq = [i for i in idx if i not in par]
    if not par or not seq or any(i in scatter or i in same_value for i in seq):
        return None
    if par + seq == idx and len(par) < 2:
        return None  # the outer loop alone is parallel: column-parallel already
    def deps(lp) -> set[str]:
        names = {b.split("%")[0].lower() for b in lp.bound_paths}
        return {n for n in names if n in idx} | {aliases[n].index for n in names if n in aliases} \
            | {x for n in names if n in aliases for x in aliases[n].also}
    order = par + seq
    for lp in nest:
        pos = order.index(lp.index)
        if any(order.index(d) >= pos for d in deps(lp) if d in idx and d != lp.index):
            return None  # its bounds would use an index set inside it
        if pos < len(par) and deps(lp) & set(par) - {lp.index}:
            return None  # collapse needs rectangular parallel loops
    return order, len(par)


def _aliases(rec: SubroutineRecord, lo: int, hi: int, indices: set[str]) -> dict[str, Alias]:
    """Integer scalars assigned in lines lo..hi from a loop index (through filters, subgrid maps, arithmetic)."""
    out: dict[str, Alias] = {}
    try:
        from spel.scripts.registrations import line_guards

        guards = line_guards(rec.tree.body)
    except Exception:  # no parse tree: no guarded forms
        guards = {}
    changed = True
    while changed:
        changed = False
        for ev in rec.events:
            if not isinstance(ev, AssignEvent) or not (lo <= ev.ln <= hi):
                continue
            lhs = ev.lhs
            if lhs.subscripts or "%" in lhs.path or lhs.path in indices or lhs.path in out:
                continue
            decl = rec.declaration(lhs.path)
            if decl is None or decl.dim or "integer" not in decl.type.lower():
                continue
            rhs_text = str(ev.rhs).replace(" ", "").lower()  # a label for reports only
            # c = fc + 1, c = fc: injective arithmetic on an index/alias
            names = _names(ev.rhs)
            direct = [n for n in names if n in indices]
            via_alias = [n for n in names if n in out]
            reads = [r for r in ev.reads if r.subscripts]
            member = _member_offset(ev.rhs, guards.get(ev.ln, ()), indices, out)
            if member is not None:
                coarse, pi = member
                base_idx, inj = (coarse, True) if coarse in indices else (out[coarse].index, out[coarse].injective)
                out[lhs.path] = Alias(base_idx, inj, rhs_text, also=(pi,))
                changed = True
            elif reads:
                r = reads[0]
                subs_names = _names(r.subscripts)
                src = [n for n in subs_names if n in indices] + [n for n in subs_names if n in out]
                if not src:
                    continue
                base_idx, inj = (src[0], True) if src[0] in indices else (out[src[0]].index, out[src[0]].injective)
                injective = inj and ("filter" in r.path.lower() or r.path.lower() in ("ipft", "icol", "ilun", "igrc"))
                if "%" in r.path and r.path.lower().split("%")[-1] in COARSER:
                    injective = False
                out[lhs.path] = Alias(base_idx, injective, r.path)
                changed = True
            elif direct or via_alias:
                src = direct[0] if direct else via_alias[0]
                base_idx, inj = (src, True) if src in indices else (out[src].index, out[src].injective)
                out[lhs.path] = Alias(base_idx, inj, rhs_text)
                changed = True
    return out


def _names(node, out: Optional[set[str]] = None) -> set[str]:
    """Identifiers of an expression (an AST walk; function and array names included)."""
    from spel.scripts.fortran_parser import spel_ast as A

    out = set() if out is None else out
    if isinstance(node, A.Identifier):
        out.add(node.value.lower())
    elif isinstance(node, (list, tuple)):
        for x in node:
            _names(x, out)
    elif hasattr(node, "__dict__") and not isinstance(node, (str, int, float)):
        for k, v in vars(node).items():
            if k != "token":
                _names(v, out)
    return out


def _cover_expr(e, aliases: dict[str, Alias], indices: set[str]) -> dict[str, bool]:
    """
    index -> injective? for one subscript expression: an index (or an alias of one) on its
    own or in +, -, * arithmetic maps one value to one element; an index that only
    reaches the subscript through an array lookup or a function (`pool(k)`) does not
    (several k may name one element), except through a filter (injective by construction).
    """
    from spel.scripts.fortran_parser import spel_ast as A

    def merge(*ds):
        out: dict[str, bool] = {}
        for d in ds:
            for k, v in d.items():
                out[k] = out.get(k, True) and v if k in out else v
        return out

    match e:
        case A.Identifier():
            n = e.value.lower()
            if n in indices:
                return {n: True}
            if n in aliases:
                a = aliases[n]
                return {a.index: a.injective, **{x: a.injective for x in a.also}}
            return {}
        case A.InfixExpression():
            d = merge(_cover_expr(e.left_expr, aliases, indices), _cover_expr(e.right_expr, aliases, indices))
            if e.operator.strip() not in ("+", "-", "*"):
                d = {k: False for k in d}  # j/2, mod(j,n), ...: several values per element
            return d
        case A.PrefixExpression():
            return _cover_expr(e.right_expr, aliases, indices)
        case A.FuncExpression():
            inner = merge(*[_cover_expr(x, aliases, indices) for x in e.args])
            fname = str(e.function).lower()
            return inner if "filter" in fname else {k: False for k in inner}
        case _:
            found = {}
            for n in _names(e):
                if n in indices:
                    found[n] = False
                elif n in aliases:
                    found[aliases[n].index] = False
            return found


def _covers(subs: tuple, aliases: dict[str, Alias], indices: set[str]) -> dict[str, bool]:
    """index -> injective? for the loop indices a subscript tuple (expression nodes) reaches."""
    found: dict[str, bool] = {}
    for sub in subs:
        for k, v in _cover_expr(sub, aliases, indices).items():
            found[k] = found.get(k, False) or v
    return found


def unsafe_lines(sub, source: Optional[list[str]] = None) -> list[tuple[int, str]]:
    """
    (0-based line, statement) of what a device cannot run: I/O statements (parse tree) and
    abort calls (the record's calls).
    """
    from spel.scripts.config import E3SM_SRCROOT
    from spel.scripts.fortran_parser import spel_ast as A
    from spel.scripts.registrations import iter_statements, parse_routine

    out: list[tuple[int, str]] = []
    try:
        if getattr(sub, "filepath", None):
            path = Path(sub.filepath)
            path = path if path.is_absolute() else Path(E3SM_SRCROOT) / path
            tree = parse_routine(path, sub.startline, sub.endline)
        else:  # a routine given as source lines (tests)
            from spel.scripts.fortran_parser.spel_parser import Parser
            from spel.scripts.utilityFunctions import LineTuple

            a, b = getattr(sub, "startline", 0), getattr(sub, "endline", len(source or []) - 1)
            lt = [LineTuple(line=source[i], ln=i) for i in range(a, b + 1)]
            tree = Parser(lines=lt).parse_program().statements[0]
        out += [(st.lineno, type(st).__name__) for st, _ in iter_statements(tree.body)
                if isinstance(st, (A.WriteStatement, A.PrintStatement))]
    except BaseException:  # the parser exits on what it cannot read: every line is suspect
        out += [(ln, "unparsed") for ln in range(getattr(sub, "startline", 0), getattr(sub, "endline", -1) + 1)]
    if sub.record is not None:
        out += [(c.ln, f"call {c.name}") for c in sub.record.calls
                if str(c.name).split("%")[-1].lower() in ABORTS]
    return sorted(set(out))


ABORTS = {"endrun", "shr_sys_abort", "abort", "mpi_abort"}


def _is_array_path(rec: SubroutineRecord, path: str, accesses) -> bool:
    if any(a.subscripts for a in accesses):
        return True
    decl = rec.declaration(path.split("%")[0])
    return bool(decl and decl.dim and "%" not in path)


def _covered_by_directive(loop: LoopInfo, directives, lines: list[str]) -> bool:
    for d in directives:
        if d.is_compute_loop and d.end_ln < loop.start_ln:
            between = lines[d.end_ln + 1 : loop.start_ln]
            if all(not l.strip() or l.strip().startswith("!") for l in between):
                return True
        if d.is_compute_region and d.start_ln < loop.start_ln:
            # region ends at the matching `end` directive
            for e in directives:
                if e.construct.startswith("end ") and e.start_ln > d.start_ln and e.start_ln > loop.end_ln:
                    return True
                if e.construct.startswith("end ") and e.start_ln > d.start_ln:
                    break
    return False


def _is_local_array(rec: SubroutineRecord, path: str) -> bool:
    if "%" in path:
        return False
    decl = rec.declaration(path)
    return bool(decl and decl.dim and decl.intent is None and all(d.name != path for d in rec.dummy_args))


_UNSAFE_CACHE: dict[str, list[tuple[int, str]]] = {}


def _unsafe(sub, source: Optional[list[str]] = None) -> list[tuple[int, str]]:
    key = getattr(sub, "id", None) or sub.name
    if key not in _UNSAFE_CACHE:
        _UNSAFE_CACHE[key] = unsafe_lines(sub, source)
    return _UNSAFE_CACHE[key]


def propose(sub, mapper, source: Optional[list[str]] = None, min_lines: int = 2) -> list[LoopProposal]:
    rec: SubroutineRecord = sub.record
    nest_roots = getattr(rec, "loop_nest", None)
    if not nest_roots:
        return []
    if source is None:
        source = Path(sub.filepath).read_text().splitlines()
    directives = find_directives(source)
    fname = Path(sub.filepath).name if getattr(sub, "filepath", None) else "<src>"
    return [p for loop in nest_roots for p in _propose_loop(sub, rec, mapper, loop, source, directives, fname, min_lines)]


def _propose_loop(sub, rec, mapper, loop, source, directives, fname, min_lines) -> list[LoopProposal]:
    atomics = []
    out: list[LoopProposal] = []
    for loop in [loop]:
        if loop.end_ln - loop.start_ln < min_lines:
            continue
        if _covered_by_directive(loop, directives, source):
            out.append(LoopProposal(sub.name, fname, loop, [loop], "has-directive"))
            continue
        # deepest perfect nest
        nest = [loop]
        while nest[-1].tight_child is not None and nest[-1].tight_child.index:
            nest.append(nest[-1].tight_child)
        if any(_covered_by_directive(l, directives, source) for l in nest[1:]):
            # an inner loop of the nest already carries a directive (an earlier pass ported it
            # while the outer loop was carried): collapsing over it would nest two compute loops
            out.append(LoopProposal(sub.name, fname, loop, [loop], "has-directive"))
            continue
        lo, hi = loop.start_ln, loop.end_ln
        indices = {l.index for l in loop.walk() if l.index}
        nest_idx = [l.index for l in nest]
        aliases = _aliases(rec, lo, hi, indices)
        by_path: dict[str, list] = {}
        for a in rec.accesses():
            if lo <= a.ln <= hi and a.context not in ("do", "size", "inquiry"):
                by_path.setdefault(a.path, []).append(a)
        carried: dict[str, list[str]] = {}
        scatter: dict[str, list[str]] = {}
        same_value: dict[str, list[str]] = {}
        private_arrays: list[str] = []
        for path, accs in by_path.items():
            writes = [a for a in accs if "w" in a.status]
            if not writes or not _is_array_path(rec, path, accs):
                continue
            reads = [a for a in accs if "r" in a.status]
            if _is_local_array(rec, path) and not any(_covers(w.subscripts, aliases, indices) for w in writes):
                private_arrays.append(path)   # a per-iteration work array
                continue
            # another element of a written array (x(j-1) beside x(j) = ..., x(1) with x(j) = ...):
            # a dependence carried over each index the write subscript uses where they differ
            for w in writes:
                for other in accs:
                    if other.subs == w.subs:
                        continue
                    for k, ws in enumerate(w.subscripts):
                        if k < len(other.subs) and other.subs[k] == w.subs[k]:
                            continue
                        for idx in _cover_expr(ws, aliases, indices):
                            if idx in indices and path not in carried.get(idx, []):
                                carried.setdefault(idx, []).append(path)
            for w in writes:
                cov = _covers(w.subscripts, aliases, indices)
                for idx in nest_idx:
                    if idx in cov and not cov[idx]:
                        scatter.setdefault(idx, []).append(path) if path not in scatter.get(idx, []) else None
                    elif idx not in cov:
                        # written without this index: carried if also read in the loop
                        if reads:
                            if path not in carried.get(idx, []):
                                carried.setdefault(idx, []).append(path)
                        elif path not in same_value.get(idx, []):
                            same_value.setdefault(idx, []).append(path)
        # shape
        outer = nest_idx[0]
        inner_carried = {i for i in carried if i != outer}
        # parallel loops first, carried ones inside: preferred over the two-loop swap when it
        # gives more than one parallel loop (`do k / do j / do fc` -> (j, fc) parallel, k seq)
        reorder = _parallel_first(nest, carried, scatter, same_value, aliases)
        if reorder is not None and reorder[1] > 1 and outer not in scatter:
            kind, reason = "interchange", (f"(reorder to `do {' / do '.join(reorder[0])}`: "
                                           f"{', '.join(reorder[0][:reorder[1]])} parallel, "
                                           f"{', '.join(reorder[0][reorder[1]:])} seq)")
        elif outer in scatter:
            kind, reason = "scatter", f"(writes of coarser-level arrays through {', '.join(sorted({aliases[a].via for a in aliases if not aliases[a].injective}))}: parallelize over that level with an inner loop over its members)"
        elif outer in carried:
            if len(nest_idx) >= 2 and nest_idx[1] not in carried and nest_idx[1] not in scatter:
                kind, reason = "interchange", f"(swap to `do {nest_idx[1]}` outer, parallel / `do {outer}` seq)"
                reorder = None
            else:
                kind, reason = "carried", "(dependence over the outer index: reduction, atomic or sequential)"
        elif outer in same_value and len(nest_idx) >= 2 and nest_idx[1] not in carried:
            kind, reason = "interchange", f"(same-value writes over {outer}; swap to `do {nest_idx[1]}` outer / `do {outer}` seq)"
            reorder = None
        elif len(nest_idx) >= 2 and not inner_carried and not any(i in same_value for i in nest_idx) \
                and not any(i in scatter for i in nest_idx[1:]):
            # (a scatter over an inner index rules out the collapse: column-parallel below runs
            # those loops seq, in order, per outer iteration)
            kind, reason = "elementwise", f"(collapse({len(nest_idx)}))"
        elif reorder is not None:
            order, npar = reorder
            kind, reason = "interchange", (f"(reorder to `do {' / do '.join(order)}`: "
                                           f"{', '.join(order[:npar])} parallel, {', '.join(order[npar:])} seq)")
        elif loop.children:
            kind, reason = "column-parallel", "(inner loops seq)"
        else:
            kind, reason = "elementwise", ""
        prop = LoopProposal(sub.name, fname, loop, nest if kind == "elementwise" else [loop], kind, reason,
                            carried, scatter, same_value, private_arrays=private_arrays)
        # a `bounds` dummy read in the loop (`do p = bounds%begp, bounds%endp`): the caller's variable is on
        # the device only when someone entered it (elm_drv's bounds_clump is, other callers' locals are not)
        if "bounds" in {d.name.lower() for d in rec.dummy_args} and any(
                a.base.lower() == "bounds" and lo <= a.ln <= hi for a in rec.accesses()):
            prop.copyin = ["bounds"]
        if kind == "interchange":  # the two-loop swap, or the reorder
            prop.order, prop.npar = reorder if reorder is not None else ([nest_idx[1], nest_idx[0]], 1)
        if kind == "carried" and loop.children:
            prop.reason = "(host loop stays sequential; proposals for its inner loops follow)"
            for child in loop.children:
                prop.children.extend(_propose_loop(sub, rec, mapper, child, source, directives, fname, min_lines))
        # clauses from the OpenMP planner on a synthetic directive
        indent = re.match(r"\s*", source[loop.start_ln]).group(0)
        text = "parallel loop" + (f" collapse({len(prop.nest)})" if len(prop.nest) > 1 else "")
        directive = parse_directive(text, loop.start_ln, loop.start_ln, indent)
        try:
            plan, notes = plan_compute_loop(sub, directive, loop, mapper, atomics)
        except Exception as err:  # analysis gaps shouldn't hide the shape
            plan, notes = None, [Note("warn", sub.name, loop.start_ln, f"planner: {err}")]
        prop.plan, prop.notes = plan, notes
        prop.unsafe = [(ln + 1, what) for ln, what in _unsafe(sub, source) if lo <= ln <= hi]
        out.append(prop)
    return out


def report(subs: Iterable, mapper, min_lines: int = 2) -> list[LoopProposal]:
    found: list[LoopProposal] = []
    for sub in subs:
        found.extend(propose(sub, mapper, min_lines=min_lines))
    return sorted(found, key=lambda p: (p.file, p.lines))
