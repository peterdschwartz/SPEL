"""
Fuse adjacent loop nests that run over the same iteration space (`spel transform loop-fusion`).

    do spel_i1 = bounds%begc, bounds%endc   ! section-loops' x(bounds%begc:bounds%endc) = 0._r8
       x(spel_i1) = 0._r8
    end do
    do spel_i1 = bounds%begc, bounds%endc   ! ... y(bounds%begc:bounds%endc) = 0._r8
       y(spel_i1) = 0._r8
    end do

becomes one nest whose body runs both statements per iteration:

    do spel_i1 = bounds%begc, bounds%endc
       x(spel_i1) = 0._r8
       y(spel_i1) = 0._r8
    end do

so that acc-apply ports one kernel instead of one per statement (section-loops writes a nest per
section assignment; five initializations in a row were five kernel launches). Meant to run in a
pipeline after the passes that write loops and before acc-apply:

    [[step]]
    transform = "section-loops"
    [[step]]
    transform = "loop-fusion"
    [[step]]
    transform = "acc-apply"

What is fused: a run of sibling loop nests with nothing but blank lines and comments between
them, whose headers agree level by level (same index; start, end and step the same expression,
or `lbound`/`ubound` of arrays allocated with the same bounds in that dimension, see below). The
nests are fused to the depth at which their headers agree (each level above it perfectly
nested); the bodies at that depth are concatenated in order.

When it is the same computation (all from the record; anything else is refused and reported):

- no calls in the bodies, no user function references (unknown effects), no exit/cycle/return/
  goto/stop, no allocate/deallocate, no OpenACC/OpenMP directive on or in the nests;
- a variable one body writes and another accesses is either an array whose every access in
  those bodies has the same subscripts, each fused index appearing as one of them (the same
  iteration touches the same element, different iterations different elements), or a scalar
  every body that accesses it writes first, unconditionally (a statement of the body itself,
  not inside an if): a per-iteration temporary whose last value is the last body's, as before;
- no body writes a variable that a later nest's bounds read, or a fused loop index.

`lbound(a, k)` and `lbound(b, k)` (`ubound` alike) count as the same when a and b are derived-type
components or module arrays whose recorded allocations (`allocations.apply_allocations`:
`allocate(this%x(begc:endc, ...))`) have the same text in dimension k. That assumes ELM's
convention that the types are initialized over the same (process) bounds. An associate name
counts only when it is associated with a whole array (not a section, whose lower bound is 1).
Local and dummy arrays need equal explicit bounds in the headers.

A fusion can make inner loops adjacent (the first body ends with a loop, the second starts with
one): run the transform again (or twice in a spec) to fuse those too.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Optional

from spel.scripts.fortran_parser import spel_ast as A
from spel.transforms.core.edits import replace
from spel.transforms.core.registry import transform
from spel.transforms.core.source import code, statement_end
from spel.transforms.core.transform import Change, Plan, Refusal, Transform

TAG = "! spel loop-fusion"
NO_FUSE = (A.ExitStatement, A.ReturnStatement, A.GotoStatement, A.StopStatement)  # CycleStatement is an ExitStatement


def _norm(e) -> str:
    return "" if e is None else str(e).replace(" ", "").lower()


def chain(loop) -> list:
    """The loop and the loops perfectly nested in it, outermost first."""
    out = [loop]
    while (c := out[-1].tight_child) is not None:
        out.append(c)
    return out


def _blank_or_comment(line: str) -> bool:
    s = line.strip()
    return not s or (s.startswith("!") and not s.startswith("!$"))


@dataclass
class Body:
    """The statements of one nest at the fusion depth."""

    nest: list  # LoopInfo chain, outermost first
    first: int  # first line of the body (0-based)
    last: int   # last line of the body
    stmts: list = field(default_factory=list)  # the body's statements (AST)


class Fuser:
    def __init__(self, ctx, sub, lines: list[str], types, directives, tree):
        self.ctx, self.sub, self.lines, self.types = ctx, sub, lines, types
        self.rec = sub.record
        self.directives = directives
        self.tree = tree
        self._assoc = None
        from spel.scripts.fortran_parser.scope_walk import AssignEvent

        self._assigns = [e for e in self.rec.events if isinstance(e, AssignEvent)]
        self.changes: list = []
        self.refusals: list[Refusal] = []

    # --- headers ---------------------------------------------------------------------------

    def _header_end(self, loop) -> int:
        return statement_end(self.lines, loop.start_ln)

    def _associations(self) -> list:
        """(first line, last line, {name: target expression}) of the routine's associate constructs."""
        if self._assoc is None:
            from spel.scripts.registrations import iter_statements

            self._assoc = []
            body = getattr(self.tree, "body", None)
            if body is not None:
                for st, _ in iter_statements(body):
                    if isinstance(st, A.AssociateConstruct):
                        self._assoc.append((st.lineno, st.end_ln, {k.lower(): v for k, v in st.associations.items()}))
        return self._assoc

    def _whole_array(self, name: str, ln: int, depth: int = 0) -> bool:
        """`name` at line `ln` is a variable or an associate name of a whole array (chains followed)."""
        from spel.scripts.fortran_parser.scope_walk import flatten_designator

        for first, last, names in sorted(self._associations(), key=lambda f: -f[0]):  # innermost first
            if first <= ln <= last and name in names:
                try:
                    comps = flatten_designator(names[name])
                except Exception:
                    return False  # an expression
                if comps[-1][1] or depth > 8:
                    return False  # a section or an element
                return self._whole_array(comps[0][0].lower(), first, depth + 1)
        return True

    def alloc_dims(self, path: str) -> Optional[list[str]]:
        """The recorded allocation bounds (per dimension, normalized) of the array `path`."""
        from spel.transforms.core.source import split_top_level

        parts = path.lower().split("%")
        bounds = None
        if len(parts) == 1:
            g = self.types.globals.get(parts[0])
            if g is not None and self.rec.declaration(parts[0]) is None:
                bounds = getattr(g, "alloc_bounds", None) or getattr(g, "bounds", None)
        else:
            b = self.types.base(self.sub, parts[0])
            typ = b[0] if b else None
            comp = None
            for p in parts[1:]:
                t = self.types.types.get(typ)
                comp = t.components.get(p) if t is not None else None
                if comp is None:
                    return None
                typ = str(comp.type).lower()
            bounds = getattr(comp, "bounds", None)
        if not bounds:
            return None
        dims = [d.replace(" ", "").lower() for d in split_top_level(bounds.strip().strip("()"), ",")]
        return None if any(d in ("", ":") for d in dims) else dims

    def _inquiry(self, e, ln: int) -> Optional[tuple[str, str]]:
        """(lbound|ubound, the allocation text of the queried dimension) of `lbound(a, k)`."""
        from spel.scripts.fortran_parser.scope_walk import flatten_designator

        if not isinstance(e, A.FuncExpression) or len(e.args) != 2:
            return None
        fn = _norm(e.function)
        if fn not in ("lbound", "ubound"):
            return None
        try:
            k = int(_norm(e.args[1]))
            comps = flatten_designator(e.args[0])
        except Exception:
            return None
        if comps[-1][1] or not self._whole_array(comps[0][0].lower(), ln):
            return None
        # the designator's path, associate names expanded (lbound and ubound of one array: one path)
        paths = {a.path for a in self.rec.accesses(ln=ln) if a.context == "inquiry"
                 and a.path.split("%")[-1] == comps[-1][0].lower()}
        if len(paths) != 1:
            return None
        dims = self.alloc_dims(next(iter(paths)))
        if dims is None or not 1 <= k <= len(dims):
            return None
        return fn, dims[k - 1]

    def same_bound(self, e1, ln1: int, e2, ln2: int) -> bool:
        if _norm(e1) == _norm(e2):
            return True
        q1, q2 = self._inquiry(e1, ln1), self._inquiry(e2, ln2)
        return q1 is not None and q1 == q2

    def same_header(self, l1, l2) -> bool:
        n1, n2 = l1.node, l2.node
        if n1 is None or n2 is None or l1.index is None or (l1.index or "").lower() != (l2.index or "").lower():
            return False
        if (n1.step is None) != (n2.step is None):
            return False
        return (self.same_bound(n1.start, l1.start_ln, n2.start, l2.start_ln)
                and self.same_bound(n1.end, l1.start_ln, n2.end, l2.start_ln)
                and (n1.step is None or self.same_bound(n1.step, l1.start_ln, n2.step, l2.start_ln)))

    def depth(self, a, b) -> int:
        """Levels at which the nests' headers agree (outermost first)."""
        ca, cb = chain(a), chain(b)
        k = 0
        while k < len(ca) and k < len(cb) and self.same_header(ca[k], cb[k]):
            k += 1
        return k

    # --- adjacency and the bodies ----------------------------------------------------------

    def adjacent(self, a, b) -> bool:
        return b.start_ln > a.end_ln and all(_blank_or_comment(self.lines[i]) for i in range(a.end_ln + 1, b.start_ln))

    def body(self, loop, k: int) -> Body:
        nest = chain(loop)[:k]
        inner = nest[-1]
        stmts = list(inner.node.body.statements) if inner.node is not None else []
        return Body(nest, self._header_end(inner) + 1, inner.end_ln - 1, stmts)

    def directive_on(self, first, last) -> bool:
        """An OpenACC/OpenMP directive on the nests (right above the first) or inside them."""
        p = first.start_ln - 1
        while p >= 0 and _blank_or_comment(self.lines[p]):
            p -= 1
        lo = p + 1
        if p >= 0 and self.lines[p].strip().lower().startswith("!$"):
            return True
        return any(lo <= d.start_ln <= last.end_ln for d in self.directives) or any(
            self.lines[i].strip().lower().startswith("!$omp") for i in range(lo, last.end_ln + 1))

    # --- legality --------------------------------------------------------------------------

    def _top(self, b: Body, ln: int):
        """The statement of the body that line `ln` is in."""
        from spel.transforms.core.query import construct_end

        for st in b.stmts:
            if st.lineno <= ln <= construct_end(st, self.lines):
                return st
        return None

    def why_not(self, bodies: list[Body]) -> Optional[str]:
        """Why fusing these bodies would change the computation (None: it would not)."""
        from spel.scripts.registrations import iter_statements

        fused = {(l.index or "").lower() for l in bodies[0].nest}
        acc: list[list] = []
        for b in bodies:
            for e in self.rec.calls:
                if b.first <= e.ln <= b.last:
                    return f"calls {e.name} (line {e.ln + 1})"
            for e in self.rec.function_calls:
                if b.first <= e.ln <= b.last and not e.intrinsic:
                    return f"references function {e.name} (line {e.ln + 1})"
            for st, _ in iter_statements(SimpleNamespace(statements=b.stmts)):
                if isinstance(st, NO_FUSE):
                    return f"{type(st).__name__} at line {st.lineno + 1}"
            mine = [a for a in self.rec.accesses() if b.first <= a.ln <= b.last]
            if any(a.context in ("allocate", "deallocate") for a in mine):
                return "allocates or deallocates"
            acc.append(mine)
        # a later nest's bounds read nothing an earlier body writes
        for j, b in enumerate(bodies[1:], start=1):
            written = {a.path for m in acc[:j] for a in m if "w" in a.status}
            for lev in b.nest:
                hit = written & set(lev.bound_paths)
                if hit:
                    return f"the bounds of the nest at line {b.nest[0].start_ln + 1} read {sorted(hit)[0]}, written above"
        paths = {a.path for m in acc for a in m}
        for p in sorted(paths):
            users = [j for j, m in enumerate(acc) if any(a.path == p for a in m)]
            writers = [j for j in users if any(a.path == p and "w" in a.status for a in acc[j])]
            if p in fused and writers:
                return f"writes the loop index {p}"
            if not writers or len(users) < 2:
                continue
            uses = [a for j in users for a in acc[j] if a.path == p]
            if all(a.subscripts for a in uses):
                subs = {a.subs for a in uses}
                if len(subs) != 1:
                    return f"{p} is accessed at different elements ({', '.join('(' + ','.join(s) + ')' for s in sorted(subs))})"
                if not fused <= set(next(iter(subs))):
                    return f"{p}({','.join(next(iter(subs)))}) does not take every fused index"
                continue
            # a scalar (or a whole array): a per-iteration temporary in every body that uses it
            for j in users:
                first = next(a for a in acc[j] if a.path == p)
                st = self._top(bodies[j], first.ln)
                # the body's own statement (not inside an if) is the assignment or loop that writes it
                ok = "w" in first.status and "r" not in first.status and st is not None and (
                    (isinstance(st, (A.AssignmentStatement, A.ExpressionStatement)) and st.lineno == first.ln
                     and any(e.ln == first.ln and e.lhs.path == p for e in self._assigns))
                    or (isinstance(st, A.DoLoop) and (st.index or "").lower() == p))
                if not ok:
                    return f"{p} is written in one nest and used in another (line {first.ln + 1})"
        return None

    # --- planning --------------------------------------------------------------------------

    def siblings(self, loops: list) -> None:
        # loops in branches that cannot run in the configuration are left alone
        loops = sorted((l for l in loops if self.ctx.active(self.sub, l.start_ln, l.end_ln)), key=lambda l: l.start_ln)
        i = 0
        while i < len(loops):
            group, k = [loops[i]], 0
            j = i + 1
            while j < len(loops) and self.adjacent(group[-1], loops[j]):
                d = self.depth(group[0], loops[j])
                if d == 0 or (k and d != k):
                    break
                cand = group + [loops[j]]
                if self.directive_on(cand[0], cand[-1]):
                    self.refusals.append(Refusal(self._where(cand), "an OpenACC/OpenMP directive on the nests"))
                    break
                why = self.why_not([self.body(l, d) for l in cand])
                if why is not None:
                    self.refusals.append(Refusal(self._where(cand), why))
                    break
                group, k = cand, d
                j += 1
            if len(group) > 1:
                self.changes.append(self.fuse(group, k))
                i = j
            else:
                self.siblings(loops[i].children)
                i += 1

    def _where(self, group) -> str:
        return f"{self.sub.id}:{group[0].start_ln + 1}-{group[-1].end_ln + 1}"

    def fuse(self, group: list, k: int) -> Change:
        lines = self.lines
        bodies = [self.body(l, k) for l in group]
        head, last = group[0], group[-1]
        new = lines[head.start_ln:bodies[0].last + 1]
        prev_end = head.end_ln
        for l, b in zip(group[1:], bodies[1:]):
            new += lines[prev_end + 1:l.start_ln]  # the comments between the nests
            new += [x for x in lines[l.start_ln:b.first] if code(x).strip() == "" and x.strip()]  # header comments
            new += lines[b.first:b.last + 1]
            new += [x for x in lines[b.last + 1:l.end_ln + 1] if code(x).strip() == "" and x.strip()]  # trailer comments
            prev_end = l.end_ln
        new += lines[bodies[0].last + 1:head.end_ln + 1]
        n = last.end_ln - head.start_ln + 1
        why = f"{len(group)} nests fused at depth {k} (lines {', '.join(str(l.start_ln + 1) for l in group)})"
        return Change(self.ctx.path(self.sub),
                      [replace(head.start_ln, new, n=n, expect=lines[head.start_ln:last.end_ln + 1], why=why)],
                      where=f"{self.sub.id}:{head.start_ln + 1}", why=why)


@dataclass
class Options:
    routines: list[str] = field(default_factory=list, metadata={"help": "routine ids (default: every analyzed routine)"})


@transform
class LoopFusion(Transform):
    """
    Matches: runs of adjacent sibling loop nests (only blank lines and comments between) whose
             headers agree level by level (same index and bounds; lbound/ubound of arrays with the
             same recorded allocation bounds).
    Changes: the run becomes one nest, fused to the depth at which the headers agree; the bodies
             are concatenated in order.
    Refuses: nests with directives, calls, user functions, exit/cycle/return/goto/stop or
             (de)allocations in the bodies; a variable written in one body and accessed in another
             at a different element, or as a scalar that is not a per-iteration temporary; bounds
             that read what an earlier body writes.
    Example: section-loops' `x(begc:endc) = 0; y(begc:endc) = 0` -> one loop over begc:endc.
    """

    name = "loop-fusion"
    summary = "fuse adjacent loop nests over the same iteration space (one kernel instead of several)"
    Options = Options

    def build(self, ctx, opts) -> Plan:
        from spel.transforms.acc_directives import find_directives
        from spel.transforms.acc_sync import _Types

        plan = Plan(self.name)
        types = _Types(ctx.fut)
        for sub in sorted(ctx.routines(opts.routines or None), key=lambda s: (str(ctx.path(s)), s.id)):
            rec = sub.record
            tops = [l for l in getattr(rec, "loop_nest", []) or [] if l.depth == 0]
            if len(tops) < 1:
                continue
            lines = ctx.lines(sub)
            f = Fuser(ctx, sub, lines, types, find_directives(lines, sub.startline, sub.endline),
                      ctx.facts.tree(sub))
            f.siblings(tops)
            plan.add(f.changes)
            plan.refusals += f.refusals
        return plan
