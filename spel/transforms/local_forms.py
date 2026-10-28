"""
Local variables that can take a simpler form (a report; the edits are made by hand).

- **per-iteration arrays** (`scalar`): a local array whose every use is inside one loop
  nest, always the same element (`x(fc,j)`, subscripts made of the indices of the loops
  around every use), and written before it is read in each iteration of the innermost
  loop around all uses. Its values never live from one iteration to the next, so it can be
  a scalar: a `private` scalar in a compute loop instead of device storage with enter/exit
  data (NitrifDenitrif's `r_min(fc,j)`, `soil_hr_vr(fc,j)`).
- **constants** (`parameter`): a local with an initializer that nothing writes
  (`rho_w = 1.e3_r8`), or one assigned once from a constant expression, unconditionally and
  outside any loop, before every read (`pH(:) = 6.5`). An array that holds one value
  everywhere becomes a scalar parameter.
- **dead stores** (`dead`): written, never read (`co2diff_con`).

Everything comes from the analysis: declarations, accesses, loops and assignments from
`Subroutine.record`, the enclosing ifs of each statement from the parse tree
(`registrations.line_guards`). Uses as actual arguments, in `size`/allocation contexts or
through pointers disqualify a variable (conservative).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

# intrinsics allowed in a constant expression (elemental and inquiry functions of constants)
CONST_INTRINSICS = {"tiny", "huge", "epsilon", "real", "int", "dble", "sqrt", "exp", "log", "abs",
                    "max", "min", "atan", "acos", "asin", "cos", "sin", "kind", "selected_real_kind"}
# not correctly rounded: the compiler's value of a parameter may differ in the last bit from the run time's
TRANSCENDENTAL = {"exp", "log", "atan", "acos", "asin", "cos", "sin"}
INTRINSIC_TYPES = ("integer", "real", "logical", "double precision", "doubleprecision", "complex", "character")
DISQUALIFY = {"arg", "ptr-assign", "allocate", "deallocate", "size", "nullify", "read", "stat", "iostat",
              "errmsg", "iomsg"}


@dataclass
class Finding:
    routine: str
    name: str
    kind: str  # scalar | parameter | dead
    ln: int  # 0-based: the loop (scalar), the initializer or assignment (parameter), the first write (dead)
    detail: str = ""

    def __str__(self) -> str:
        what = {"scalar": "array used one element per iteration -> scalar",
                "parameter": "constant -> parameter",
                "dead": "written, never read -> remove"}[self.kind]
        return f"{self.routine} line {self.ln + 1}: {self.name}: {what}{f' ({self.detail})' if self.detail else ''}"


def _is_const(e, rec) -> bool:
    from spel.scripts.fortran_parser import spel_ast as A

    match e:
        case A.IntegerLiteral() | A.FloatLiteral() | A.LogicalLiteral() | A.StringLiteral():
            return True
        case A.Identifier():
            d = rec.declaration(e.value.lower())
            return d is not None and "parameter" in {a.lower() for a in d.attrs}
        case A.InfixExpression():
            return _is_const(e.left_expr, rec) and _is_const(e.right_expr, rec)
        case A.PrefixExpression():
            return _is_const(e.right_expr, rec)
        case A.FuncExpression():
            return str(e.function).lower() in CONST_INTRINSICS and all(_is_const(a, rec) for a in e.args)
    return False


def _rounding_note(e) -> str:
    from spel.scripts.fortran_parser import spel_ast as A

    def calls(x):
        if isinstance(x, A.FuncExpression):
            yield str(x.function).lower()
            for a in x.args:
                yield from calls(a)
        for k in ("left_expr", "right_expr"):
            if hasattr(x, k):
                yield from calls(getattr(x, k))

    found = sorted(set(calls(e)) & TRANSCENDENTAL)
    return f"; compile-time {', '.join(found)} may differ in the last bit: check bit-for-bit" if found else ""


def _initializer(decl):
    node = decl.node
    for ent in getattr(node, "entities", None) or []:
        if ent.token.literal.lower() == decl.name.lower():
            return ent.init
    return None


def _guards_at(guards: dict[int, tuple], ln: int) -> tuple:
    """Enclosing if conditions of the statement holding line `ln` (the nearest statement at or above)."""
    starts = [s for s in guards if s <= ln]
    return guards[max(starts)] if starts else ()


def check_routine(sub, tree=None) -> list[Finding]:
    from spel.scripts.fortran_parser.scope_walk import AssignEvent
    from spel.scripts.fortran_parser.symbols import Origin
    from spel.scripts.registrations import line_guards

    rec = sub.record
    if rec is None:
        return []
    if tree is None:
        tree = rec.tree
    guards = line_guards(tree.body) if tree is not None else {}
    by_name: dict[str, list] = {}
    for a in rec.accesses(origin=Origin.LOCAL):
        by_name.setdefault(a.base.lower(), []).append(a)
    assigns = [e for e in rec.events if isinstance(e, AssignEvent)]
    out: list[Finding] = []
    for name, decl in rec.local_variables.items():
        name = name.lower()
        attrs = {a.lower() for a in decl.attrs}
        if "parameter" in attrs or attrs & {"pointer", "target"}:
            continue
        if not decl.type.lower().startswith(INTRINSIC_TYPES):
            continue
        accs = by_name.get(name, [])
        if any(a.context in DISQUALIFY for a in accs) or any(a.via is not None for a in accs):
            continue
        writes = [a for a in accs if "w" in a.status and a.context != "init"]  # (the initializer is an `init` write)
        reads = [a for a in accs if "r" in a.status]
        init = _initializer(decl)
        if init is not None and not writes:
            if _is_const(init, rec):
                out.append(Finding(sub.id, name, "parameter", decl.ln, f"initializer {init}, never written"))
            continue
        if writes and not reads:
            if init is None:  # an initialized never-read variable is reported by the compiler anyway
                out.append(Finding(sub.id, name, "dead", min(a.ln for a in writes)))
            continue
        if not writes or not reads:
            continue
        f = _constant_assignment(sub, rec, name, decl, writes, reads, assigns, guards) \
            or _per_iteration(sub, rec, name, decl, accs, writes, reads, guards)
        if f is not None:
            out.append(f)
    return sorted(out, key=lambda f: (f.ln, f.name))


def _constant_assignment(sub, rec, name, decl, writes, reads, assigns, guards) -> Optional[Finding]:
    """Assigned once, unconditionally outside loops, from a constant, before every read."""
    from spel.scripts.fortran_parser import spel_ast as A

    w_lines = {a.ln for a in writes}
    if len(w_lines) != 1 or _initializer(decl) is not None:
        return None
    (ln,) = w_lines
    if rec.enclosing_loops(ln) or _guards_at(guards, ln) or any(r.ln <= ln for r in reads):
        return None
    ev = [e for e in assigns if e.ln == ln and e.lhs.path.lower() == name]
    if len(ev) != 1 or not _is_const(ev[0].rhs, rec):
        return None
    lhs_subs = ev[0].lhs.subscripts
    if decl.dim and lhs_subs and not all(isinstance(s, A.BoundsExpression) and s.start is None and s.end is None
                                         for s in lhs_subs):
        return None  # one element of the array: not the whole array
    shape = "array of one value -> scalar parameter" if decl.dim else "assigned once"
    return Finding(sub.id, name, "parameter", ln, f"{shape}: {ev[0].rhs}{_rounding_note(ev[0].rhs)}")


def _per_iteration(sub, rec, name, decl, accs, writes, reads, guards) -> Optional[Finding]:
    """Every use in one loop iteration, one element, written before it is read."""
    if not decl.dim:
        return None
    loops = [rec.enclosing_loops(a.ln) for a in accs]
    if not all(loops):
        return None
    common = [lp for lp in loops[0] if all(lp in ls for ls in loops[1:])]
    if not common:
        return None
    indices = {lp.index for lp in common if lp.index}
    subs = {tuple(str(s).replace(" ", "").lower() for s in a.subscripts) for a in accs}
    if len(subs) != 1:
        return None
    (key,) = subs
    if not key or not all(s in indices for s in key):
        return None
    inner = common[-1]
    for r in reads:
        r_guards = _guards_at(guards, r.ln)
        dominated = any(
            w.ln < r.ln and rec.enclosing_loops(w.ln) == common
            and _guards_at(guards, w.ln) == r_guards[:len(_guards_at(guards, w.ln))]
            for w in writes
        )
        if not dominated:
            return None
    return Finding(sub.id, name, "scalar", inner.start_ln,
                   f"{name}({','.join(key)}) inside the loop on line {inner.start_ln + 1}; private there")


def check(fut, routines: Iterable[str]) -> list[Finding]:
    """The findings of the given routine ids (guards from the record's own parse tree, whose
    line numbers are the record's; the preprocessed module when the record has none)."""
    from spel.transforms.acc_apply import routine_tree

    out: list[Finding] = []
    for sid in sorted(routines):
        sub = fut.subroutine_dict.get(sid)
        if sub is None or sub.record is None:
            continue
        tree = getattr(sub.record, "tree", None)
        if tree is None:
            try:
                tree = routine_tree(sub)
            except BaseException:  # the parser exits on what it cannot read
                tree = None
        out.extend(check_routine(sub, tree))
    return out
