"""
Array-section assignments as explicit loops (`spel section-loops`).

    x(bounds%begc:bounds%endc, 1:nlevdecomp) = 1._r8
    this%core%external_to_decomp_pools(:,:,:) = 0._r8
    forc_solad_pp(:,:) = forc_solad(:,:)
    decompmicc(fc,:) = 0._r8              ! inside a column loop

A section assignment outside a kernel is a host segment of its own: the planner ports
loops, not statements, so each one kept its `update self`/`update device` pair in a
device routine (col_pf_summaryint, decomp_rate_constants_cn, RootDynamics, DUSTMod, the
CH4 setup, SurfaceRadiation). Written as a loop nest

    do spel_i2 = 1, nlevdecomp
       do spel_i1 = bounds%begc, bounds%endc
          x(spel_i1, spel_i2) = 1._r8
       end do
    end do

the planner sees an elementwise nest and ports it (collapse); inside a column loop the
nest becomes the column's sequential work.

Statements come from the record (`AssignEvent`: the left-hand side's subscripts are AST
nodes, a `BoundsExpression` is a section); the right-hand side's sections are the
`FuncExpression` nodes of the assignment's expression with a `BoundsExpression` argument.
The j-th section of a right-hand-side array pairs with the j-th section of the left-hand
side (Fortran conformance), as `lo_rhs + (index - lo_lhs)`, or the index itself when the
bounds read the same. A `:` becomes `lbound(array, k)`/`ubound(array, k)`. Only the
subscript lists are rewritten in the statement's text (the parser identified them); the
rest of the text, kind suffixes included, is kept as written. Left alone, and reported:
statements inside a compute construct, strides, a right-hand side whose sections do not
pair with the left-hand side's, or a reduction intrinsic over a section (`sum(x(:))`).
The loop indices are new integers declared before the first executable statement.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

TAG = "! spel section-loop"
INDEX = "spel_i"
# intrinsics whose result is not elementwise in their section argument
REDUCTIONS = {"sum", "product", "maxval", "minval", "any", "all", "count", "dot_product", "matmul",
              "transpose", "spread", "reshape", "pack", "unpack", "size", "shape", "lbound", "ubound",
              "maxloc", "minloc", "cshift", "eoshift", "merge", "norm2", "findloc"}
# elemental intrinsics: applied element by element, so the statement stays elementwise
ELEMENTAL = {"abs", "exp", "log", "log10", "sqrt", "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
             "sinh", "cosh", "tanh", "min", "max", "mod", "modulo", "sign", "real", "dble", "int", "nint",
             "floor", "ceiling", "aint", "anint", "dim", "huge", "tiny", "epsilon", "ieee_is_nan", "isnan"}


@dataclass
class SectionStmt:
    ln: int
    end_ln: int
    text: str  # the statement, continuations joined, comments removed
    lhs: str  # the array as written (`this%core%a`)
    lhs_subs: list  # subscript nodes
    sections: list[int]  # positions (0-based) of the sections among the subscripts
    rhs_refs: list = field(default_factory=list)  # (name as written, subscript nodes) with sections
    others: tuple = ((), ())  # other statements on the same line (`;`), before and after


def _is_section(e) -> bool:
    from spel.scripts.fortran_parser.spel_ast import BoundsExpression

    return isinstance(e, BoundsExpression)


def _norm(e) -> str:
    """Normalized text of an expression or a piece of source: no spaces, no parentheses (the
    AST renders infix expressions parenthesized, `n+1` as `(n+1)`), lower case."""
    return str(e).replace(" ", "").replace("(", "").replace(")", "").lower()


def _code(line: str) -> str:
    """The code of a line: no comment, no OpenACC directive."""
    from spel.transforms.compact.select import _code as code

    return code(line)


def statement_text(lines: list[str], ln: int, end_ln: int) -> str:
    parts = []
    for i in range(ln, end_ln + 1):
        c = _code(lines[i]).strip()
        if c.endswith("&"):
            c = c[:-1]
        if c.startswith("&"):
            c = c[1:]
        parts.append(c.strip())
    return " ".join(p for p in parts if p)


def _refs_with_sections(expr) -> list[tuple[str, list, bool]]:
    """(name as written, subscripts, inside a reduction intrinsic) of every array reference of
    the expression that has a section subscript."""
    from spel.scripts.fortran_parser import spel_ast as A

    out: list[tuple[str, list, bool]] = []

    def walk(node, in_reduction: bool):
        if isinstance(node, A.FuncExpression):
            name = str(node.function)
            args = list(node.args or [])
            red = in_reduction or name.split("%")[-1].lower() in REDUCTIONS
            if any(_is_section(a) for a in args):
                out.append((name, args, red))
            for a in args:
                walk(a, red)
            return
        if isinstance(node, A.Node):
            for v in vars(node).values():
                if isinstance(v, A.Node):
                    walk(v, in_reduction)
                elif isinstance(v, (list, tuple)):
                    for x in v:
                        if isinstance(x, A.Node):
                            walk(x, in_reduction)

    walk(expr, False)
    return out


def _array_valued_call(expr, rec) -> Optional[str]:
    """A function reference in the expression that may return an array (a user function, or an
    intrinsic that is not elemental): `rootfr(:,1:n) = zeng2001_rootfr(bounds, n, nlev2bed)`.
    The stored expression may hold unclassified `name(...)` nodes: a name the record does not
    know as a variable is a function."""
    from spel.scripts.fortran_parser import spel_ast as A
    from spel.scripts.fortran_parser.symbols import SymbolKind

    def is_call(node) -> bool:
        if isinstance(node, A.FunctionCall):
            name = node.get_name().lower()
            return not node.intrinsic or (name not in ELEMENTAL and name not in REDUCTIONS)
        if isinstance(node, A.FuncExpression) and not isinstance(node, A.ArrayRef):
            base = str(node.function).split("%")[0].lower()
            if "%" in str(node.function):
                return False  # a component reference: data
            sym = rec.lookup(base) if rec is not None else None
            if sym is not None and sym.kind is SymbolKind.VARIABLE:
                return False
            return base not in ELEMENTAL and base not in REDUCTIONS
        return False

    def walk(node):
        if is_call(node):
            return str(node.function) if isinstance(node, A.FuncExpression) else node.get_name()
        if isinstance(node, A.Node):
            for v in vars(node).values():
                if isinstance(v, A.Node):
                    r = walk(v)
                    if r:
                        return r
                elif isinstance(v, (list, tuple)):
                    for x in v:
                        if isinstance(x, A.Node):
                            r = walk(x)
                            if r:
                                return r
        return None

    return walk(expr)


def find_section_statements(sub, lines: list[str], device: list[tuple[int, int]] = (),
                            rank=None) -> tuple[list[SectionStmt], list[str]]:
    """The section assignments of a routine the transform can rewrite, and notes on those it leaves.
    `rank(name)`: the rank of a variable (0 scalar, -1 unknown); by default from the record's
    declarations (module variables are unknown)."""
    from spel.scripts.fortran_parser.scope_walk import AssignEvent
    from spel.transforms.compact.select import statement_end

    rec = sub.record
    if rank is None:
        def rank(name: str) -> int:
            d = rec.declarations.get(name.lower())
            return d.dim if d is not None else -1
    out: list[SectionStmt] = []
    notes: list[str] = []
    for ev in rec.events:
        if not isinstance(ev, AssignEvent) or not ev.lhs.subscripts:
            continue
        subs = list(ev.lhs.subscripts)
        sections = [k for k, s in enumerate(subs) if _is_section(s)]
        if not sections:
            continue
        where = f"{sub.id} line {ev.ln + 1}"
        if any(a <= ev.ln <= b for a, b in device):
            notes.append(f"{where}: inside a compute construct, left alone")
            continue
        end_ln = statement_end(lines, ev.ln)
        text = statement_text(lines, ev.ln, end_ln)
        parts = _split_statements(text)
        others = ([], [])
        if len(parts) > 1:  # `allocate(x(n)); x(:) = -1`: the part that assigns the array
            k = next((i for i, part in enumerate(parts)
                      if (pos := _assignment_split(part)) is not None
                      and re.match(r"\s*" + re.escape(ev.lhs.base.split("%")[0]) + r"\b", part, re.IGNORECASE)), None)
            if k is None:
                notes.append(f"{where}: several statements on the line, none assigns {ev.lhs.base}; left alone")
                continue
            text, others = parts[k], (parts[:k], parts[k + 1:])
        if any(_norm(s).count(":") > 1 for s in subs):
            notes.append(f"{where}: stride, left alone")
            continue
        pos = _assignment_split(text)
        if pos is None:
            notes.append(f"{where}: not a plain assignment ({text[:50]}), left alone")
            continue
        lhs_text = text[:pos].strip()
        m = re.match(r"([A-Za-z_][\w%]*)\s*\(", lhs_text)
        if m is None:
            notes.append(f"{where}: left-hand side {lhs_text[:40]} not an array reference, left alone")
            continue
        call = _array_valued_call(ev.rhs, rec)
        if call is not None:
            notes.append(f"{where}: {call}(...) on the right-hand side may return an array, left alone")
            continue
        if _has_node(ev.rhs, "ArrayInit"):
            notes.append(f"{where}: array constructor on the right-hand side, left alone")
            continue
        whole = [a.path for a in ev.reads if not a.subscripts and rank(a.base) != 0]
        if whole:
            # `hconc(1:n) = hconc_def`, a whole array on the right-hand side (or one whose rank
            # the record does not know): an array expression, not elementwise in the loop index
            notes.append(f"{where}: {whole[0]} is a whole array on the right-hand side, left alone")
            continue
        lhs_key = (m.group(1).lower(), tuple(_norm(a) for a in subs))
        refs = [(n, a, r) for n, a, r in _refs_with_sections(ev.rhs)
                if (n.lower(), tuple(_norm(x) for x in a)) != lhs_key]  # not the left-hand side itself
        ok = True
        for name, args, red in refs:
            nsec = sum(1 for a in args if _is_section(a))
            if red:
                notes.append(f"{where}: {name} inside a reduction intrinsic, left alone")
                ok = False
            elif nsec != len(sections):
                notes.append(f"{where}: {name} has {nsec} sections, the left-hand side {len(sections)}, left alone")
                ok = False
            elif any(_norm(a).count(":") > 1 for a in args):
                notes.append(f"{where}: stride in {name}, left alone")
                ok = False
        if not ok:
            continue
        out.append(SectionStmt(ev.ln, end_ln, text, m.group(1), subs, sections, [(n, a) for n, a, _ in refs], others))
    return out, notes


def _has_node(expr, class_name: str) -> bool:
    from spel.scripts.fortran_parser import spel_ast as A

    def walk(node) -> bool:
        if type(node).__name__ == class_name:
            return True
        if isinstance(node, A.Node):
            for v in vars(node).values():
                if isinstance(v, A.Node) and walk(v):
                    return True
                if isinstance(v, (list, tuple)) and any(isinstance(x, A.Node) and walk(x) for x in v):
                    return True
        return False

    return walk(expr)


def _split_statements(text: str) -> list[str]:
    """The statements of a line, split on `;` outside quotes and parentheses."""
    parts, depth, quote, cur = [], 0, "", []
    for c in text:
        if quote:
            if c == quote:
                quote = ""
        elif c in "'\"":
            quote = c
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif c == ";" and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
            continue
        cur.append(c)
    if "".join(cur).strip():
        parts.append("".join(cur).strip())
    return parts


def _assignment_split(text: str) -> Optional[int]:
    depth, quote = 0, ""
    for i, c in enumerate(text):
        if quote:
            if c == quote:
                quote = ""
            continue
        if c in "'\"":
            quote = c
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif c == "=" and depth == 0:
            if text[i + 1: i + 2] in ("=", ">") or text[i - 1: i] in ("=", "/", "<", ">"):
                return None
            return i
    return None


def _bounds(sec, array: str, dim: int) -> tuple[str, str]:
    lo = str(sec.start) if sec.start is not None else f"lbound({array},{dim})"
    hi = str(sec.end) if sec.end is not None else f"ubound({array},{dim})"
    return lo, hi


def _replace_subscripts(text: str, name: str, old_args: list, new_args: list[str]) -> Optional[str]:
    """Replace the subscript list of the reference `name(old_args)` in `text` (matched by its
    normalized form) with `new_args`. None when the reference is not found exactly once."""
    last = name.split("%")[-1]
    pat = re.compile(r"(?<![\w])" + re.escape(last) + r"\s*\(", re.IGNORECASE)
    want = ",".join(_norm(a) for a in old_args)
    hits = []
    for m in pat.finditer(text):
        i = m.end() - 1  # the '('
        depth, j = 0, i
        while j < len(text):
            if text[j] == "(":
                depth += 1
            elif text[j] == ")":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if j >= len(text):
            return None
        if _norm(text[i + 1:j]) == want:
            hits.append((i, j))
    if not hits:
        return None
    for i, j in reversed(hits):  # every occurrence (`x(:,:) * 2 + max(x(:,:), tiny)`)
        text = text[:i] + "(" + ", ".join(new_args) + ")" + text[j + 1:]
    return text


def rewrite(stmt: SectionStmt, indent: str, first_index: int = 1) -> tuple[list[str], int]:
    """The loop nest replacing the statement. Returns (lines, number of indices used)."""
    n = len(stmt.sections)
    idx = [f"{INDEX}{first_index + k}" for k in range(n)]
    # the left-hand side
    lhs_bounds = []
    new_lhs = [str(s) for s in stmt.lhs_subs]
    for k, pos in enumerate(stmt.sections):
        lo, hi = _bounds(stmt.lhs_subs[pos], stmt.lhs, pos + 1)
        lhs_bounds.append((lo, hi))
        new_lhs[pos] = idx[k]
    text = _replace_subscripts(stmt.text, stmt.lhs, stmt.lhs_subs, new_lhs)
    if text is None:
        raise ValueError(f"line {stmt.ln + 1}: left-hand side {stmt.lhs} not found in `{stmt.text}`")
    # the right-hand side references, j-th section <-> j-th left-hand-side section
    done: set[str] = set()
    for name, args in stmt.rhs_refs:
        key = name.lower() + "(" + ",".join(_norm(a) for a in args) + ")"
        if key in done:
            continue  # the same reference again: replaced with the first
        done.add(key)
        new = [str(a) for a in args]
        k = 0
        for pos, a in enumerate(args):
            if not _is_section(a):
                continue
            lo_r, _ = _bounds(a, name, pos + 1)
            lo_l, _ = lhs_bounds[k]
            same = _norm(a) == _norm(stmt.lhs_subs[stmt.sections[k]])
            new[pos] = idx[k] if same else f"{lo_r} + ({idx[k]} - ({lo_l}))"
            k += 1
        text2 = _replace_subscripts(text, name, args, new)
        if text2 is None:
            raise ValueError(f"line {stmt.ln + 1}: {name} not found in `{text}`")
        text = text2
    out = [indent + o for o in stmt.others[0]]
    for k in reversed(range(n)):  # outermost loop over the last dimension (column-major order)
        lo, hi = lhs_bounds[k]
        pad = indent + "   " * (n - 1 - k)
        out.append(f"{pad}do {idx[k]} = {lo}, {hi}" + (f"  {TAG}" if k == n - 1 else ""))
    out.append(indent + "   " * n + text)
    for k in range(n):
        out.append(indent + "   " * (n - 1 - k) + "end do")
    out += [indent + o for o in stmt.others[1]]
    return out, n


def apply_to_lines(sub, lines: list[str], stmts: list[SectionStmt], tree, notes: Optional[list[str]] = None) -> int:
    """Rewrite the statements of one routine in `lines` (entries become multi-line strings or None).
    Declares the loop indices before the routine's first executable statement (`tree`: the
    routine's parse tree). A statement whose references the text matcher cannot place is left
    alone (a note). Returns the count."""
    from spel.transforms.acc_locals import first_executable

    nmax, done = 0, 0
    for st in sorted(stmts, key=lambda s: -s.ln):
        indent = lines[st.ln][: len(lines[st.ln]) - len(lines[st.ln].lstrip())]
        try:
            new, n = rewrite(st, indent)
        except ValueError as err:
            if notes is not None:
                notes.append(f"{sub.id}: {err}; left alone")
            continue
        nmax = max(nmax, n)
        done += 1
        lines[st.ln] = "\n".join(new)
        for ln in range(st.ln + 1, st.end_ln + 1):
            lines[ln] = None
    if not done:
        return 0
    from spel.transforms.compact.select import statement_end

    # after the last declaration, not at the first executable statement: `!$acc enter data`
    # directives placed between them by acc-locals are executable
    last = getattr(sub.record, "last_decl_ln", -1)
    at = statement_end(lines, last) + 1 if last >= 0 else first_executable(tree, sub.record)
    ref = lines[last] if last >= 0 and lines[last] else lines[at]
    indent = ref[: len(ref) - len(ref.lstrip())] if ref else "    "
    decl = f"{indent}integer :: {', '.join(f'{INDEX}{k}' for k in range(1, nmax + 1))}  {TAG} indices"
    lines[at] = decl + "\n" + (lines[at] or "")
    return done


def apply(fut, routine_ids, dry_run: bool = False) -> list[str]:
    """Rewrite the section assignments of the given routines (outside kernels). Returns a log."""
    from spel.transforms.acc_apply import _Facts, _path
    from spel.transforms.acc_directives import find_directives
    from spel.transforms.acc_sync import _Types, _spans

    by_file: dict[Path, list] = {}
    for sid in sorted(set(routine_ids)):
        sub = fut.subroutine_dict.get(sid)
        if sub is not None and sub.record is not None:
            by_file.setdefault(_path(sub), []).append(sub)
    log: list[str] = []
    facts = _Facts(fut)
    types = _Types(fut)
    for path, subs in sorted(by_file.items()):
        lines = path.read_text().split("\n")
        n = 0
        for sub in subs:
            device = _spans(find_directives(lines, sub.startline, sub.endline), sub.record)

            def rank(name: str, sub=sub) -> int:
                b = types.base(sub, name.lower())
                return b[1] if b is not None else -1

            stmts, notes = find_section_statements(sub, lines, device, rank)
            log += notes
            if stmts:
                k = apply_to_lines(sub, lines, stmts, facts.tree(sub), log)
                log.append(f"{sub.id}: {k} section assignments -> loops ("
                           + ", ".join(str(s.ln + 1) for s in stmts) + ")")
                n += k
        if n and not dry_run:
            path.write_text("\n".join(l for l in lines if l is not None))
    return log
