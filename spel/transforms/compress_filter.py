"""
Compress subgrid-sized local work arrays onto the filter that uses them.

ELM routines loop over filters (``do fp = 1, num_lakep; p = filter_lakep(fp)``),
but their local work arrays are declared on the whole clump
(``real(r8) :: dqh(bounds%begp:bounds%endp)``). When every use of such an array
is inside loops over one filter, the array can be declared with the filter's
length (``dqh(num_lakep)``) and indexed by the filter position (``dqh(fp)``)
instead of the subgrid index (``dqh(p)``).

What is handled (anything else leaves the array alone, with a reason):

* loops over a filter argument: ``do j = 1, N`` with ``g = F(j)`` where ``F`` is an
  integer array dummy and ``N`` a scalar dummy. ``x(g)`` becomes ``x(j)``.
* indices derived from the filter index (``c = veg_pp%column(p)``) for arrays of
  another subgrid, only if that subgrid is listed in ``one_to_one`` (the filter
  maps one-to-one onto it, e.g. one pft per lake column).
* iteration filters: a local copy ``F2`` of ``F`` that is compacted in place
  (``k = k + 1; F2(k) = g``) together with a map array ``M`` (``M(k) = ...``).
  Inside loops over ``F2`` the position in ``F`` is ``M(j)``. If ``M`` holds offsets
  into ``begX:endX`` slices (``M = F - begX + 1``) it is rewritten to hold
  positions, which is only done if every array the callees address through ``M``
  is compressed too.
* whole arrays passed to a callee: the callee is scanned to see how it indexes the
  dummy. Arrays indexed through the map dummy (``a = filter_map(i); x(a)``) are
  passed as ``x(1:N)``; arrays indexed through the filter dummy (``n = filtern(i);
  x(n)``) keep their full bounds.
* whole-array assignments of a scalar expression (``x(begp:endp) = 0._r8``).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

SUBGRIDS = "gtlcp"
# last component of an index-mapping array -> subgrid it maps to
MAPPING_SUBGRID = {
    "gridcell": "g",
    "topounit": "t",
    "landunit": "l",
    "column": "c",
    "pft": "p",
    "patch": "p",
}

DECL_RE = re.compile(
    r"^(integer|real|logical|character|complex|double\s*precision|type\s*\(|class\s*\("
    r"|use\b|implicit\b|save\b|parameter\b|dimension\b|external\b|intrinsic\b|data\b"
    r"|procedure\b|import\b)"
)
DO_RE = re.compile(r"^(?:(\w+)\s*:\s*)?do\b\s*(.*)$")
END_DO_RE = re.compile(r"^end\s*do\b")
ASSIGN_RE = re.compile(r"^(\w+)\s*(\(.*?\))?\s*=(?!=)\s*(.*)$")
CALL_RE = re.compile(r"^call\s+(\w+)\s*(?:\((.*)\))?\s*$")
SIMPLE_GATHER_RE = re.compile(r"^(\w+)\s*=\s*(\w+)\s*\(\s*(\w+)\s*\)$")
IDENT_RE = re.compile(r"^[a-z_]\w*$")


class CompressError(Exception):
    pass


# --------------------------------------------------------------------------
# source scanning
# --------------------------------------------------------------------------
def mask_line(line: str) -> tuple[str, bool]:
    """
    The code part of a physical line with string literals blanked (same length
    and columns as `line`, lower case). Returns (masked, is_acc_directive).
    """
    stripped = line.lstrip()
    if stripped[:5].lower() == "!$acc":
        start = len(line) - len(stripped) + 5
        return " " * start + line[start:].rstrip("\n").lower(), True
    out = []
    quote = ""
    for ch in line.rstrip("\n"):
        if quote:
            out.append(" ")
            if ch == quote:
                quote = ""
        elif ch in "'\"":
            quote = ch
            out.append(" ")
        elif ch == "!":
            break
        else:
            out.append(ch.lower())
    return "".join(out), False


@dataclass
class Stmt:
    text: str  # lower case, comments/continuations removed
    start: int  # 0-based physical line numbers
    end: int
    lines: list[int] = field(default_factory=list)


def logical_statements(
    lines: list[str], start: int, end: int
) -> tuple[list[Stmt], list[str]]:
    """Statements of physical lines start..end (inclusive), plus warnings."""
    stmts: list[Stmt] = []
    warnings: list[str] = []
    cur: Optional[Stmt] = None
    for ln in range(start, end + 1):
        raw = lines[ln]
        if raw.lstrip().startswith("#"):
            warnings.append(f"preprocessor line {ln + 1} ignored")
            continue
        code, directive = mask_line(raw)
        if directive:
            continue
        code = code.strip()
        if not code:
            continue
        if cur is not None:
            code = code[1:].lstrip() if code.startswith("&") else code
        cont = code.endswith("&")
        if cont:
            code = code[:-1].rstrip()
        if cur is None:
            cur = Stmt(text=code, start=ln, end=ln, lines=[ln])
        else:
            cur.text = f"{cur.text} {code}"
            cur.end = ln
            cur.lines.append(ln)
        if not cont:
            stmts.append(cur)
            cur = None
    if cur is not None:
        stmts.append(cur)
    return stmts, warnings


def split_top_level(text: str, sep: str = ",") -> list[str]:
    parts: list[str] = []
    depth = 0
    cur = []
    for ch in text:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return parts


def matching_paren(text: str, open_idx: int) -> int:
    """Index of the parenthesis closing text[open_idx], or -1."""
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


def nospace(text: str) -> str:
    return re.sub(r"\s+", "", text)


# --------------------------------------------------------------------------
# loops and filters
# --------------------------------------------------------------------------
@dataclass
class DoLoop:
    first: int  # statement indices of `do` and `end do`
    last: int
    start: int  # physical lines of `do` and `end do`
    end: int
    var: str = ""
    lo: str = ""
    hi: str = ""
    parent: Optional["DoLoop"] = None
    # filter information (set for loops over a filter)
    filter: str = ""  # array the loop gathers from (base or derived filter)
    gidx: str = ""  # subgrid index set from the filter, e.g. p
    gidx_stmt: int = -1
    derived: dict[str, tuple[str, int]] = field(default_factory=dict)  # c -> (subgrid, stmt)

    def contains(self, istmt: int) -> bool:
        return self.first < istmt < self.last


@dataclass
class BaseFilter:
    name: str  # filter_lakep
    count: str  # num_lakep
    subgrid: str  # p


@dataclass
class DerivedFilter:
    """A local copy of a base filter, possibly compacted in place."""

    name: str
    base: str
    compacted: bool = False
    map: str = ""  # array holding each entry's position (or offset) in base
    map_kind: str = ""  # "offset" (g - begX + 1) or "position"
    map_init: Optional[int] = None  # statement index of the map initialization
    map_init_loop: Optional[DoLoop] = None
    map_compactions: list[tuple[int, DoLoop]] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


def _subgrid_of_filter(name: str, gidx: str) -> str:
    if len(gidx) == 1 and gidx in SUBGRIDS:
        return gidx
    if name[-1:] in SUBGRIDS:
        return name[-1]
    return ""


def _is_full_slice(text: str, subgrid: str, assoc: dict[str, str]) -> bool:
    text = nospace(text)
    if text == ":":
        return True
    if ":" not in text:
        return False
    lo, _, hi = text.partition(":")
    lo = assoc.get(lo, lo)
    hi = assoc.get(hi, hi)
    return lo == f"bounds%beg{subgrid}" and hi == f"bounds%end{subgrid}"


def _offset_rhs(text: str, base: str, assoc: dict[str, str]) -> Optional[str]:
    """If text is `<base> - begX + 1`, the subgrid letter X, else None."""
    m = re.match(rf"^{re.escape(base)}-([\w%]+)\+1$", nospace(text))
    if not m:
        return None
    beg = assoc.get(m.group(1), m.group(1))
    m2 = re.match(r"^bounds%beg([a-z])$", beg)
    return m2.group(1) if m2 else None


# --------------------------------------------------------------------------
# callee analysis
# --------------------------------------------------------------------------
@dataclass
class CalleeInfo:
    dummies: list[str]
    dims: dict[str, int]
    intents: dict[str, str]
    lower_one: dict[str, bool]
    # dummy array -> dummy whose entries it is indexed by (`a = M(i); x(a)`), or ""
    indexed_via: dict[str, str]


def analyze_callee(callee) -> CalleeInfo:
    lines = Path(callee.filepath).read_text().splitlines(keepends=True)
    stmts, _ = logical_statements(lines, callee.startline, callee.endline)
    dummies = [d.lower() for d in callee.dummy_args_list]
    args = {k.lower(): v for k, v in callee.arguments.items()}
    dims = {d: getattr(args.get(d), "dim", 0) for d in dummies}
    intents = {d: getattr(args.get(d), "intent", "") for d in dummies}
    lower_one = {}
    for d in dummies:
        bounds = nospace(getattr(args.get(d), "bounds", "") or "")
        first = split_top_level(bounds[1:-1])[0] if bounds.startswith("(") else ""
        lower_one[d] = first in (":", "*", "1:") or first.startswith("1:") or (
            ":" not in first and first != ""
        )

    gathered: dict[str, str] = {}  # local index var -> dummy it is read from
    for st in stmts:
        m = SIMPLE_GATHER_RE.match(st.text)
        if m and m.group(2) in dummies and dims.get(m.group(2), 0) > 0:
            gathered[m.group(1)] = m.group(2)

    indexed_via: dict[str, str] = {}
    for d in dummies:
        if dims[d] == 0 or d in gathered.values():
            continue
        via: set[str] = set()
        ok = True
        regex = re.compile(rf"(?<![%\w]){re.escape(d)}\b")
        for st in stmts[1:]:
            if DECL_RE.match(st.text):
                continue
            for m in regex.finditer(st.text):
                rest = st.text[m.end() :].lstrip()
                if not rest.startswith("("):
                    ok = False
                    continue
                open_idx = len(st.text) - len(rest)
                close = matching_paren(st.text, open_idx)
                sub = split_top_level(st.text[open_idx + 1 : close])[0].strip()
                if sub in gathered:
                    via.add(gathered[sub])
                else:
                    ok = False
        indexed_via[d] = via.pop() if ok and len(via) == 1 else ""
    return CalleeInfo(dummies, dims, intents, lower_one, indexed_via)


# --------------------------------------------------------------------------
# main transformation
# --------------------------------------------------------------------------
@dataclass
class Occurrence:
    var: str
    ln: int  # physical line
    span: tuple[int, int]  # columns of the first subscript (or name for bare refs)
    kind: str  # elem | slice | bare
    istmt: int  # -1 for directives
    replacement: str = ""  # filled for elem/slice occurrences


@dataclass
class Candidate:
    name: str
    subgrid: str
    decl_ln: int
    occurrences: list[Occurrence] = field(default_factory=list)
    bases: set[str] = field(default_factory=set)
    needs_maps: set[str] = field(default_factory=set)
    reasons: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.reasons


@dataclass
class CompressResult:
    subroutine: str
    filepath: str
    compressed: dict[str, str]  # var -> new size expression
    skipped: dict[str, str]  # var -> reason
    maps_rewritten: list[str]
    new_lines: list[str]
    warnings: list[str]
    output: Optional[Path] = None

    def summary(self) -> str:
        out = [f"compress_on_filter: {self.subroutine} ({self.filepath})"]
        out.append(f"  compressed {len(self.compressed)} arrays:")
        for var, size in sorted(self.compressed.items()):
            out.append(f"    {var}({size})")
        if self.maps_rewritten:
            out.append(f"  filter maps rewritten to positions: {', '.join(self.maps_rewritten)}")
        out.append(f"  kept {len(self.skipped)} arrays:")
        for var, why in sorted(self.skipped.items()):
            out.append(f"    {var}: {why}")
        for w in self.warnings:
            out.append(f"  warning: {w}")
        if self.output:
            out.append(f"  wrote {self.output}")
        return "\n".join(out)


class _Compressor:
    def __init__(self, sub, callees: dict, one_to_one: Iterable[str], logger):
        self.sub = sub
        self.logger = logger
        self.one_to_one = {s.lower() for s in one_to_one}
        self.callees = {k.lower(): v for k, v in callees.items()}
        self.callee_info: dict[str, Optional[CalleeInfo]] = {}
        self.lines = Path(sub.filepath).read_text().splitlines(keepends=True)
        self.stmts, self.warnings = logical_statements(
            self.lines, sub.startline, sub.endline
        )
        self.assoc = {k.lower(): v.lower() for k, v in sub.associate_vars.items()}
        self.args = {k.lower(): v for k, v in sub.arguments.items()}
        self.locals = {k.lower(): v for k, v in sub.local_variables.items()}
        self.stmt_of_line: dict[int, int] = {}
        for i, st in enumerate(self.stmts):
            for ln in st.lines:
                self.stmt_of_line[ln] = i
        self.first_exec = self._first_executable()
        self.loops = self._build_loops()
        self.base_filters: dict[str, BaseFilter] = {}
        self.derived: dict[str, DerivedFilter] = {}

    # ---- structure ------------------------------------------------------
    def _first_executable(self) -> int:
        for i, st in enumerate(self.stmts[1:], start=1):
            if not DECL_RE.match(st.text):
                return i
        return len(self.stmts)

    def _build_loops(self) -> list[DoLoop]:
        loops: list[DoLoop] = []
        stack: list[DoLoop] = []
        for i, st in enumerate(self.stmts):
            m = DO_RE.match(st.text)
            if m and i >= self.first_exec:
                rest = m.group(2)
                if re.match(r"^\d", rest):
                    raise CompressError(f"labeled DO at line {st.start + 1} unsupported")
                loop = DoLoop(first=i, last=-1, start=st.start, end=-1)
                mc = re.match(r"^(\w+)\s*=\s*(.*)$", rest)
                if mc and not rest.startswith("while"):
                    loop.var = mc.group(1)
                    bounds = [b.strip() for b in split_top_level(mc.group(2))]
                    loop.lo, loop.hi = bounds[0], bounds[1] if len(bounds) > 1 else ""
                    if len(bounds) > 2 and nospace(bounds[2]) != "1":
                        loop.var = ""  # strided loops are not filter loops
                loop.parent = stack[-1] if stack else None
                stack.append(loop)
                loops.append(loop)
            elif END_DO_RE.match(st.text):
                if not stack:
                    raise CompressError(f"unmatched END DO at line {st.start + 1}")
                loop = stack.pop()
                loop.last, loop.end = i, st.end
        if stack:
            raise CompressError("unterminated DO loop")
        return loops

    def body(self, loop: DoLoop, direct: bool = False) -> list[int]:
        """Statement indices inside loop (only its own level if direct)."""
        idx = range(loop.first + 1, loop.last)
        if not direct:
            return list(idx)
        inner = [lp for lp in self.loops if lp.parent is loop]
        return [i for i in idx if not any(lp.first <= i <= lp.last for lp in inner)]

    def enclosing(self, istmt: int) -> list[DoLoop]:
        """Loops containing a statement, innermost first."""
        loops = [lp for lp in self.loops if lp.contains(istmt)]
        return sorted(loops, key=lambda lp: lp.first, reverse=True)

    def action(self, istmt: int) -> str:
        """Statement text, without the condition of a one-line IF."""
        text = self.stmts[istmt].text
        m = re.match(r"^if\s*\(", text)
        if m:
            close = matching_paren(text, m.end() - 1)
            rest = text[close + 1 :].strip() if close > 0 else ""
            if rest and rest != "then":
                return rest
        return text

    def assign(self, istmt: int):
        return ASSIGN_RE.match(self.action(istmt))

    def assignments(self, var: str, stmts: Iterable[int]) -> list[int]:
        out = []
        for i in stmts:
            text = self.action(i)
            m = ASSIGN_RE.match(text)
            if m and m.group(1) == var:
                out.append(i)
            elif text.startswith("call "):
                _, args = self.call_args(i)
                if any(nospace(a) == var for _, a in args):
                    out.append(i)  # may be written by the callee
            elif re.match(rf"^(?:\w+\s*:\s*)?do\s+{var}\s*=", text):
                out.append(i)
            elif re.match(rf"^read\s*\(", text) and re.search(rf"\b{var}\b", text):
                out.append(i)
        return out

    # ---- filters --------------------------------------------------------
    def _int_array(self, name: str, table: dict) -> bool:
        var = table.get(name)
        return bool(var) and var.type == "integer" and var.dim == 1

    def find_filters(self) -> None:
        pairs: dict[str, set[str]] = {}
        gathers: list[tuple[DoLoop, int, str, str]] = []
        for loop in self.loops:
            if not loop.var or nospace(loop.lo) != "1":
                continue
            for i in self.body(loop, direct=True):
                m = SIMPLE_GATHER_RE.match(self.stmts[i].text)
                if m and m.group(3) == loop.var:
                    gathers.append((loop, i, m.group(1), m.group(2)))
                    break
        for loop, i, gidx, arr in gathers:
            if self._int_array(arr, self.args):
                count = nospace(loop.hi)
                scalar = self.args.get(count)
                if scalar is not None and scalar.dim == 0:
                    pairs.setdefault(arr, set()).add(count)
        for arr, counts in pairs.items():
            if len(counts) != 1:
                self.warnings.append(f"filter {arr} looped with counts {sorted(counts)}")
                continue
            gidx = next(g for lp, _, g, a in gathers if a == arr)
            sg = _subgrid_of_filter(arr, gidx)
            if sg:
                self.base_filters[arr] = BaseFilter(arr, counts.pop(), sg)

        for name in self.locals:
            if self._int_array(name, self.locals):
                df = self._derived_filter(name)
                if df:
                    self.derived[name] = df

        for loop, i, gidx, arr in gathers:
            if arr in self.base_filters:
                if nospace(loop.hi) != self.base_filters[arr].count:
                    continue
            elif arr not in self.derived:
                continue
            loop.filter, loop.gidx, loop.gidx_stmt = arr, gidx, i
            body = self.body(loop)
            if self.assignments(gidx, body) != [i]:
                loop.filter = ""
                self.warnings.append(f"index {gidx} reassigned in loop at line {loop.start + 1}")
                continue
            for j in self.body(loop, direct=True):
                m = re.match(rf"^(\w+)\s*=\s*([\w%]+)\s*\(\s*{gidx}\s*\)$", self.stmts[j].text)
                if not m or m.group(1) == gidx:
                    continue
                sg = MAPPING_SUBGRID.get(m.group(2).split("%")[-1], "")
                if sg and self.assignments(m.group(1), body) == [j]:
                    loop.derived[m.group(1)] = (sg, j)

    def _derived_filter(self, name: str) -> Optional[DerivedFilter]:
        """`name` is a copy of a base filter, optionally compacted with a map."""
        writes = [
            i
            for i in range(self.first_exec, len(self.stmts))
            if (m := ASSIGN_RE.match(self.stmts[i].text)) and m.group(1) == name
        ]
        if not writes:
            return None
        df: Optional[DerivedFilter] = None
        compactions: list[tuple[int, DoLoop, str]] = []
        for i in writes:
            m = ASSIGN_RE.match(self.stmts[i].text)
            lhs_sub = nospace(m.group(2) or "")
            rhs = nospace(m.group(3))
            mcopy = re.match(r"^(\w+)(\((.*)\))?$", rhs)
            if mcopy and mcopy.group(1) in self.base_filters and not self.enclosing(i):
                base = self.base_filters[mcopy.group(1)]
                full = ("", "(:)", f"(1:{base.count})")
                if lhs_sub in full and nospace(mcopy.group(2) or "") in full:
                    df = df or DerivedFilter(name=name, base=base.name)
                    continue
            loops = self.enclosing(i)
            loop = loops[0] if loops else None
            if loop and loop.var and lhs_sub.startswith("("):
                k = lhs_sub[1:-1]
                gm = self._gather_in(loop, name)
                if gm and rhs == gm and k != loop.var:
                    compactions.append((i, loop, k))
                    continue
            return None
        if df is None:
            return None
        for i, loop, k in compactions:
            if not self._is_counter(k, loop, i):
                return None
        df.compacted = bool(compactions)
        if df.compacted:
            self._find_map(df, compactions)
        return df

    def _gather_in(self, loop: DoLoop, arr: str) -> str:
        for i in self.body(loop, direct=True):
            m = SIMPLE_GATHER_RE.match(self.stmts[i].text)
            if m and m.group(2) == arr and m.group(3) == loop.var:
                return m.group(1)
        return ""

    def _is_counter(self, k: str, loop: DoLoop, istmt: int) -> bool:
        """k = 0 before the loop and k = k + 1 inside it before istmt."""
        inc = [
            i
            for i in self.assignments(k, self.body(loop))
            if nospace(self.stmts[i].text) == f"{k}={k}+1"
        ]
        if not inc or not all(i < istmt for i in inc):
            return False
        if len(self.assignments(k, self.body(loop))) != len(inc):
            return False
        before = self.assignments(k, range(self.first_exec, loop.first))
        return bool(before) and nospace(self.stmts[before[-1]].text) == f"{k}=0"

    def _find_map(self, df: DerivedFilter, compactions) -> None:
        """A local integer array written alongside each compaction of df."""
        base = self.base_filters[df.base]
        candidates: dict[str, list[tuple[int, DoLoop]]] = {}
        for i, loop, k in compactions:
            for j in self.body(loop):
                m = ASSIGN_RE.match(self.stmts[j].text)
                if (
                    m
                    and m.group(1) != df.name
                    and nospace(m.group(2) or "") == f"({k})"
                    and self._int_array(m.group(1), self.locals)
                ):
                    candidates.setdefault(m.group(1), []).append((j, loop))
        for name, comps in candidates.items():
            if len(comps) != len(compactions):
                continue
            writes = [
                i
                for i in range(self.first_exec, len(self.stmts))
                if (m := ASSIGN_RE.match(self.stmts[i].text)) and m.group(1) == name
            ]
            inits = [i for i in writes if i not in {j for j, _ in comps}]
            if len(inits) != 1 or self.enclosing(inits[0]):
                continue
            m = ASSIGN_RE.match(self.stmts[inits[0]].text)
            lhs_sub = nospace(m.group(2) or "")
            if lhs_sub not in ("", "(:)", f"(1:{base.count})"):
                continue
            rhs = nospace(m.group(3))
            src = re.sub(rf"\((1:{base.count}|:)\)", "", rhs, count=1)
            init_sg = _offset_rhs(src, base.name, self.assoc)
            kinds = set()
            for j, loop in comps:
                crhs = nospace(ASSIGN_RE.match(self.stmts[j].text).group(3))
                g = self._gather_in(loop, df.name)
                if init_sg and _offset_rhs(crhs, g, self.assoc) == init_sg:
                    kinds.add("offset")
                elif crhs == f"{name}({loop.var})":
                    kinds.add("position")
                else:
                    kinds.add("?")
            if kinds == {"offset"} and init_sg == base.subgrid:
                df.map, df.map_kind = name, "offset"
            elif kinds == {"position"} and re.match(r"^\(/.*/\)$|^\[.*\]$", rhs):
                df.map, df.map_kind = name, "position"
            else:
                continue
            df.map_init = inits[0]
            df.map_compactions = comps
            return

    # ---- positions -------------------------------------------------------
    def position(self, loop: DoLoop) -> tuple[str, str, str]:
        """(base filter, position expression, map needed) for a filter loop."""
        if loop.filter in self.base_filters:
            return loop.filter, loop.var, ""
        df = self.derived[loop.filter]
        if not df.compacted:
            return df.base, loop.var, ""
        if not df.map:
            return df.base, "", ""
        need = df.map if df.map_kind == "offset" else ""
        return df.base, f"{df.map}({loop.var})", need

    def callee(self, name: str):
        if name not in self.callee_info:
            sub = self.callees.get(name)
            try:
                self.callee_info[name] = analyze_callee(sub) if sub else None
            except Exception as exc:  # unreadable callee: treat as unknown
                self.logger.warning(f"could not analyze callee {name}: {exc}")
                self.callee_info[name] = None
        return self.callee_info[name]

    def call_args(self, istmt: int) -> tuple[str, list[tuple[str, str]]]:
        """(callee name, [(dummy or '', actual text)])"""
        m = CALL_RE.match(self.action(istmt))
        if not m:
            return "", []
        name = m.group(1)
        actuals = [a.strip() for a in split_top_level(m.group(2) or "")] if m.group(2) else []
        info = self.callee(name)
        out = []
        for pos, act in enumerate(actuals):
            mk = re.match(r"^(\w+)\s*=(?!=)\s*(.*)$", act)
            if mk:
                out.append((mk.group(1), mk.group(2)))
            else:
                dummy = info.dummies[pos] if info and pos < len(info.dummies) else ""
                out.append((dummy, act))
        return name, out

    # ---- candidates ------------------------------------------------------
    def candidates(self) -> tuple[dict[str, Candidate], dict[str, str]]:
        cands: dict[str, Candidate] = {}
        skipped: dict[str, str] = {}
        for name, var in self.locals.items():
            if var.dim < 1:
                continue
            m = re.match(r"^\((bounds%beg(\w)):(bounds%end(\w))[,)]", nospace(var.bounds))
            if not m or m.group(2) != m.group(4):
                continue
            if var.parameter or var.allocatable or var.pointer:
                skipped[name] = "pointer/allocatable/parameter"
                continue
            if name in self.assoc:
                skipped[name] = f"shadowed by associate name ({name} => {self.assoc[name]})"
                continue
            decl = var.ln
            if not re.search(rf"\b{name}\s*\(", mask_line(self.lines[decl])[0]):
                raise CompressError(
                    f"declaration of {name} not at line {decl + 1} of {self.sub.filepath};"
                    " the analysis is stale (re-run with --reanalyze)"
                )
            cands[name] = Candidate(name=name, subgrid=m.group(2), decl_ln=decl)
        return cands, skipped

    def scan(self, cand: Candidate) -> None:
        regex = re.compile(rf"(?<![%\w]){cand.name}\b")
        last_ln = self.stmts[-1].end
        start_ln = self.stmts[self.first_exec].start if self.first_exec < len(self.stmts) else last_ln
        for ln in range(start_ln, last_ln + 1):
            code, directive = mask_line(self.lines[ln])
            for m in regex.finditer(code):
                self._classify(cand, ln, code, m, directive)

    def _classify(self, cand: Candidate, ln: int, code: str, m, directive: bool) -> None:
        name = cand.name
        istmt = -1 if directive else self.stmt_of_line.get(ln, -1)
        after = code[m.end() :]
        rest = after.lstrip()
        if not directive and istmt < 0:
            return
        stmt = self.stmts[istmt] if istmt >= 0 else None
        is_call = bool(stmt) and self.action(istmt).startswith("call ")
        if is_call and re.match(r"^=(?!=)", rest):
            return  # keyword name of a dummy argument
        if rest.startswith("("):
            open_idx = len(code) - len(rest)
            close = matching_paren(code, open_idx)
            if close < 0:
                cand.reasons.append(f"subscript spans lines at line {ln + 1}")
                return
            first = split_top_level(code[open_idx + 1 : close])[0]
            fstart = open_idx + 1
            fend = fstart + len(first)
            a0 = first.strip()
            if _is_full_slice(a0, cand.subgrid, self.assoc):
                kind = "slice"
            elif IDENT_RE.match(a0):
                kind = "elem"
            else:
                cand.reasons.append(f"index '{a0}' at line {ln + 1}")
                return
            occ = Occurrence(name, ln, (fstart, fend), kind, istmt)
            ref = nospace(code[m.start() : close + 1])
        else:
            kind = "bare"
            occ = Occurrence(name, ln, (m.start(), m.end()), kind, istmt)
            ref = name

        if directive:
            if kind == "elem":
                cand.reasons.append(f"element reference in directive at line {ln + 1}")
                return
            occ.replacement = f"1:SIZE" if kind == "slice" else ""
            cand.occurrences.append(occ)
            return

        if kind == "elem":
            self._classify_elem(cand, occ, a0, ref, is_call)
        elif is_call:
            self._classify_actual(cand, occ, ref)
        else:
            self._classify_whole(cand, occ, ref)

    def _classify_elem(self, cand: Candidate, occ: Occurrence, idx: str, ref: str, is_call: bool) -> None:
        line = occ.ln + 1
        for loop in self.enclosing(occ.istmt):
            if not loop.filter:
                continue
            if idx == loop.gidx:
                sg, set_at = _subgrid_of_filter(loop.filter, loop.gidx), loop.gidx_stmt
                if loop.filter in self.derived:
                    sg = self.base_filters[self.derived[loop.filter].base].subgrid
            elif idx in loop.derived:
                sg, set_at = loop.derived[idx]
            else:
                continue
            base, pos, need = self.position(loop)
            if occ.istmt <= set_at:
                cand.reasons.append(f"{cand.name}({idx}) used before {idx} is set at line {line}")
                return
            base_sg = self.base_filters[base].subgrid
            if sg != cand.subgrid:
                cand.reasons.append(f"indexed by {idx} (subgrid {sg}) at line {line}")
                return
            if sg != base_sg and sg not in self.one_to_one:
                cand.reasons.append(
                    f"{sg}-index {idx} derived from {base} ({base_sg}) at line {line};"
                    f" allow with one_to_one={sg!r} if each {base_sg} has its own {sg}"
                )
                return
            if not pos:
                cand.reasons.append(
                    f"loop at line {loop.start + 1} over compacted {loop.filter} has no position map"
                )
                return
            if is_call and not self._scalar_actual(occ.istmt, ref):
                cand.reasons.append(f"element passed to a non-scalar dummy at line {line}")
                return
            cand.bases.add(base)
            if need:
                cand.needs_maps.add(need)
            occ.replacement = pos
            cand.occurrences.append(occ)
            return
        cand.reasons.append(f"{cand.name}({idx}) outside a loop over a filter at line {line}")

    def _scalar_actual(self, istmt: int, ref: str) -> bool:
        name, args = self.call_args(istmt)
        info = self.callee(name)
        for dummy, act in args:
            if nospace(act) == ref:
                return bool(info) and info.dims.get(dummy, 1) == 0
        return True  # part of an expression: passed by value

    def _classify_actual(self, cand: Candidate, occ: Occurrence, ref: str) -> None:
        line = occ.ln + 1
        name, args = self.call_args(occ.istmt)
        info = self.callee(name)
        match = [(d, a) for d, a in args if nospace(a) == ref]
        if not match:
            cand.reasons.append(f"whole array in an expression in call {name} at line {line}")
            return
        if not info:
            cand.reasons.append(f"passed to unanalyzed callee {name} at line {line}")
            return
        dummy = match[0][0]
        via = info.indexed_via.get(dummy, "")
        if not via:
            cand.reasons.append(f"{name} does not index dummy {dummy} through a filter (line {line})")
            return
        actual_via = nospace(dict(args).get(via, ""))
        maps = {df.map: df for df in self.derived.values() if df.map}
        if actual_via in maps and info.lower_one.get(dummy):
            df = maps[actual_via]
            cand.bases.add(df.base)
            if df.map_kind == "offset":
                cand.needs_maps.add(df.map)
            occ.replacement = "1:SIZE" if occ.kind == "slice" else ""
            cand.occurrences.append(occ)
            return
        cand.reasons.append(
            f"{name} indexes dummy {dummy} by subgrid index (via {via}) at line {line}"
        )

    def _classify_whole(self, cand: Candidate, occ: Occurrence, ref: str) -> None:
        line = occ.ln + 1
        m = self.assign(occ.istmt) if self.action(occ.istmt) == self.stmts[occ.istmt].text else None
        lhs = nospace(f"{m.group(1)}{m.group(2) or ''}") if m else ""
        if m and lhs == ref and "(" not in m.group(3) and not self._mentions_array(m.group(3)):
            occ.replacement = "1:SIZE" if occ.kind == "slice" else ""
            cand.occurrences.append(occ)
            return
        cand.reasons.append(f"whole-array use at line {line}")

    def _mentions_array(self, text: str) -> bool:
        for word in re.findall(r"(?<![%\w])([a-z_]\w*)", text):
            var = self.locals.get(word) or self.args.get(word)
            if var is not None and var.dim > 0:
                return True
        return False

    # ---- maps --------------------------------------------------------------
    def map_users(self, df: DerivedFilter) -> tuple[list[tuple[int, str]], list[str]]:
        """
        Calls passing df.map: [(stmt, actual array expression addressed through it)]
        and problems (other reads of the map).
        """
        users: list[tuple[int, str]] = []
        problems: list[str] = []
        regex = re.compile(rf"(?<![%\w]){df.map}\b")
        writes = {df.map_init} | {j for j, _ in df.map_compactions}
        for i in range(self.first_exec, len(self.stmts)):
            st = self.stmts[i]
            if i in writes or not regex.search(st.text):
                continue
            if self.action(i).startswith("call "):
                name, args = self.call_args(i)
                info = self.callee(name)
                dummy = next((d for d, a in args if nospace(a) == df.map), None)
                if info is None or dummy is None or info.intents.get(dummy) != "r":
                    problems.append(f"passed to {name} at line {st.start + 1}")
                    continue
                for d, a in args:
                    if info.indexed_via.get(d) == dummy:
                        users.append((i, a))
                continue
            # reads as a position in loops over the derived filter are fine
            loops = self.enclosing(i)
            texts = regex.split(st.text)
            if loops and all(t.startswith(f"({loops[0].var})") for t in texts[1:]) and loops[0].filter == df.name:
                continue
            problems.append(f"used at line {st.start + 1}")
        return users, problems

    def resolve(self, cands: dict[str, Candidate]) -> dict[str, bool]:
        """Fixed point: maps are rewritten only if all arrays read through them compress."""
        feasible: dict[str, bool] = {}
        while True:
            changed = False
            for df in self.derived.values():
                if not df.map or df.map_kind != "offset":
                    continue
                users, problems = self.map_users(df)
                bad = list(problems)
                for i, act in users:
                    m = re.match(r"^(\w+)(\(.*\))?$", nospace(act))
                    c = cands.get(m.group(1)) if m else None
                    if c is None or not c.ok:
                        bad.append(f"{act} at line {self.stmts[i].start + 1} cannot be compressed")
                df.problems = bad
                feasible[df.map] = not bad
            for c in cands.values():
                if not c.ok:
                    continue
                missing = [m for m in c.needs_maps if not feasible.get(m, False)]
                if missing:
                    df = next(d for d in self.derived.values() if d.map == missing[0])
                    c.reasons.append(f"needs map {missing[0]} rewritten, but {'; '.join(df.problems)}")
                    changed = True
                elif len(c.bases) > 1:
                    c.reasons.append(f"used with several filters: {sorted(c.bases)}")
                    changed = True
                elif not c.bases:
                    c.reasons.append("not used in any filter loop")
                    changed = True
            if not changed:
                return feasible

    # ---- edits -------------------------------------------------------------
    def apply(self, cands: dict[str, Candidate], feasible: dict[str, bool]):
        edits: dict[int, list[tuple[int, int, str]]] = {}
        replace_stmt: dict[int, str] = {}  # first line -> new text (other lines deleted)
        delete: set[int] = set()
        compressed: dict[str, str] = {}

        def add(ln, start, end, text):
            edits.setdefault(ln, []).append((start, end, text))

        for c in cands.values():
            if not c.ok:
                continue
            size = self.base_filters[next(iter(c.bases))].count
            compressed[c.name] = size
            for occ in c.occurrences:
                if occ.replacement:
                    add(occ.ln, *occ.span, occ.replacement.replace("SIZE", size))
            code, _ = mask_line(self.lines[c.decl_ln])
            m = re.search(
                rf"\b{c.name}\s*\(\s*(bounds%beg{c.subgrid}\s*:\s*bounds%end{c.subgrid})\s*",
                code,
            )
            add(c.decl_ln, m.start(1), m.end(1), size)

        used_maps = {m for c in cands.values() if c.ok for m in c.needs_maps}
        rewritten = []
        for df in self.derived.values():
            if df.map not in used_maps or not feasible.get(df.map):
                continue
            rewritten.append(df.map)
            base = self.base_filters[df.base]
            ivar = next(
                (lp.var for lp in self.loops if lp.filter == base.name), ""
            )
            if not ivar:
                raise CompressError(f"no loop variable over {base.name} to initialize {df.map}")
            st = self.stmts[df.map_init]
            indent = re.match(r"^\s*", self.lines[st.start]).group()
            replace_stmt[st.start] = (
                f"{indent}{df.map}(1:{base.count}) = (/ ({ivar}, {ivar} = 1, {base.count}) /)\n"
            )
            delete.update(st.lines[1:])
            for j, loop in df.map_compactions:
                st = self.stmts[j]
                m = ASSIGN_RE.match(st.text)
                indent = re.match(r"^\s*", self.lines[st.start]).group()
                replace_stmt[st.start] = f"{indent}{df.map}{nospace(m.group(2))} = {df.map}({loop.var})\n"
                delete.update(st.lines[1:])
            users, _ = self.map_users(df)
            for i in {i for i, _ in users}:
                self._resize_size_actuals(i, base, add)
            decl = self.locals[df.map].ln
            code, _ = mask_line(self.lines[decl])
            if "!" in self.lines[decl][len(code.rstrip()) :]:
                cut = self.lines[decl].index("!", len(code.rstrip()))
                add(
                    decl,
                    cut,
                    len(self.lines[decl].rstrip("\n")),
                    f"! position in {base.name} of each {df.name} entry (compressed work-array index)",
                )

        new_lines: list[str] = []
        for ln, line in enumerate(self.lines):
            if ln in delete:
                continue
            if ln in replace_stmt:
                new_lines.append(replace_stmt[ln])
                continue
            for start, end, text in sorted(edits.get(ln, []), reverse=True):
                line = line[:start] + text + line[end:]
            new_lines.append(line)
        return compressed, rewritten, new_lines

    def _resize_size_actuals(self, istmt: int, base: BaseFilter, add) -> None:
        """`endX-begX+1` actuals (length of the old slices) become the filter count."""
        st = self.stmts[istmt]
        _, args = self.call_args(istmt)
        sg = base.subgrid
        for _, act in args:
            m = re.match(r"^([\w%]+)-([\w%]+)\+1$", nospace(act))
            if not m:
                continue
            end = self.assoc.get(m.group(1), m.group(1))
            beg = self.assoc.get(m.group(2), m.group(2))
            if end != f"bounds%end{sg}" or beg != f"bounds%beg{sg}":
                continue
            pattern = r"\s*".join(re.escape(ch) for ch in nospace(act))
            for ln in st.lines:
                code, _ = mask_line(self.lines[ln])
                for mm in re.finditer(rf"(?<![\w%]){pattern}(?![\w%])", code):
                    add(ln, mm.start(), mm.end(), base.count)


def compress_on_filter(
    sub,
    callees: Optional[dict] = None,
    one_to_one: Iterable[str] = (),
    output: Optional[Path | str] = None,
    inplace: bool = False,
    logger: Optional[logging.Logger] = None,
) -> CompressResult:
    """
    Compress the subgrid-sized local arrays of `sub` (an analyzed Subroutine) onto
    the filters that use them.

    callees: name (lower case, as called) -> analyzed Subroutine, used to see how
        callees index whole arrays passed to them. Defaults to sub.child_subroutines.
    one_to_one: subgrid letters (e.g. "c") onto which the routine's filters map
        one-to-one, so arrays of that subgrid can share the filter's positions.
    output: file to write (default: `spel_output_dir`/<file name>); inplace writes
        the routine's own source file.
    """
    logger = logger or getattr(sub, "logger", None) or logging.getLogger("compress")
    if callees is None:
        callees = {}
        for key, child in (getattr(sub, "child_subroutines", None) or {}).items():
            callees[key.split("::")[-1]] = child
        for desc in (getattr(sub, "sub_call_desc", None) or {}).values():
            child = (sub.child_subroutines or {}).get(desc.fn)
            if child is not None:
                callees[desc.alias] = child

    comp = _Compressor(sub, callees, one_to_one, logger)
    comp.find_filters()
    cands, skipped = comp.candidates()
    for cand in cands.values():
        comp.scan(cand)
    feasible = comp.resolve(cands)
    compressed, rewritten, new_lines = comp.apply(cands, feasible)
    for c in cands.values():
        if not c.ok:
            skipped[c.name] = c.reasons[0]

    result = CompressResult(
        subroutine=f"{sub.module}::{sub.name}",
        filepath=str(sub.filepath),
        compressed=compressed,
        skipped=skipped,
        maps_rewritten=rewritten,
        new_lines=new_lines,
        warnings=comp.warnings,
    )
    if inplace:
        out = Path(sub.filepath)
    elif output is not None:
        out = Path(output)
    else:
        from spel.scripts.config import spel_output_dir

        out = Path(spel_output_dir) / Path(sub.filepath).name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(new_lines))
    result.output = out
    logger.info(result.summary())
    return result
