"""
OpenACC directives in Fortran source.

SPEL's parser treats `!$acc` lines as comments, so the analysis knows nothing
about them. This module finds them in the source text and parses each one
(with its continuation lines) into an AccDirective: the construct
(`parallel loop`, `data`, `end data`, ...) and its clauses.

Line numbers are 0-based indices into the file's lines, as in SPEL's analysis.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

SENTINEL_RE = re.compile(r"^(\s*)!\$acc\b", re.IGNORECASE)
IDENT_RE = re.compile(r"[a-z_][a-z0-9_]*")
CODE_RE = re.compile(r"^\s*[^!\s]")

# construct names, longest first so that "parallel loop" wins over "parallel"
CONSTRUCTS = sorted(
    [
        "parallel loop",
        "kernels loop",
        "serial loop",
        "end parallel loop",
        "end kernels loop",
        "end serial loop",
        "enter data",
        "exit data",
        "end parallel",
        "end kernels",
        "end serial",
        "end data",
        "end host_data",
        "end atomic",
        "host_data",
        "parallel",
        "kernels",
        "serial",
        "loop",
        "data",
        "update",
        "routine",
        "declare",
        "atomic",
        "wait",
        "cache",
        "init",
        "shutdown",
        "set",
    ],
    key=lambda c: -len(c),
)
COMPUTE_LOOPS = {"parallel loop", "kernels loop", "serial loop"}
COMPUTE_REGIONS = {"parallel", "kernels", "serial"}
DATA_CONSTRUCTS = {
    "data",
    "end data",
    "enter data",
    "exit data",
    "update",
    "declare",
    "host_data",
    "end host_data",
    "wait",
    "cache",
    "init",
    "shutdown",
    "set",
}


class DirectiveError(Exception):
    pass


def split_top_level(text: str, sep: str = ",") -> list[str]:
    """`text` split on `sep` outside parentheses, items stripped."""
    items, depth, cur = [], 0, []
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == sep and depth == 0:
            items.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    tail = "".join(cur).strip()
    if tail:
        items.append(tail)
    return items


@dataclass
class Clause:
    """`name(args)`; args is the text inside the parentheses (None if absent)."""

    name: str
    args: Optional[str] = None

    def items(self) -> list[str]:
        """Comma-separated items of the argument list (e.g. private variables)."""
        return split_top_level(self.args) if self.args else []

    def variables(self) -> list[str]:
        """
        Variable names of a var-list clause: array sections and derived-type
        components reduce to their base name (`a(:,:)` -> `a`).
        """
        out = []
        for item in self.items():
            m = IDENT_RE.match(item.strip())
            if m and m.group(0) not in out:
                out.append(m.group(0))
        return out

    def reduction(self) -> tuple[str, list[str]]:
        """(operator, variables) of a `reduction(op:vars)` clause."""
        if self.args is None or ":" not in self.args:
            raise DirectiveError(f"malformed reduction clause: {self}")
        op, vars_ = self.args.split(":", 1)
        return op.strip(), Clause("vars", vars_).variables()

    def __str__(self) -> str:
        return self.name if self.args is None else f"{self.name}({self.args})"


@dataclass
class AccDirective:
    """
    start_ln/end_ln: first/last line index (continuations included)
    construct:       normalized construct name, e.g. "parallel loop"
    construct_args:  parenthesized argument of the construct itself
                     (`routine(name)`, `wait(1)`, `cache(a)`)
    """

    start_ln: int
    end_ln: int
    indent: str
    construct: str
    clauses: list[Clause] = field(default_factory=list)
    construct_args: Optional[str] = None
    text: str = ""

    def clause(self, name: str) -> Optional[Clause]:
        return next((c for c in self.clauses if c.name == name), None)

    def clauses_named(self, name: str) -> list[Clause]:
        return [c for c in self.clauses if c.name == name]

    def var_list(self, name: str) -> list[str]:
        """Variables of every `name(...)` clause (private, firstprivate, ...)."""
        out: list[str] = []
        for c in self.clauses_named(name):
            out.extend(v for v in c.variables() if v not in out)
        return out

    def reductions(self) -> list[tuple[str, list[str]]]:
        return [c.reduction() for c in self.clauses_named("reduction")]

    def collapse(self) -> int:
        c = self.clause("collapse")
        if c is None:
            return 1
        if c.args is None or not c.args.strip().isdigit():
            raise DirectiveError(f"unsupported collapse clause on line {self.start_ln + 1}: {c}")
        return int(c.args)

    @property
    def is_compute_loop(self) -> bool:
        return self.construct in COMPUTE_LOOPS

    @property
    def is_compute_region(self) -> bool:
        return self.construct in COMPUTE_REGIONS

    def __str__(self) -> str:
        head = self.construct
        if self.construct_args is not None:
            head += f"({self.construct_args})"
        return " ".join([head, *(str(c) for c in self.clauses)])


def strip_comment(text: str) -> str:
    """Drops a trailing `! ...` comment (directive text has no strings)."""
    depth = 0
    for i, ch in enumerate(text):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "!" and depth == 0:
            return text[:i]
    return text


def matching_paren(text: str, open_idx: int) -> int:
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    raise DirectiveError(f"unbalanced parentheses: {text}")


def parse_clauses(text: str) -> list[Clause]:
    clauses: list[Clause] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch.isspace() or ch == ",":
            i += 1
            continue
        m = IDENT_RE.match(text, i)
        if not m:
            raise DirectiveError(f"cannot parse clauses at '{text[i:]}'")
        name = m.group(0)
        i = m.end()
        while i < len(text) and text[i].isspace():
            i += 1
        args = None
        if i < len(text) and text[i] == "(":
            close = matching_paren(text, i)
            args = text[i + 1 : close].strip()
            i = close + 1
        clauses.append(Clause(name, args))
    return clauses


def parse_directive(text: str, start_ln: int, end_ln: int, indent: str) -> AccDirective:
    """`text`: the directive with the sentinel and continuations removed."""
    body = " ".join(text.lower().split())
    for construct in CONSTRUCTS:
        if body == construct or re.match(rf"{re.escape(construct)}\b", body):
            rest = body[len(construct) :].lstrip()
            break
    else:
        raise DirectiveError(f"unknown OpenACC directive on line {start_ln + 1}: !$acc {body}")
    construct_args = None
    if rest.startswith("(") and construct in {"routine", "wait", "cache"}:
        close = matching_paren(rest, 0)
        construct_args = rest[1:close].strip()
        rest = rest[close + 1 :]
    return AccDirective(
        start_ln=start_ln,
        end_ln=end_ln,
        indent=indent,
        construct=construct,
        clauses=parse_clauses(rest),
        construct_args=construct_args,
        text=body,
    )


def find_directives(
    lines: list[str], first_ln: int = 0, last_ln: Optional[int] = None
) -> list[AccDirective]:
    """
    OpenACC directives starting on lines first_ln..last_ln (0-based, inclusive)
    of `lines`. A directive continues while its line ends with `&`;
    continuation lines start with `!$acc` (an optional leading `&` is dropped).
    """
    last_ln = len(lines) - 1 if last_ln is None else last_ln
    out: list[AccDirective] = []
    ln = first_ln
    while ln <= last_ln:
        m = SENTINEL_RE.match(lines[ln])
        if not m:
            ln += 1
            continue
        indent = m.group(1)
        start = ln
        parts: list[str] = []
        while True:
            if ln >= len(lines) or not (m := SENTINEL_RE.match(lines[ln])):
                raise DirectiveError(f"expected a `!$acc` continuation on line {ln + 1}")
            line = lines[ln]
            part = strip_comment(line[m.end() :]).strip()
            if part.startswith("&"):
                part = part[1:].strip()
            if part.endswith("&"):
                parts.append(part[:-1])
                ln += 1
                continue
            parts.append(part)
            break
        out.append(parse_directive(" ".join(parts), start, ln, indent))
        ln += 1
    return out


def next_code_line(lines: list[str], ln: int) -> Optional[int]:
    """First line after `ln` that is neither blank nor a comment/directive."""
    for i in range(ln + 1, len(lines)):
        if CODE_RE.match(lines[i]):
            return i
    return None
