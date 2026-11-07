"""
Source text helpers shared by the transforms: one implementation of the free-form Fortran line
mechanics (comments, strings, `&` continuations, indentation) the transforms used to re-implement.

Line numbers are 0-based indices into `lines`, as everywhere in SPEL. `lines` are physical lines
without their newline (`text.split("\n")`, so a trailing newline gives a final "").

These helpers only do text mechanics. What a statement *is* comes from the parser and the record
(see AGENTS.md); use these to place and format edits at the lines the analysis gives.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from spel.scripts.types import directive_sentinel


def split_code(line: str) -> tuple[str, str]:
    """(code, comment) of a line; a `!` inside a string literal is code. The comment keeps its `!`."""
    quote = None
    for i, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch == "!":
            return line[:i], line[i:]
    return line, ""


def code(line: str) -> str:
    """The line without its trailing comment (string-aware)."""
    return split_code(line)[0]


def comment(line: str) -> str:
    """The trailing comment of the line including `!`, or ''."""
    return split_code(line)[1]


def indent(line: str) -> str:
    """The leading white space of a line."""
    return line[: len(line) - len(line.lstrip())]


def continues(line: str) -> bool:
    """The line's code ends with a continuation `&` (for a directive: its text after the sentinel)."""
    if sent := directive_sentinel(line):
        return code(line.lstrip()[len(sent):]).rstrip().endswith("&")
    return code(line).rstrip().endswith("&")


def statement_end(lines: list[str], ln: int) -> int:
    """Last physical line of the statement (or directive) starting on line `ln`. Blank and comment
    lines may sit between continuation lines; a directive continues on the next directive line."""
    directive = bool(directive_sentinel(lines[ln]))
    end = ln
    while continues(lines[end]):
        i = end + 1
        if directive:
            while i < len(lines) and not directive_sentinel(lines[i]):
                i += 1
        else:
            while i < len(lines) and not code(lines[i]).strip():
                i += 1
        if i >= len(lines):
            break
        end = i
    return end


def statement_start(lines: list[str], ln: int) -> int:
    """First physical line of the (non-directive) statement that line `ln` belongs to."""
    start, i = ln, ln - 1
    while i >= 0:
        if not code(lines[i]).strip():
            i -= 1
            continue
        if not continues(lines[i]):
            break
        start = i
        i -= 1
    return start


def statement_text(lines: list[str], ln: int) -> str:
    """The logical statement starting on line `ln`: continuations joined, comments dropped."""
    end = statement_end(lines, ln)
    parts = []
    for i in range(ln, end + 1):
        c = code(lines[i]).strip()
        if not c:
            continue
        if i > ln and c.startswith("&"):
            c = c[1:].lstrip()
        if c.endswith("&"):
            c = c[:-1].rstrip()
        parts.append(c)
    return " ".join(parts)


def matching_paren(text: str, open_idx: int) -> int:
    """Index of the `)` closing the `(` at `open_idx` (string-aware); -1 if unbalanced."""
    depth, quote = 0, None
    for i in range(open_idx, len(text)):
        ch = text[i]
        if quote:
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


def split_top_level(text: str, sep: str = ",") -> list[str]:
    """`text` split at `sep` outside parentheses and strings, parts stripped."""
    parts, cur, depth, quote = [], [], 0, None
    for ch in text:
        if quote:
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch == sep and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
            continue
        cur.append(ch)
    parts.append("".join(cur).strip())
    return parts


@dataclass(frozen=True)
class Span:
    """Physical lines `start..end` (0-based, inclusive) of a file; `cols` = (first, last+1)
    character columns when the span lies on one line (an expression)."""

    path: Optional[Path]
    start: int
    end: int
    cols: Optional[tuple[int, int]] = None

    def __str__(self) -> str:
        name = self.path.name if self.path else "<source>"
        where = f"{self.start + 1}" if self.end == self.start else f"{self.start + 1}-{self.end + 1}"
        return f"{name}:{where}"


class SourceFile:
    """A file's original lines; never mutated (edits go to an EditSet)."""

    def __init__(self, path, text: str):
        self.path = Path(path) if path is not None else None
        self.text = text
        self.lines: list[str] = text.split("\n")

    @classmethod
    def read(cls, path) -> "SourceFile":
        return cls(Path(path), Path(path).read_text())

    def indent(self, ln: int) -> str:
        return indent(self.lines[ln])

    def code(self, ln: int) -> str:
        return code(self.lines[ln])

    def statement(self, ln: int) -> Span:
        """Span of the statement containing line `ln`."""
        start = statement_start(self.lines, ln)
        return Span(self.path, start, statement_end(self.lines, start))

    def span(self, start: int, end: Optional[int] = None) -> Span:
        return Span(self.path, start, start if end is None else end)

    def text_of(self, span: Span) -> list[str]:
        return self.lines[span.start:span.end + 1]


class Sources:
    """SourceFiles by path, read once per run; `put` installs edited text for later passes."""

    def __init__(self):
        self._files: dict[Path, SourceFile] = {}

    def __getitem__(self, path) -> SourceFile:
        p = Path(path)
        if p not in self._files:
            self._files[p] = SourceFile.read(p)
        return self._files[p]

    def lines(self, path) -> list[str]:
        return self[path].lines

    def put(self, path, text: str) -> SourceFile:
        self._files[Path(path)] = SourceFile(Path(path), text)
        return self._files[Path(path)]
