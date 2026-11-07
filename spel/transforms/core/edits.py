"""
Line edits: the one edit model of the transforms.

An `Edit` replaces `n` original lines starting at line `ln` (0-based) with `new` lines; `n = 0`
is an insertion before line `ln` (`ln = len(lines)` appends). Edits are positioned against the
*original* lines, so a transform never tracks line shifts. An `EditSet` holds the edits of one
file and applies them all at once:

- replaced ranges must not overlap, and nothing may be inserted strictly inside a replaced range
  (`EditConflict`): two passes touching the same lines is an error, not an ordering accident;
- edits at the same line come out in `order` (lower first, i.e. higher up), then in the order
  they were added; a replacement's new lines are placed among the insertions at its line by
  its own `order`;
- `expect` (the original lines an edit replaces) is checked (`EditMismatch`), so an edit planned
  on other text than it is applied to fails instead of corrupting the file.

A `TextEdit` replaces columns `[c0, c1)` of one line (an expression the parser located with
`keep_spans`, see `span_text`/`replace_span`). Text edits on one line must not overlap (identical
ones merge; insertions at one column keep the order they were added in) and must not touch a
line that a line edit replaces; they are applied before the line edits.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence


class EditConflict(ValueError):
    pass


class EditMismatch(ValueError):
    pass


@dataclass(frozen=True)
class Edit:
    ln: int
    n: int
    new: tuple[str, ...]
    order: float = 0
    expect: Optional[tuple[str, ...]] = None
    why: str = ""

    @property
    def end(self) -> int:
        """One past the last replaced line."""
        return self.ln + self.n


@dataclass(frozen=True)
class TextEdit:
    ln: int
    c0: int
    c1: int
    new: str
    expect: Optional[str] = None
    why: str = ""


def replace_cols(ln: int, c0: int, c1: int, new: str, expect: Optional[str] = None, why: str = "") -> TextEdit:
    """Replace columns `[c0, c1)` of line `ln` by `new` (`c0 == c1` inserts)."""
    if c1 < c0:
        raise ValueError(f"bad columns {c0}:{c1}")
    return TextEdit(ln, c0, c1, new, expect, why)


def replace_span(span, new: str, lines: Optional[Sequence[str]] = None, why: str = "") -> TextEdit:
    """Replace the text of a parser span `(ln0, c0, ln1, c1)` (one line only) by `new`; with
    `lines`, the current text is recorded as `expect`."""
    ln0, c0, ln1, c1 = span
    if ln0 != ln1:
        raise ValueError(f"span {span} covers several lines")
    return replace_cols(ln0, c0, c1, new, expect=lines[ln0][c0:c1] if lines is not None else None, why=why)


def span_text(lines: Sequence[str], span) -> str:
    """The source text of a parser span `(ln0, c0, ln1, c1)` (lines joined with newlines)."""
    ln0, c0, ln1, c1 = span
    if ln0 == ln1:
        return lines[ln0][c0:c1]
    return "\n".join([lines[ln0][c0:], *lines[ln0 + 1:ln1], lines[ln1][:c1]])


def insert(ln: int, new: Sequence[str] | str, order: float = 0, why: str = "") -> Edit:
    """Insert `new` before line `ln`."""
    return Edit(ln, 0, _lines(new), order, None, why)


def replace(ln: int, new: Sequence[str] | str, n: int = 1, order: float = 0,
            expect: Optional[Sequence[str]] = None, why: str = "") -> Edit:
    """Replace `n` lines from `ln` by `new`."""
    return Edit(ln, n, _lines(new), order, tuple(expect) if expect is not None else None, why)


def delete(ln: int, n: int = 1, why: str = "") -> Edit:
    return Edit(ln, n, (), 0, None, why)


def stacked(items: Iterable[tuple[int, Sequence[str], float]]) -> list[Edit]:
    """Insertions `(ln, lines, order)` the way the older passes spliced them (bottom-up, each at
    its line): of insertions at one line with the same order, a later one ends up above."""
    return [insert(ln, new, order=order - k * 1e-6) for k, (ln, new, order) in enumerate(items)]


def changed_lines(old: Sequence[str], new: Sequence[str], order: float = 1e9) -> list[Edit]:
    """Replacements turning `old` into `new` line by line (same length); `order` places them
    after (below) insertions at the same line."""
    if len(old) != len(new):
        raise ValueError("changed_lines needs texts of the same length")
    return [replace(ln, b, order=order, expect=[a]) for ln, (a, b) in enumerate(zip(old, new)) if a != b]


def diff_edits(old: Sequence[str], new: Sequence[str], why: str = "") -> list[Edit]:
    """Edits turning `old` into `new`, from a line diff: for passes that rebuild a file whole
    (a module losing most of its contents), so their output still goes through an EditSet."""
    out = []
    sm = difflib.SequenceMatcher(None, list(old), list(new), autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "insert":
            out.append(insert(i1, new[j1:j2], why=why))
        elif op != "equal":
            out.append(replace(i1, new[j1:j2], n=i2 - i1, expect=old[i1:i2], why=why))
    return out


def _lines(new) -> tuple[str, ...]:
    return (new,) if isinstance(new, str) else tuple(new)


@dataclass
class EditSet:
    """The edits of one file."""

    path: Optional[Path] = None
    edits: list[Edit] = field(default_factory=list)
    text_edits: list[TextEdit] = field(default_factory=list)

    def add(self, *edits: Edit | TextEdit) -> "EditSet":
        for e in edits:
            if isinstance(e, TextEdit):
                self._add_text(e)
                continue
            if e.n < 0 or e.ln < 0:
                raise ValueError(f"bad edit {e}")
            for o in self.edits:
                if e.n and o.n and e.ln < o.end and o.ln < e.end and e != o:
                    raise EditConflict(f"{self._where(e)} overlaps {self._where(o)}")
                if (not e.n and o.ln < e.ln < o.end) or (not o.n and e.ln < o.ln < e.end):
                    raise EditConflict(f"insertion inside a replaced range: {self._where(e)} / {self._where(o)}")
            for t in self.text_edits:
                if e.ln <= t.ln < e.end:
                    raise EditConflict(f"{self._where(e)} replaces a line with a text edit ({self._where(t)})")
            if e.n and e in self.edits:
                continue  # the same replacement planned twice
            self.edits.append(e)
        return self

    def _add_text(self, t: TextEdit) -> None:
        if t in self.text_edits:
            return
        for o in self.text_edits:
            if o.ln == t.ln and t.c0 < o.c1 and o.c0 < t.c1:
                raise EditConflict(f"{self._where(t)} overlaps {self._where(o)}")
            if o.ln == t.ln and t.c0 < t.c1 and o.c0 == o.c1 and t.c0 < o.c0 < t.c1:
                raise EditConflict(f"insertion inside a replaced text: {self._where(t)} / {self._where(o)}")
            if o.ln == t.ln and o.c0 < o.c1 and t.c0 == t.c1 and o.c0 < t.c0 < o.c1:
                raise EditConflict(f"insertion inside a replaced text: {self._where(t)} / {self._where(o)}")
        for e in self.edits:
            if e.ln <= t.ln < e.end:
                raise EditConflict(f"{self._where(t)} is on a line {self._where(e)} replaces")
        self.text_edits.append(t)

    def extend(self, edits: Iterable[Edit | TextEdit]) -> "EditSet":
        return self.add(*edits)

    def __bool__(self) -> bool:
        return bool(self.edits or self.text_edits)

    def __len__(self) -> int:
        return len(self.edits) + len(self.text_edits)

    def _where(self, e: Edit | TextEdit) -> str:
        name = self.path.name if self.path else "<source>"
        if isinstance(e, TextEdit):
            return f"{name}: line {e.ln + 1} columns {e.c0 + 1}-{e.c1}" + (f" ({e.why})" if e.why else "")
        what = f"lines {e.ln + 1}-{e.end}" if e.n else f"insertion at line {e.ln + 1}"
        return f"{name}: {what}" + (f" ({e.why})" if e.why else "")

    def apply_text_edits(self, lines: Sequence[str]) -> list[str]:
        """`lines` with only the text edits applied."""
        out = list(lines)
        by_line: dict[int, list[tuple[int, TextEdit]]] = {}
        for k, t in enumerate(self.text_edits):
            if t.ln >= len(lines) or t.c1 > len(lines[t.ln]):
                raise EditMismatch(f"{self._where(t)} beyond the end of the line")
            if t.expect is not None and lines[t.ln][t.c0:t.c1] != t.expect:
                raise EditMismatch(f"{self._where(t)}: the source differs from the planned text")
            by_line.setdefault(t.ln, []).append((k, t))
        for ln, items in by_line.items():
            line = out[ln]
            # right to left; at one column, a replacement's start after the insertions there
            for k, t in sorted(items, key=lambda kt: (kt[1].c0, kt[1].c1 > kt[1].c0, kt[0]), reverse=True):
                line = line[:t.c0] + t.new + line[t.c1:]
            out[ln] = line
        return out

    def apply(self, lines: Sequence[str]) -> list[str]:
        """The edited lines (the input is not modified)."""
        if self.text_edits:
            lines = self.apply_text_edits(lines)
        at: dict[int, list[tuple[float, int, Edit]]] = {}
        for k, e in enumerate(self.edits):
            if e.end > len(lines) or e.ln > len(lines):
                raise EditMismatch(f"{self._where(e)} beyond the end of the file ({len(lines)} lines)")
            if e.expect is not None and tuple(lines[e.ln:e.end]) != e.expect:
                raise EditMismatch(f"{self._where(e)}: the source differs from the planned text")
            at.setdefault(e.ln, []).append((e.order, k, e))
        out: list[str] = []
        i = 0
        while i <= len(lines):
            skip = 0
            for _, _, e in sorted(at.get(i, ()), key=lambda t: (t[0], t[1])):
                out.extend(e.new)
                skip = max(skip, e.n)
            if skip:
                i += skip
                continue
            if i < len(lines):
                out.append(lines[i])
            i += 1
        return out

    def apply_text(self, text: str) -> str:
        return "\n".join(self.apply(text.split("\n")))

    def diff(self, lines: Sequence[str], context: int = 3) -> str:
        name = str(self.path).lstrip("/") if self.path else "source"
        new = self.apply(lines)
        return "".join(difflib.unified_diff([l + "\n" for l in lines], [l + "\n" for l in new],
                                            f"a/{name}", f"b/{name}", n=context))
