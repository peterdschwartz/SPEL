"""A small TOML writer for spec-shaped data (no third-party writer is installed).

What specs hold: tables, arrays of tables, inline tables (rules, the gridcell hierarchy),
arrays, strings (long Fortran expressions as multi-line strings), numbers and booleans. A dict
is a `[section]` down to three levels (`[scenario.base.inputs]`) and a list of dicts an array of
tables at the top two (`[[property]]`, `[[domain.gridcell]]`); anything deeper or inside an
array of tables is inline (rules, landunits, `follow`). Keys
that are not bare (`"lun_pp%ht_roof"`) are quoted. `comments` puts `# ...` lines above a key or
a table, addressed by its path: `("scenario", "base", "inputs", "lun_pp%ht_roof")`, or
`("domain", "gridcell", 0)` for one table of an array. `tomllib.loads(dumps(d)) == d`.
"""

from __future__ import annotations

import math
import re
from typing import Any, Optional

Path = tuple[Any, ...]
_BARE = re.compile(r"^[A-Za-z0-9_-]+$")


def dumps(data: dict, comments: Optional[dict[Path, str]] = None, width: int = 100) -> str:
    """TOML of `data`. Inline tables and arrays longer than `width` are written one element per
    line."""
    w = _Writer(comments or {}, width)
    w.table((), data, header=False)
    return "\n".join(w.lines).strip("\n") + "\n"


def key(k: str) -> str:
    return k if _BARE.match(k) else _string(k)


class _Writer:
    def __init__(self, comments: dict[Path, str], width: int):
        self.comments = comments
        self.width = width
        self.lines: list[str] = []

    def comment(self, path: Path) -> None:
        text = self.comments.get(path)
        if text:
            self.lines += [f"# {line}".rstrip() for line in text.splitlines()]

    def table(self, path: Path, data: dict, header: bool = True, array: bool = False) -> None:
        free = not any(isinstance(p, int) for p in path)  # not inside an array of tables
        tables = {k: v for k, v in data.items() if free and len(path) < 3 and isinstance(v, dict)}
        arrays = {k: v for k, v in data.items() if free and len(path) < 2 and _is_table_array(v)}
        scalars = {k: v for k, v in data.items() if k not in tables and k not in arrays}
        if header and (scalars or array or not (tables or arrays)):
            if self.lines:
                self.lines.append("")
            self.comment(path)
            name = ".".join(key(str(p)) for p in path if not isinstance(p, int))
            self.lines.append(f"[[{name}]]" if array else f"[{name}]")
        for k, v in scalars.items():
            self.comment(path + (k,))
            self.lines += self.assignment(k, v)
        for k, v in tables.items():
            self.table(path + (k,), v)
        for k, items in arrays.items():
            for n, item in enumerate(items):
                self.table(path + (k, n), item, array=True)

    def assignment(self, k: str, v: Any) -> list[str]:
        head = f"{key(k)} = "
        text = _inline(v)
        if len(head) + len(text) <= self.width or not isinstance(v, (list, dict)) or not v:
            if isinstance(v, str) and len(head) + len(text) > self.width:
                return [head + _multiline(v)]
            return [head + text]
        return [head + line for line in [_broken(v, self.width)]]


def _is_table_array(v: Any) -> bool:
    return isinstance(v, list) and bool(v) and all(isinstance(x, dict) for x in v)


def _inline(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return repr(v)
    if isinstance(v, str):
        return _string(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_inline(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{ " + ", ".join(f"{key(str(k))} = {_inline(x)}" for k, x in v.items()) + " }" if v else "{}"
    raise TypeError(f"cannot write {type(v).__name__} {v!r} as TOML")


def _broken(v: Any, width: int, indent: str = "") -> str:
    """An array or inline table one element per line (TOML 1.0 allows newlines in arrays only;
    a long inline table stays on one line inside its array)."""
    if isinstance(v, (list, tuple)):
        inner = indent + "  "
        lines = []
        for x in v:
            text = _inline(x)
            if isinstance(x, (list, tuple)) and len(inner) + len(text) > width:
                text = _broken(x, width, inner)
            lines.append(f"{inner}{text},")
        return "[\n" + "\n".join(lines) + f"\n{indent}]"
    return _inline(v)


def _string(s: str) -> str:
    out = s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\t", "\\t")
    return f'"{out}"'


def _multiline(s: str) -> str:
    """A long expression as a multi-line basic string (a line break is a blank to the parser)."""
    if '"""' in s or "\\" in s:
        return _string(s)
    return '"""' + s + '"""'
