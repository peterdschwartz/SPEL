"""
Resolve preprocessor conditionals on macros whose value is settled (`spel transform cpp-resolve`).

ELM always builds with MODAL_AER (user decision, 2026-11-03): the `#ifndef MODAL_AER` code and the
`#else` branches of `#ifdef MODAL_AER` are dead, and two code paths per routine cost every port
(OpenACC directives, analysis, review). This pass applies such conditionals in the source:

    #ifdef MODAL_AER          ->  the first branch's lines, without the directives
       a                       ->  a
    #else
       b
    #endif

`--define NAME` takes NAME as defined, `--undefine NAME` as undefined (default: --define MODAL_AER).
Only `#ifdef NAME`, `#ifndef NAME`, `#if defined(NAME)` and `#if !defined(NAME)` on those names are
resolved, with their `#else`/`#endif`; every other conditional (nested or around) stays as it is.
A file with an `#elif` in a resolved conditional, or an `#if` expression combining a resolved macro
with anything else, is refused. Pure preprocessor-text work: no Fortran is classified.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from spel.transforms.core.edits import delete
from spel.transforms.core.registry import transform
from spel.transforms.core.transform import Change, Plan, Refusal, Transform

DIRECTIVE = re.compile(r"^\s*#\s*(ifdef|ifndef|if|elif|else|endif)\b(.*)$")


class CppError(ValueError):
    pass


def _condition(kind: str, rest: str, values: dict[str, bool]):
    """True/False for a conditional this pass resolves, None for one it leaves alone."""
    rest = rest.split("//")[0].split("/*")[0].strip()
    if kind in ("ifdef", "ifndef"):
        name = rest.split()[0] if rest.split() else ""
        if name in values:
            return values[name] if kind == "ifdef" else not values[name]
        return None
    m = re.fullmatch(r"(!?)\s*defined\s*\(?\s*(\w+)\s*\)?", rest)
    if m and m.group(2) in values:
        return values[m.group(2)] != bool(m.group(1))
    if any(re.search(rf"\b{re.escape(n)}\b", rest) for n in values):
        raise CppError(f"`#if {rest}` combines a resolved macro with something else")
    return None


def resolve_lines(lines: list[str], values: dict[str, bool]) -> list[int]:
    """0-based lines to delete."""
    drop: list[int] = []
    stack: list[dict] = []  # {"resolved": bool, "keep_first": bool, "in_else": bool}

    def dead() -> bool:
        return any(f["resolved"] and (f["keep_first"] == f["in_else"]) for f in stack)

    for i, line in enumerate(lines):
        m = DIRECTIVE.match(line)
        if m is None:
            if dead():
                drop.append(i)
            continue
        kind, rest = m.group(1), m.group(2)
        if kind in ("ifdef", "ifndef", "if"):
            outer_dead = dead()
            cond = _condition(kind, rest, values)
            if cond is None:
                stack.append({"resolved": False, "keep_first": True, "in_else": False})
                if outer_dead:
                    drop.append(i)
            else:
                stack.append({"resolved": True, "keep_first": cond, "in_else": False})
                drop.append(i)
        elif kind == "elif":
            if stack and stack[-1]["resolved"]:
                raise CppError(f"line {i + 1}: #elif in a resolved conditional")
            if dead():
                drop.append(i)
        elif kind == "else":
            if not stack:
                raise CppError(f"line {i + 1}: #else without #if")
            top = stack[-1]
            if top["resolved"]:
                top["in_else"] = True
                drop.append(i)
            elif dead():
                drop.append(i)
        else:  # endif
            if not stack:
                raise CppError(f"line {i + 1}: #endif without #if")
            top = stack.pop()
            if top["resolved"] or dead():
                drop.append(i)
    if stack:
        raise CppError("unterminated conditional")
    return drop


@dataclass
class Options:
    define: list[str] = field(default_factory=lambda: ["MODAL_AER"])  # macros taken as defined
    undefine: list[str] = field(default_factory=list)  # macros taken as undefined
    files: list[str] = field(default_factory=list)  # default: every ELM source file


@transform
class CppResolve(Transform):
    """
    Matches: `#ifdef/#ifndef/#if [!]defined(NAME)` conditionals on the macros of --define/--undefine.
    Changes: keeps the branch that the macro's value selects, without the directive lines; drops the
             other branch. Other conditionals stay.
    Refuses: files where a resolved conditional has an #elif, or an #if mixes a resolved macro with
             other terms.
    """

    name = "cpp-resolve"
    summary = "apply preprocessor conditionals on settled macros (default: MODAL_AER always defined)"
    Options = Options

    def build(self, ctx, opts) -> Plan:
        out = Plan(self.name)
        values = {n: True for n in opts.define} | {n: False for n in opts.undefine}
        elm = ctx.srcroot / "components" / "elm" / "src"
        if opts.files:
            files = [Path(f) if Path(f).is_absolute() else ctx.srcroot / f for f in opts.files]
        elif elm.is_dir():
            files = [f for f in sorted(elm.rglob("*.F90")) if "external_models" not in f.parts]
        else:
            files = sorted({ctx.path(s) for s in ctx.fut.subroutine_dict.values()})
        for path in files:
            lines = ctx.lines(path)
            if not any(n in l for l in lines for n in values):
                continue
            try:
                drop = resolve_lines(lines, values)
            except CppError as err:
                out.add(Refusal(path.name, str(err)))
                continue
            if drop:
                out.add(Change(path, [delete(i, why="cpp-resolve") for i in drop], where=path.name,
                               why=f"{len(drop)} lines ({', '.join(sorted(values))})"))
        return out
