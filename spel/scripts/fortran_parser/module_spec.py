"""
What a module's source declares, from SPEL's parser (not from regexes over the text):

- `module_spec(path)`: per module variable its attributes (allocatable, pointer,
  parameter, target, ...), rank and accessibility; per derived type its accessibility
  and the accessibility of each component; the names in `!$acc declare` directives;
  whether the file has OpenACC compute constructs.
- `parse_modules(path)`: the parsed ModuleConstructs (preprocessed like the analysis,
  `#include "shr_assert.h"` removed, original 0-based line numbers).
- `statements_with_next(block)`: every statement with the line where the statement
  after it starts, for placing an insertion after a statement.

Directives are comments to the parser; they come from `spel.transforms.acc_directives`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

from spel.scripts.fortran_parser import spel_ast as A

class ModuleSpecError(ValueError):
    """The file does not parse (the parser's errors are in the log)."""


_PARSE_CACHE: dict[tuple[str, float], list] = {}
_SPEC_CACHE: dict[tuple[str, float], "ModuleSpec"] = {}
ACCESS = {"public", "private"}


@dataclass
class VarInfo:
    name: str
    attrs: set[str]  # lowercase attribute names: allocatable, pointer, parameter, target, ...
    rank: int
    access: str  # "public" | "private"


@dataclass
class TypeInfo:
    name: str
    access: str
    components: dict[str, str] = field(default_factory=dict)  # component -> "public" | "private"
    component_attrs: dict[str, set[str]] = field(default_factory=dict)  # component -> attributes
    component_rank: dict[str, int] = field(default_factory=dict)

    def storage_test(self, component: str) -> Optional[str]:
        """`associated`/`allocated` for a pointer/allocatable component, None if inline."""
        attrs = self.component_attrs.get(component.lower(), set())
        return "associated" if "pointer" in attrs else ("allocated" if "allocatable" in attrs else None)


@dataclass
class ModuleSpec:
    name: str
    variables: dict[str, VarInfo] = field(default_factory=dict)
    types: dict[str, TypeInfo] = field(default_factory=dict)
    declared: set[str] = field(default_factory=set)  # names in `!$acc declare` directives
    compute_constructs: bool = False  # `!$acc parallel/kernels/serial` anywhere in the file
    contains_ln: Optional[int] = None  # module `contains`, or None
    end_ln: Optional[int] = None  # `end module`
    routines: dict[str, tuple[int, int]] = field(default_factory=dict)  # module procedure -> (first, last line)
    compute_routines: set[str] = field(default_factory=set)  # procedures with compute constructs

    def storage_test(self, variable: str) -> Optional[str]:
        v = self.variables.get(variable.lower())
        if v is None:
            return None
        return "associated" if "pointer" in v.attrs else ("allocated" if "allocatable" in v.attrs else None)


def _key(path: Path) -> tuple[str, float]:
    return (str(path.resolve()), path.stat().st_mtime)


def parse_modules(path: Path) -> list:
    """The ModuleConstructs of a file, parsed as the analysis parses sources."""
    from spel.scripts.edit_files import apply_preprocessor, remove_cpp_directives
    from spel.scripts.fortran_parser.spel_parser import Parser

    path = Path(path)
    key = _key(path)
    if key not in _PARSE_CACHE:
        lines = remove_cpp_directives(apply_preprocessor(path), str(path), logging.getLogger("module_spec"))
        # remove_cpp_directives keeps a stand-in `#include 'shr_assert.h'` line (for the
        # unit-test copies); it holds no Fortran
        lines = [lt for lt in lines if not lt.line.lstrip().startswith("#")]
        try:
            prog = Parser(lines=lines).parse_program()
        except SystemExit as exc:  # the parser exits on errors; callers decide what to do
            raise ModuleSpecError(f"{path}: does not parse") from exc
        _PARSE_CACHE[key] = [s for s in prog.statements if isinstance(s, A.ModuleConstruct)]
    return _PARSE_CACHE[key]


def _attr_names(spec: Optional[A.AttributeSpec]) -> set[str]:
    out: set[str] = set()
    for a in spec.attrs if spec is not None else []:
        if isinstance(a, A.Identifier):
            out.add(a.value.lower())
        elif isinstance(a, A.FuncExpression):  # dimension(...), intent(...)
            out.add(str(a.function).lower())
    return out


def _entity_name(e: A.EntityDecl) -> str:
    return e.token.literal.lower()


def _rank(attrs: Optional[A.AttributeSpec], e: A.EntityDecl) -> int:
    if e.bounds:
        return len(e.bounds)
    return max(attrs.get_dim(), 0) if attrs is not None else 0  # get_dim: -1 without dimension()


def _access_lists(statements) -> tuple[str, dict[str, str]]:
    """(default access, {name: access}) from the access statements of a scoping unit."""
    default, named = "public", {}
    for st in statements:
        if isinstance(st, A.AccessStatement):
            if st.names:
                for n in st.names:
                    named[str(n).lower()] = st.access.lower()
            else:
                default = st.access.lower()
    return default, named


def _type_info(td: A.TypeDef, access: str) -> TypeInfo:
    info = TypeInfo(td.name.lower(), access)
    members = td.members.statements if td.members is not None else []
    # a bare `private` in the type definition makes the components private by default
    default = "private" if any(isinstance(s, A.AccessStatement) and not s.names and s.access.lower() == "private"
                               for s in members) else "public"
    for st in members:
        if isinstance(st, A.VariableDecl):
            attrs = _attr_names(st.attrs)
            acc = next((a for a in attrs if a in ACCESS), default)
            for e in st.entities:
                n = _entity_name(e)
                info.components[n] = acc
                info.component_attrs[n] = attrs - ACCESS
                info.component_rank[n] = _rank(st.attrs, e)
    return info


def module_spec(path: Path) -> ModuleSpec:
    """Variables, types and acc declares of the (first) module in `path`."""
    from spel.transforms.acc_directives import find_directives

    path = Path(path)
    key = _key(path)
    if key in _SPEC_CACHE:
        return _SPEC_CACHE[key]
    mods = parse_modules(path)
    if not mods:
        raise ModuleSpecError(f"{path}: no module")
    mod = mods[0]
    statements = mod.spec.statements
    default, named = _access_lists(statements)
    spec = ModuleSpec(mod.name.lower(), contains_ln=mod.contains_ln, end_ln=mod.end_ln)
    for st in statements:
        if isinstance(st, A.VariableDecl):
            attrs = _attr_names(st.attrs)
            for e in st.entities:
                n = _entity_name(e)
                acc = next((a for a in attrs if a in ACCESS), named.get(n, default))
                spec.variables[n] = VarInfo(n, attrs - ACCESS, _rank(st.attrs, e), acc)
        elif isinstance(st, A.TypeDef):
            attrs = _attr_names(st.attrs)
            n = st.name.lower()
            acc = next((a for a in attrs if a in ACCESS), named.get(n, default))
            spec.types[n] = _type_info(st, acc)
    for sub in mod.subprograms:
        spec.routines[sub.name.lower()] = (sub.lineno, sub.end_ln)
    for d in find_directives(path.read_text(errors="replace").split("\n")):
        if d.construct == "declare":
            for c in d.clauses:
                spec.declared.update(v.lower() for v in c.variables())
        elif d.is_compute_region or d.is_compute_loop:
            spec.compute_constructs = True
            for name, (a, b) in spec.routines.items():
                if a <= d.start_ln <= b:
                    spec.compute_routines.add(name)
    _SPEC_CACHE[key] = spec
    return spec


def statements_with_next(block, end_ln: Optional[int] = None) -> Iterator[tuple[object, int]]:
    """
    Every statement of `block` (recursively) with the line at which an insertion after
    it goes: the next statement of the same block, else the end of the enclosing
    construct (`end_ln`).
    """
    stmts = list(block.statements)
    for i, st in enumerate(stmts):
        nxt = stmts[i + 1].lineno if i + 1 < len(stmts) else end_ln
        if isinstance(st, A.IfConstruct) or getattr(st, "end_ln", None) is not None:
            yield st, st.end_ln + 1
        else:
            yield st, nxt
        for child, child_end in _child_blocks(st):
            yield from statements_with_next(child, child_end)


def _child_blocks(st) -> list[tuple[object, Optional[int]]]:
    if isinstance(st, A.IfConstruct):
        out = [(st.consequence, None)]
        out += [(e.consequence, None) for e in st.else_ifs]
        if st.else_ is not None:
            out.append((st.else_.alternative, None))
        return [(b, st.end_ln) for b, _ in out]
    body = getattr(st, "body", None)
    if body is not None and hasattr(body, "statements"):
        return [(body, getattr(st, "end_ln", None))]
    return []


def callee_name(stmt) -> Optional[str]:
    """The called routine's name (last part: `x%update` -> 'update') of a call statement."""
    from spel.scripts.registrations import _designator_parts

    if not isinstance(stmt, A.SubCallStatement):
        return None
    parts = _designator_parts(stmt.function)
    if parts is None:
        return None
    base, rest = parts
    return (rest.split("%")[-1] if rest else base).lower()


def first_string_arg(stmt) -> Optional[str]:
    args = getattr(stmt.function, "args", None) or []
    if args and isinstance(args[0], A.StringLiteral):
        return str(args[0].value).lower()
    return None


def contains_call(stmt, name: str) -> bool:
    """Whether a construct's body (any depth) calls `name`."""
    for child, _ in _child_blocks(stmt):
        for st, _ in statements_with_next(child):
            if callee_name(st) == name:
                return True
    return False
