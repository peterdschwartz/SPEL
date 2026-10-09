"""
ModuleResolver: SymbolResolver backed by SPEL's module data.

Names visible at module scope (in priority order):
    own declarations: global_vars, defined_types, module procedures,
                      generic interfaces and interface-body procedures
    use-associated names from module-head `use` statements
      (only the public names of the used module, transitively)

Modules SPEL can't load (bad_modules, missing files, libraries, intrinsic
modules) are "unavailable": their only-list names are EXTERNAL, and a whole
module `use` makes any otherwise undeclared name EXTERNAL via `fallback`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Optional

from spel.scripts.fortran_parser.symbols import (
    Origin,
    Symbol,
    SymbolKind,
    apply_use,
)
from spel.scripts.types import LineTuple

if TYPE_CHECKING:
    from spel.scripts.analyze_subroutines import Subroutine
    from spel.scripts.fortran_modules import FortranModule
    from spel.scripts.fortran_parser.spel_ast import UseStatement


@dataclass
class ModuleHead:
    """
    default_private: module has a bare `private` statement
    access:          explicit accessibility, name -> is_public
    protected:       names with the `protected` attribute
    stmt_ln:         name -> line of the `public/private/protected :: name`
                     statement(s) naming it (not declarations)
    generics:        generic interface names
    interface_procs: procedures declared by interface bodies (abstract or external)
    """

    default_private: bool = False
    access: dict[str, bool] = field(default_factory=dict)
    protected: set[str] = field(default_factory=set)
    stmt_ln: dict[str, list[int]] = field(default_factory=dict)
    generics: set[str] = field(default_factory=set)
    interface_procs: set[str] = field(default_factory=set)

    def is_public(self, name: str) -> bool:
        return self.access.get(name, not self.default_private)

    def is_protected(self, name: str) -> bool:
        return name in self.protected


IDENT = r"[a-z_]\w*"
regex_access_stmt = re.compile(r"^(public|private|protected)\b\s*(?:::)?\s*(.*)$")
regex_type_start = re.compile(
    rf"^type\b(?!\s*\()\s*(?:,(?P<attrs>[^:]*))?(?:::)?\s*(?P<name>{IDENT})"
)
regex_end_type = re.compile(r"^end\s*type\b")
regex_interface = re.compile(rf"^(?P<abstract>abstract\s+)?interface\b\s*(?P<name>{IDENT})?")
regex_end_interface = re.compile(r"^end\s*interface\b")
regex_body_start = re.compile(rf"^(?!end\b).*?\b(?:subroutine|function)\s+(?P<name>{IDENT})")
regex_end_body = re.compile(r"^end\b")
regex_access_attr = re.compile(r",\s*(public|private)\b")
regex_protected_attr = re.compile(r",\s*protected\b")


def split_top_level(text: str) -> list[str]:
    """Split on commas not nested in parentheses/brackets."""
    items, depth, start = [], 0, 0
    for i, ch in enumerate(text):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch == "," and depth == 0:
            items.append(text[start:i])
            start = i + 1
    items.append(text[start:])
    return [s.strip() for s in items if s.strip()]


def entity_names(text: str) -> list[str]:
    names = []
    for item in split_top_level(text):
        if item.startswith(("operator", "assignment")):
            continue
        m = re.match(IDENT, item)
        if m:
            names.append(m.group())
    return names


def scan_module_head(lines: Iterable[LineTuple]) -> ModuleHead:
    """
    Accessibility and interface names from the (logical, lowercased) lines of
    a module's specification part. Type and interface bodies are skipped.
    """
    head = ModuleHead()
    in_type = False
    in_interface = False
    in_body = False
    for lt in lines:
        line = lt.line.strip().lower()
        if not line:
            continue
        if in_type:
            in_type = not regex_end_type.match(line)
            continue
        if in_interface:
            if in_body:
                in_body = not regex_end_body.match(line)
            elif regex_end_interface.match(line):
                in_interface = False
            elif m := regex_body_start.match(line):
                head.interface_procs.add(m.group("name"))
                in_body = True
            continue

        if m := regex_interface.match(line):
            in_interface = True
            name = m.group("name")
            if name and not m.group("abstract") and name not in ("operator", "assignment"):
                head.generics.add(name)
            continue
        if m := regex_type_start.match(line):
            in_type = True
            attrs = m.group("attrs") or ""
            if access := re.search(r"\b(public|private)\b", attrs):
                head.access[m.group("name")] = access.group(1) == "public"
            continue
        if m := regex_access_stmt.match(line):
            names = entity_names(m.group(2))
            for name in names:
                head.stmt_ln.setdefault(name, []).append(lt.ln)
            if m.group(1) == "protected":
                head.protected.update(names)
                continue
            is_public = m.group(1) == "public"
            if not m.group(2).strip():
                head.default_private = not is_public
            for name in names:
                head.access[name] = is_public
            continue
        if "::" in line:
            spec, _, ents = line.partition("::")
            access = regex_access_attr.search(spec)
            protected = regex_protected_attr.search(spec)
            if access or protected:
                for name in entity_names(ents):
                    if access:
                        head.access[name] = access.group(1) == "public"
                    if protected:
                        head.protected.add(name)
    return head


def module_head(fort_mod: FortranModule) -> ModuleHead:
    return scan_module_head(
        lt for lt in fort_mod.module_lines if lt.ln < fort_mod.end_of_head_ln
    )


@dataclass
class _Exports:
    symbols: dict[str, Symbol]
    external_fallback: bool  # re-exports a whole-use of an unavailable module


class ModuleResolver:
    """
    mod_name:    module containing the routine being walked
    line_ranges: source ranges of the routine (and its hosts). `use`
                 statements of unavailable modules inside them were commented
                 out before parsing, so they are recovered from fort_mod.use_stmts.
    scopes:      module-level cache; share one across routines of a unit test
    """

    def __init__(
        self,
        mod_name: str,
        mod_dict: dict[str, FortranModule],
        sub_dict: dict[str, Subroutine],
        line_ranges: Optional[list[tuple[int, int]]] = None,
        scopes: Optional[ModuleScopes] = None,
    ):
        self.mod_name = mod_name
        self.scopes = scopes if scopes is not None else ModuleScopes(mod_dict, sub_dict)

        scope = self.scopes.visible(mod_name)
        self.symbols: dict[str, Symbol] = dict(scope.symbols)
        self.external_fallback: bool = scope.external_fallback
        for stmt in self.routine_uses(line_ranges or []):
            if self.scopes.is_available(stmt.module):
                continue  # still in the routine's AST -> use_module
            self.symbols.update(self.use_module(stmt))

    @classmethod
    def for_subroutine(
        cls,
        sub: Subroutine,
        mod_dict: dict[str, FortranModule],
        sub_dict: dict[str, Subroutine],
        scopes: Optional[ModuleScopes] = None,
    ) -> ModuleResolver:
        """Resolver for `sub`. Internal subprograms also see their hosts' uses."""
        family = [sub]
        while family[-1].is_internal and family[-1].host is not None:
            family.append(family[-1].host)
        ranges = [(s.startline, s.endline) for s in family]
        return cls(sub.module, mod_dict, sub_dict, line_ranges=ranges, scopes=scopes)

    # --- SymbolResolver ---
    def lookup(self, name: str) -> Optional[Symbol]:
        return self.symbols.get(name)

    def fallback(self, name: str) -> Optional[Symbol]:
        if self.external_fallback:
            return external(name)
        return None

    def use_module(self, stmt: UseStatement) -> dict[str, Symbol]:
        visible, whole_external = self.scopes.use_symbols(stmt)
        if whole_external:
            self.external_fallback = True
        return visible

    def routine_uses(self, ranges: list[tuple[int, int]]) -> list[UseStatement]:
        fort_mod = self.scopes.mod_dict[self.mod_name]
        return [
            s
            for s in fort_mod.use_stmts
            if s.lineno >= fort_mod.end_of_head_ln
            and any(start <= s.lineno <= end for start, end in ranges)
        ]


class ModuleScopes:
    """Names visible in / exported by each module, cached."""

    def __init__(
        self,
        mod_dict: dict[str, FortranModule],
        sub_dict: dict[str, Subroutine],
    ):
        self.mod_dict = mod_dict
        self.sub_dict = sub_dict
        self._heads: dict[str, ModuleHead] = {}
        self._visible: dict[str, _Exports] = {}
        self._exports: dict[str, _Exports] = {}
        self._in_progress: set[str] = set()

    def is_available(self, mod_name: str) -> bool:
        return mod_name in self.mod_dict

    def head(self, mod_name: str) -> ModuleHead:
        if mod_name not in self._heads:
            self._heads[mod_name] = module_head(self.mod_dict[mod_name])
        return self._heads[mod_name]

    def head_uses(self, mod_name: str) -> list[UseStatement]:
        fort_mod = self.mod_dict[mod_name]
        return [s for s in fort_mod.use_stmts if s.lineno < fort_mod.end_of_head_ln]

    def own_symbols(self, mod_name: str) -> dict[str, Symbol]:
        fort_mod = self.mod_dict[mod_name]
        head = self.head(mod_name)
        syms: dict[str, Symbol] = {}
        for name in set(head.generics) | head.interface_procs:
            syms[name] = Symbol(name, SymbolKind.PROCEDURE, Origin.GLOBAL)
        for sub_id in fort_mod.subroutines:
            sub = self.sub_dict.get(sub_id)
            name = sub_id.split("::")[-1]
            if (sub is None or not sub.is_internal) and name not in head.generics:
                syms[name] = Symbol(
                    name, SymbolKind.PROCEDURE, Origin.GLOBAL, target=sub_id
                )
        for name in fort_mod.defined_types:
            syms[name] = Symbol(name, SymbolKind.DERIVED_TYPE, Origin.GLOBAL)
        for name in fort_mod.global_vars:
            syms[name] = Symbol(name, SymbolKind.VARIABLE, Origin.GLOBAL)
        return syms

    def use_symbols(self, stmt: UseStatement) -> tuple[dict[str, Symbol], bool]:
        """(names made visible by stmt, whether it is a whole-use of unavailable names)"""
        if not self.is_available(stmt.module):
            return apply_use(stmt, {}, missing=external), not stmt.has_only
        exports = self.exports(stmt.module)
        missing = external if exports.external_fallback else None
        visible = apply_use(stmt, exports.symbols, missing=missing)
        return visible, exports.external_fallback and not stmt.has_only

    def visible(self, mod_name: str) -> _Exports:
        """All names visible in the specification part of `mod_name`."""
        if mod_name in self._visible:
            return self._visible[mod_name]
        if mod_name in self._in_progress:
            # circular use: only the module's own names are known
            return _Exports(self.own_symbols(mod_name), False)
        self._in_progress.add(mod_name)
        try:
            symbols: dict[str, Symbol] = {}
            fallback = False
            for stmt in self.head_uses(mod_name):
                used, whole_external = self.use_symbols(stmt)
                symbols.update(used)
                fallback |= whole_external
            symbols.update(self.own_symbols(mod_name))
        finally:
            self._in_progress.discard(mod_name)
        self._visible[mod_name] = _Exports(symbols, fallback)
        return self._visible[mod_name]

    def exports(self, mod_name: str) -> _Exports:
        """Public names of `mod_name`."""
        if mod_name not in self._exports:
            scope = self.visible(mod_name)
            head = self.head(mod_name)
            public = {n: s for n, s in scope.symbols.items() if head.is_public(n)}
            exp = _Exports(public, scope.external_fallback and not head.default_private)
            if mod_name in self._in_progress:
                return exp  # incomplete; don't cache
            self._exports[mod_name] = exp
        return self._exports[mod_name]


def external(name: str) -> Symbol:
    return Symbol(name, SymbolKind.EXTERNAL, Origin.EXTERNAL)
