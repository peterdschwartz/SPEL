"""
Symbols and scopes used to classify AST nodes and walk subroutines.

Lookup order (innermost first):
    associate scopes -> routine scope -> host scope (internal subprograms)
    -> external resolver (module globals / use-association / procedures)
    -> Fortran intrinsics
    -> resolver fallback (whole-module use of an unavailable module -> EXTERNAL)

Every name must resolve: SPEL does not allow implicit typing.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING, Callable, Optional, Protocol

from spel.scripts.fortran_parser.spel_ast import SemanticError

if TYPE_CHECKING:
    from spel.scripts.fortran_parser.spel_ast import UseStatement, VariableDecl


class SymbolKind(Enum):
    VARIABLE = "variable"
    PROCEDURE = "procedure"
    INTRINSIC = "intrinsic"
    DERIVED_TYPE = "derived_type"
    ALIAS = "alias"  # associate-name for a variable designator
    NAMELIST = "namelist"  # namelist group name
    # from a module/library SPEL can't parse: may be data or a procedure
    EXTERNAL = "external"


class Origin(Enum):
    DUMMY = "dummy"
    LOCAL = "local"
    RESULT = "result"
    ASSOCIATE = "associate"  # associate-name for a non-designator expression
    HOST = "host"
    GLOBAL = "global"
    INTRINSIC = "intrinsic"
    EXTERNAL = "external"  # use-associated from an unavailable module


@dataclass(frozen=True)
class Symbol:
    """
    name:   name as visible in the scope
    kind:   what the name denotes
    origin: where the name is declared. For aliases, the origin of the target's base.
    target: expanded designator path for ALIAS symbols (e.g. "this%x"), or the
            defining routine id ("mod::name") for module procedures
    decl:   declaring VariableDecl when available
    """

    name: str
    kind: SymbolKind
    origin: Origin
    target: Optional[str] = None
    decl: Optional[VariableDecl] = field(default=None, compare=False, repr=False)

    @property
    def is_data(self) -> bool:
        return self.kind in (SymbolKind.VARIABLE, SymbolKind.ALIAS)


class SymbolResolver(Protocol):
    """Resolves names not declared inside the routine (module scope)."""

    def lookup(self, name: str) -> Optional[Symbol]: ...

    def fallback(self, name: str) -> Optional[Symbol]:
        """Last resort after intrinsics (e.g. names from an unavailable module)."""
        ...

    def use_module(self, stmt: UseStatement) -> dict[str, Symbol]:
        """Symbols made visible by a `use` statement inside the routine."""
        ...


class DictResolver:
    """
    Dictionary backed resolver.
        globals: names visible at module scope
        modules: module name -> public names of that module
    """

    def __init__(
        self,
        globals: Optional[dict[str, Symbol]] = None,
        modules: Optional[dict[str, dict[str, Symbol]]] = None,
    ):
        self.globals: dict[str, Symbol] = globals or {}
        self.modules: dict[str, dict[str, Symbol]] = modules or {}

    def lookup(self, name: str) -> Optional[Symbol]:
        return self.globals.get(name)

    def fallback(self, name: str) -> Optional[Symbol]:
        return None

    def use_module(self, stmt: UseStatement) -> dict[str, Symbol]:
        public = self.modules.get(stmt.module)
        if public is None:
            raise SemanticError(f"Unknown module '{stmt.module}' in {stmt}")
        return apply_use(stmt, public)


def apply_use(
    stmt: UseStatement,
    public: dict[str, Symbol],
    missing: Optional[Callable[[str], Optional[Symbol]]] = None,
) -> dict[str, Symbol]:
    """
    Names made visible by `stmt` given the public names of the used module.
    `missing(remote)` may supply a symbol for a name not in `public`
    (otherwise that is an error).
    """
    if not stmt.has_only:
        visible = dict(public)
        for local, remote in (rename_pair(r) for r in stmt.renames):
            visible.pop(remote, None)
            visible[local] = rename_symbol(public, remote, local, stmt, missing)
        return visible
    visible = {}
    for obj in stmt.objs:
        pair = rename_pair(obj)
        local, remote = pair if pair else (str(obj), str(obj))
        if local.startswith(("operator(", "assignment(")):
            continue
        visible[local] = rename_symbol(public, remote, local, stmt, missing)
    return visible


def rename_pair(expr) -> Optional[tuple[str, str]]:
    """`local => remote` -> (local, remote)"""
    if getattr(expr, "operator", None) == "=>":
        return str(expr.left_expr), str(expr.right_expr)
    return None


def rename_symbol(
    public: dict[str, Symbol],
    remote: str,
    local: str,
    stmt,
    missing: Optional[Callable[[str], Optional[Symbol]]] = None,
) -> Symbol:
    sym = public.get(remote)
    if sym is None and missing is not None:
        sym = missing(remote)
    if sym is None:
        raise SemanticError(f"'{remote}' is not public in module {stmt.module}")
    return replace(sym, name=local)


class Scope:
    """
    kind: "routine" | "associate" | "host"
    start_ln/end_ln: source range (associate scopes)
    """

    def __init__(
        self,
        kind: str,
        parent: Optional[Scope] = None,
        resolver: Optional[SymbolResolver] = None,
        start_ln: int = -1,
        end_ln: int = -1,
    ):
        self.kind = kind
        self.parent = parent
        self.resolver = resolver
        self.start_ln = start_ln
        self.end_ln = end_ln
        self.symbols: dict[str, Symbol] = {}

    def define(self, sym: Symbol) -> None:
        self.symbols[sym.name] = sym

    def lookup(self, name: str) -> Optional[Symbol]:
        scope: Optional[Scope] = self
        resolver: Optional[SymbolResolver] = None
        while scope is not None:
            if name in scope.symbols:
                return scope.symbols[name]
            if scope.parent is None and scope.resolver is not None:
                resolver = scope.resolver
                sym = resolver.lookup(name)
                if sym is not None:
                    return sym
            scope = scope.parent
        if name in FORTRAN_INTRINSICS:
            return Symbol(name, SymbolKind.INTRINSIC, Origin.INTRINSIC)
        return resolver.fallback(name) if resolver is not None else None

    def resolve(self, name: str, context: str = "") -> Symbol:
        sym = self.lookup(name)
        if sym is None:
            where = f" in {context}" if context else ""
            raise SemanticError(f"Undeclared symbol '{name}'{where}")
        return sym


FORTRAN_INTRINSICS: frozenset[str] = frozenset(
    {
        # numeric / math
        "abs", "acos", "acosh", "aimag", "aint", "anint", "asin", "asinh", "atan",
        "atan2", "atanh", "ceiling", "cmplx", "conjg", "cos", "cosh", "dble", "dim",
        "dprod", "erf", "erfc", "erfc_scaled", "exp", "float", "floor", "gamma", "hypot", "int",
        "log", "log10", "log_gamma", "max", "min", "mod", "modulo", "nint", "real",
        "sign", "sin", "sinh", "sqrt", "tan", "tanh", "logical", "amax1", "amin1",
        "dmax1", "dmin1", "dabs", "dsqrt", "dexp", "dlog", "alog", "alog10", "sngl",
        "idnint", "ifix",
        # numeric inquiry
        "digits", "epsilon", "exponent", "fraction", "huge", "kind", "maxexponent",
        "minexponent", "nearest", "precision", "radix", "range", "rrspacing",
        "scale", "selected_int_kind", "selected_real_kind", "set_exponent",
        "spacing", "tiny", "bit_size", "storage_size",
        # character
        "achar", "adjustl", "adjustr", "char", "iachar", "ichar", "index", "len",
        "len_trim", "lge", "lgt", "lle", "llt", "repeat", "scan", "trim", "verify",
        "new_line",
        # array
        "all", "any", "count", "cshift", "dot_product", "eoshift", "lbound",
        "matmul", "maxloc", "maxval", "merge", "minloc", "minval", "norm2", "pack",
        "product", "reshape", "shape", "size", "spread", "sum", "transpose",
        "ubound", "unpack", "findloc",
        # pointer / allocation / misc inquiry
        "allocated", "associated", "null", "present", "transfer",
        "is_iostat_end", "is_iostat_eor",
        # bit
        "btest", "iand", "ibclr", "ibits", "ibset", "ieor", "ior", "ishft",
        "ishftc", "not",
        # intrinsic subroutines
        "cpu_time", "date_and_time", "random_number", "random_seed",
        "system_clock", "move_alloc", "mvbits", "get_command_argument",
        "get_environment_variable", "execute_command_line",
    }
)
