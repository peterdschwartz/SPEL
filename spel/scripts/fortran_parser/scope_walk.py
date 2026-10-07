"""
Single in-order, scope-aware walk over a SubroutineDefinitionConstruct.

The SubroutineRecord returned by walk_subroutine is the source of truth for
routine data (declarations, variable accesses, associate scopes, calls,
pointer associations). Every name must resolve; unresolved names raise
SemanticError (SPEL does not support implicit typing).

Evaluation order of events within a statement:
  * subscripts of a designator are read before the designator is accessed
  * the rhs of an assignment is evaluated before the lhs is written
  * do-loop bounds are read before the index is written
  * associate selectors are evaluated in the enclosing scope

Access.status:
  "r"   read
  "w"   write
  "arg" designator passed as an actual argument to a user procedure;
        resolved later against the callee's dummy intent.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

from spel.scripts.fortran_parser.spel_ast import (
    AllocateStatement,
    ArrayInit,
    ArrayRef,
    AssignmentStatement,
    AssociateConstruct,
    BlockStatement,
    BoundsExpression,
    ContinueStatement,
    CycleStatement,
    DataImpliedDo,
    DataStatement,
    DoLoop,
    DoWhile,
    ExitStatement,
    Expression,
    ExpressionStatement,
    FieldAccessExpression,
    FloatLiteral,
    FormatStatement,
    FuncExpression,
    FunctionCall,
    GenericOperatorExpression,
    GotoStatement,
    Identifier,
    IfConstruct,
    ImplicitNoneStatement,
    ImportStatement,
    InfixExpression,
    IntegerLiteral,
    IntrinsicStatement,
    IOExpression,
    KeywordArgument,
    LogicalLiteral,
    MacroCallStatement,
    MacroDefine,
    MacroIf,
    NameListStatement,
    NullifyStatement,
    PointerAssignment,
    PrefixExpression,
    PrintStatement,
    ProcedureStatement,
    ReadStatement,
    ReturnStatement,
    SelectCaseConstruct,
    SemanticError,
    Statement,
    StopStatement,
    StringLiteral,
    SubCallStatement,
    SubroutineDefinitionConstruct,
    TypeDef,
    UseStatement,
    VariableDecl,
    WriteStatement,
)
from spel.scripts.fortran_parser.symbols import (
    FORTRAN_INTRINSICS,
    Origin,
    Scope,
    Symbol,
    SymbolKind,
    SymbolResolver,
)

# ---------------------------------------------------------------------------
# Record types
# ---------------------------------------------------------------------------


@dataclass
class Declaration:
    name: str
    ln: int
    type: str
    intent: Optional[str] = None
    dim: int = 0
    attrs: list[str] = field(default_factory=list)
    node: Optional[VariableDecl] = field(default=None, repr=False)


@dataclass(frozen=True)
class Access:
    """
    path:   designator with subscripts dropped and associate names expanded
    base:   first component of path
    via:    associate name used to reach path (None if accessed directly)
    context: "expr" | "assign" | "do" | "ptr-assign" | "allocate" | "deallocate"
             | "stat" | "errmsg" | "arg" | "nullify" | "read" | "iostat" | "iomsg" | "size"
    """

    path: str
    base: str
    status: str
    ln: int
    origin: Origin
    via: Optional[str] = None
    context: str = "expr"


@dataclass(frozen=True)
class CallArg:
    keyword: Optional[str]
    path: Optional[str]  # None for non-designator (expression) actual args
    expr: Expression = field(compare=False, repr=False)


@dataclass
class CallEvent:
    """passed_object: as written; object_path: with associate names expanded"""

    name: str
    ln: int
    args: list[CallArg]
    passed_object: Optional[str] = None
    node: Optional[SubCallStatement] = field(default=None, repr=False)
    object_path: Optional[str] = None


@dataclass
class FunctionRef:
    name: str
    intrinsic: bool
    ln: int
    args: list[CallArg]
    node: Optional[FunctionCall] = field(default=None, repr=False)


@dataclass(frozen=True)
class PointerAssoc:
    pointer: str
    target: Optional[str]  # None when the target is not a designator (null())
    ln: int


@dataclass(frozen=True)
class AssociateFrame:
    start_ln: int
    end_ln: int
    names: dict  # associate-name -> expanded target path | None (expression)


class SubroutineRecord:
    def __init__(self, tree: SubroutineDefinitionConstruct, scope: Scope):
        self.name: str = tree.name
        self.is_function: bool = tree.is_function
        self.tree = tree
        self.scope: Scope = scope
        self.declarations: dict[str, Declaration] = {}
        self.dummy_args: list[Declaration] = []
        self.result: Optional[Declaration] = None
        self.last_decl_ln: int = -1
        self.events: list = []
        self.frames: list[AssociateFrame] = []
        self.selects: list[tuple[int, int]] = []
        self.namelists: dict[str, list[str]] = {}

    # --- declarations ---
    @property
    def local_variables(self) -> dict[str, Declaration]:
        exclude = {d.name for d in self.dummy_args}
        if self.result:
            exclude.add(self.result.name)
        return {k: v for k, v in self.declarations.items() if k not in exclude}

    def declaration(self, name: str) -> Optional[Declaration]:
        return self.declarations.get(name)

    def lookup(self, name: str) -> Optional[Symbol]:
        """Lookup in the routine scope (outside of any associate construct)."""
        return self.scope.lookup(name)

    # --- accesses ---
    def accesses(
        self,
        path: Optional[str] = None,
        status: Optional[str] = None,
        ln: Optional[int] = None,
        origin: Optional[Origin] = None,
    ) -> list[Access]:
        return [
            e
            for e in self.events
            if isinstance(e, Access)
            and (path is None or e.path == path)
            and (status is None or e.status == status)
            and (ln is None or e.ln == ln)
            and (origin is None or e.origin is origin)
        ]

    def access_by_path(
        self, origin: Optional[Origin] = None
    ) -> dict[str, list[Access]]:
        result: dict[str, list[Access]] = {}
        for a in self.accesses(origin=origin):
            result.setdefault(a.path, []).append(a)
        return result

    # --- associate ---
    def associate_ranges(self) -> list[tuple[int, int]]:
        return [(f.start_ln, f.end_ln) for f in self.frames]

    def associations(self, ln: int) -> dict[str, Optional[str]]:
        """Associate names visible at line `ln` (outer frames first)."""
        result: dict[str, Optional[str]] = {}
        for f in self.frames:
            if f.start_ln <= ln <= f.end_ln:
                result.update(f.names)
        return result

    def select_ranges(self) -> list[tuple[int, int]]:
        return list(self.selects)

    # --- calls / pointers ---
    @property
    def calls(self) -> list[CallEvent]:
        return [e for e in self.events if isinstance(e, CallEvent)]

    @property
    def function_calls(self) -> list[FunctionRef]:
        return [e for e in self.events if isinstance(e, FunctionRef)]

    def pointer_targets(self) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for e in self.events:
            if isinstance(e, PointerAssoc) and e.target is not None:
                targets = result.setdefault(e.pointer, [])
                if e.target not in targets:
                    targets.append(e.target)
        return result


# ---------------------------------------------------------------------------
# Walk
# ---------------------------------------------------------------------------

SKIPPED_STATEMENTS = (
    UseStatement,
    ImplicitNoneStatement,
    ImportStatement,
    MacroDefine,
    NameListStatement,
    ProcedureStatement,
    TypeDef,
    ExitStatement,  # includes CycleStatement
    ReturnStatement,
    GotoStatement,
    ContinueStatement,
    IntrinsicStatement,
    FormatStatement,
)

# io-control specifiers whose variable is written by the statement
IO_OUTPUT_SPECS = {"iostat", "iomsg", "size"}

LITERALS = (IntegerLiteral, FloatLiteral, StringLiteral, LogicalLiteral)


def walk_subroutine(
    tree: SubroutineDefinitionConstruct,
    resolver: SymbolResolver,
    host: Optional[SubroutineRecord] = None,
) -> SubroutineRecord:
    """
    Walks `tree` (bodies of internal subprograms excluded). For an internal
    subprogram, `host` is the already walked record of its host routine.
    """
    if host is not None:
        host_scope = Scope("host", resolver=resolver)
        for sym in host.scope.symbols.values():
            if sym.origin not in (Origin.GLOBAL, Origin.EXTERNAL):
                sym = replace(sym, origin=Origin.HOST)
            host_scope.define(sym)
        scope = Scope("routine", parent=host_scope)
    else:
        scope = Scope("routine", resolver=resolver)

    rec = SubroutineRecord(tree, scope)
    walker = _Walker(rec, resolver)
    walker.define_routine_symbols()
    walker.walk_block(tree.body)
    return rec


class _Walker:
    def __init__(self, rec: SubroutineRecord, resolver: SymbolResolver):
        self.rec = rec
        self.resolver = resolver
        self.scope: Scope = rec.scope

    # --- specification part ---
    def define_routine_symbols(self) -> None:
        tree = self.rec.tree
        rec = self.rec
        result_name = (tree.result or tree.name) if tree.is_function else None

        decls: list[Declaration] = []
        for stmt in spec_statements(tree.body):
            if isinstance(stmt, UseStatement):
                for sym in self.resolver.use_module(stmt).values():
                    self.scope.define(sym)
            elif isinstance(stmt, VariableDecl):
                decls.extend(make_declarations(stmt))
            elif isinstance(stmt, NameListStatement):
                rec.namelists[stmt.namelist_group] = list(stmt.vars)
                self.scope.define(
                    Symbol(stmt.namelist_group, SymbolKind.NAMELIST, Origin.LOCAL)
                )
            elif isinstance(stmt, IntrinsicStatement):
                for name in stmt.names:
                    if name not in FORTRAN_INTRINSICS:
                        raise SemanticError(
                            f"'{name}' is not a known intrinsic @{stmt.lineno}"
                        )
                    self.scope.define(
                        Symbol(name, SymbolKind.INTRINSIC, Origin.INTRINSIC)
                    )
            rec.last_decl_ln = max(rec.last_decl_ln, stmt.lineno)

        for sub in tree.contains:
            self.scope.define(Symbol(sub.name, SymbolKind.PROCEDURE, Origin.LOCAL))
        if tree.name != result_name:
            self.scope.define(Symbol(tree.name, SymbolKind.PROCEDURE, Origin.LOCAL))

        for d in decls:
            rec.declarations[d.name] = d
            if d.name in tree.args:
                origin = Origin.DUMMY
            elif d.name == result_name:
                origin = Origin.RESULT
            else:
                origin = Origin.LOCAL
            self.scope.define(Symbol(d.name, SymbolKind.VARIABLE, origin, decl=d.node))

        for arg in tree.args:
            if arg not in rec.declarations:
                raise SemanticError(
                    f"Dummy argument '{arg}' of {tree.name} has no declaration"
                )
            rec.dummy_args.append(rec.declarations[arg])

        if result_name is not None:
            if result_name not in rec.declarations:
                rtype = str(tree.return_type) if tree.return_type else ""
                rec.declarations[result_name] = Declaration(
                    result_name, tree.lineno, rtype
                )
                self.scope.define(
                    Symbol(result_name, SymbolKind.VARIABLE, Origin.RESULT)
                )
            rec.result = rec.declarations[result_name]

    # --- statements ---
    def walk_block(self, block: BlockStatement) -> None:
        for stmt in block.statements:
            self.walk_statement(stmt)

    def walk_statement(self, stmt: Statement) -> None:
        ln = stmt.lineno
        match stmt:
            case ExpressionStatement():
                self.walk_statement(stmt.classify(self.scope))
            case AssignmentStatement():
                self.walk_expr(stmt.right, ln)
                self.walk_designator(stmt.left, "w", ln, context="assign")
            case PointerAssignment():
                self.walk_pointer_assignment(stmt)
            case SubCallStatement():
                self.walk_call(stmt.classify(self.scope))
            case VariableDecl():
                self.walk_declaration(stmt)
            case DoLoop():
                for e in (stmt.start, stmt.end, stmt.step):
                    if e is not None:
                        self.walk_expr(e, ln)
                if stmt.index is not None:
                    self.walk_designator(
                        Identifier(stmt.token, stmt.index), "w", ln, context="do"
                    )
                self.walk_block(stmt.body)
            case DoWhile():
                self.walk_expr(stmt.condition, ln)
                self.walk_block(stmt.body)
            case IfConstruct():
                self.walk_expr(stmt.condition, ln)
                self.walk_block(stmt.consequence)
                for elif_ in stmt.else_ifs:
                    self.walk_expr(elif_.condition, elif_.lineno)
                    self.walk_block(elif_.consequence)
                if stmt.else_ is not None:
                    self.walk_block(stmt.else_.alternative)
            case AssociateConstruct():
                self.walk_associate(stmt)
            case SelectCaseConstruct():
                self.walk_expr(stmt.selector, ln)
                for case in stmt.cases:
                    for v in case.values or []:
                        self.walk_expr(v, case.lineno)
                    self.walk_block(case.body)
                self.rec.selects.append((ln, stmt.end_ln))
            case AllocateStatement():
                self.walk_allocate(stmt)
            case DataStatement():
                for dset in stmt.sets:
                    for obj in dset.objects:
                        self.walk_data_object(obj, ln)
                    for v in dset.values:
                        if v.repeat is not None:
                            self.walk_expr(v.repeat, ln)
                        self.walk_expr(v.value, ln)
            case NullifyStatement():
                for obj in stmt.objects:
                    path = self.walk_designator(obj, "w", ln, context="nullify")
                    self.rec.events.append(PointerAssoc(path, None, ln))
            case ReadStatement():
                for c in stmt.controls:
                    self.walk_io_control(c, ln, is_read=True)
                for item in stmt.items:
                    self.walk_designator(item, "w", ln, context="read")
            case MacroCallStatement():
                for a in stmt.args:
                    self.walk_expr(a, ln)
            case StopStatement():
                if stmt.code is not None:
                    self.walk_expr(stmt.code, ln)
            case WriteStatement():
                for e in (stmt.unit, stmt.fmt):
                    if e is not None:
                        self.walk_io_control(e, ln, is_read=False)
                for e in stmt.exprs:
                    if e is not None:
                        self.walk_expr(e, ln)
            case PrintStatement():
                for e in (stmt.fmt, *stmt.exprs):
                    if e is not None:
                        self.walk_expr(e, ln)
            case MacroIf():
                # the configuration is unknown: every branch is walked
                self.walk_block(stmt.body)
                for branch in stmt.branches:
                    self.walk_block(branch.body)
            case BlockStatement():
                self.walk_block(stmt)
            case _ if isinstance(stmt, SKIPPED_STATEMENTS):
                pass
            case _:
                raise SemanticError(
                    f"Unsupported statement {type(stmt).__name__} @{ln}: {stmt}"
                )

    def walk_declaration(self, stmt: VariableDecl) -> None:
        """
        Specification expressions are reads. An initializer writes its entity,
        except for named constants (parameters are read-only).
        """
        ln = stmt.lineno
        is_parameter = False
        if stmt.attrs:
            for attr in stmt.attrs.attrs:
                if attr.get_name() == "parameter":
                    is_parameter = True
                if isinstance(attr, FuncExpression) and attr.get_name() == "dimension":
                    for e in attr.args:
                        self.walk_expr(e, ln)
        for ent in stmt.entities:
            for b in ent.bounds:
                self.walk_expr(b, ln)
            if ent.init is not None:
                self.walk_expr(ent.init, ln)
                if not is_parameter:
                    ident = Identifier(ent.token, ent.token.literal)
                    self.walk_designator(ident, "w", ln, context="init")

    def walk_io_control(self, ctrl: Expression, ln: int, is_read: bool) -> None:
        """
        Positional unit/format specs are reads. `nml=group` reads (write) or
        writes (read) every member of the group; iostat/iomsg/size are written.
        """
        if not (isinstance(ctrl, InfixExpression) and ctrl.operator == "="):
            self.walk_expr(ctrl, ln)
            return
        spec = str(ctrl.left_expr).lower()
        value = ctrl.right_expr
        if spec in IO_OUTPUT_SPECS:
            self.walk_designator(value, "w", ln, context=spec)
        elif spec == "nml":
            group = str(value)
            sym = self.scope.resolve(group, context=f"nml= @{ln}")
            if sym.kind is not SymbolKind.NAMELIST:
                raise SemanticError(f"'{group}' is not a namelist group @{ln}")
            for var in self.rec.namelists[group]:
                ident = Identifier(ctrl.token, var)
                if is_read:
                    self.walk_designator(ident, "w", ln, context="read")
                else:
                    self.walk_designator(ident, "r", ln)
        else:
            self.walk_expr(value, ln)

    def walk_pointer_assignment(self, stmt: PointerAssignment) -> None:
        ln = stmt.lineno
        target = None
        if self.is_designator(stmt.target):
            target, _, _, _ = self.resolve_designator(stmt.target, ln)
        else:
            self.walk_expr(stmt.target, ln)
        pointer = self.walk_designator(stmt.pointer, "w", ln, context="ptr-assign")
        self.rec.events.append(PointerAssoc(pointer, target, ln))

    def walk_call(self, stmt: SubCallStatement) -> None:
        ln = stmt.lineno
        name = str(stmt.function.function)
        passed_object = None
        object_path = None
        if "%" in name:
            passed_object = name.rsplit("%", 1)[0]
            sym = self.scope.resolve(
                passed_object.split("%")[0], context=f"call {name}"
            )
            if not is_data_ref(sym, name):
                raise SemanticError(f"'{passed_object}' is not a data object @{ln}")
            obj = Identifier(stmt.function.function.token, passed_object)
            object_path = self.walk_designator(obj, "arg", ln, context="arg")
        else:
            sym = self.scope.resolve(name, context=f"call @{ln}")
            if sym.kind not in (
                SymbolKind.PROCEDURE,
                SymbolKind.INTRINSIC,
                SymbolKind.EXTERNAL,
            ):
                raise SemanticError(f"'{name}' is not a subroutine @{ln}")
        args = self.walk_actual_args(stmt.function.args, ln)
        self.rec.events.append(
            CallEvent(name, ln, args, passed_object, stmt, object_path=object_path)
        )

    def walk_associate(self, stmt: AssociateConstruct) -> None:
        ln = stmt.lineno
        inner = Scope("associate", parent=self.scope, start_ln=ln, end_ln=stmt.end_ln)
        names: dict[str, Optional[str]] = {}
        # selectors are evaluated in the enclosing scope
        for name, selector in stmt.associations.items():
            if self.is_designator(selector):
                path, _, origin, _ = self.resolve_designator(selector, ln)
                inner.define(Symbol(name, SymbolKind.ALIAS, origin, target=path))
                names[name] = path
            else:
                self.walk_expr(selector, ln)
                inner.define(Symbol(name, SymbolKind.VARIABLE, Origin.ASSOCIATE))
                names[name] = None
        self.rec.frames.append(AssociateFrame(ln, stmt.end_ln, names))
        outer, self.scope = self.scope, inner
        try:
            self.walk_block(stmt.body)
        finally:
            self.scope = outer

    def walk_data_object(self, obj: Expression, ln: int) -> None:
        if not isinstance(obj, DataImpliedDo):
            self.walk_designator(obj, "w", ln, context="data")
            return
        loop = obj.loop
        for e in (loop.start_expr, loop.end_expr, loop.step_expr):
            if e is not None:
                self.walk_expr(e, ln)
        # the implied-do index has statement scope: it needs no declaration
        # and its uses are not accesses of any routine variable
        inner = Scope("data-implied-do", parent=self.scope)
        inner.define(
            Symbol(loop.index.literal, SymbolKind.VARIABLE, Origin.ASSOCIATE)
        )
        outer, self.scope = self.scope, inner
        try:
            for o in obj.objects:
                self.walk_data_object(o, ln)
        finally:
            self.scope = outer

    def walk_allocate(self, stmt: AllocateStatement) -> None:
        ln = stmt.lineno
        context = "deallocate" if stmt.is_deallocate else "allocate"
        for obj in stmt.objects:
            self.walk_designator(obj, "w", ln, context=context)
        for key, val in stmt.options.items():
            if key in ("stat", "errmsg"):
                self.walk_designator(val, "w", ln, context=key)
            else:  # source / mold
                self.walk_expr(val, ln)

    # --- expressions ---
    def walk_expr(self, expr: Expression, ln: int) -> None:
        match expr:
            case _ if isinstance(expr, LITERALS):
                pass
            case Identifier():
                sym = self.scope.resolve(expr.value.split("%")[0], context=f"@{ln}")
                if is_data_ref(sym, expr.value):
                    self.walk_designator(expr, "r", ln)
            case FuncExpression():
                node = expr.classify(self.scope)
                if isinstance(node, ArrayRef):
                    self.walk_designator(node, "r", ln)
                else:
                    self.walk_function_call(node, ln)
            case FieldAccessExpression():
                self.walk_designator(expr, "r", ln)
            case InfixExpression():
                self.walk_expr(expr.left_expr, ln)
                self.walk_expr(expr.right_expr, ln)
            case PrefixExpression():
                self.walk_expr(expr.right_expr, ln)
            case BoundsExpression():
                for e in (expr.start, expr.end):
                    if e is not None:
                        self.walk_expr(e, ln)
            case ArrayInit():
                ido = expr.implied_do
                if ido is not None:
                    for e in (ido.start_expr, ido.end_expr, ido.step_expr):
                        if e is not None:
                            self.walk_expr(e, ln)
                for e in expr.elements:
                    self.walk_expr(e, ln)
            case IOExpression():
                if expr.expr is not None:
                    self.walk_expr(expr.expr, ln)
            case KeywordArgument():
                self.walk_expr(expr.value, ln)
            case GenericOperatorExpression():
                pass
            case _:
                raise SemanticError(
                    f"Unsupported expression {type(expr).__name__} @{ln}: {expr}"
                )

    def walk_function_call(self, node: FunctionCall, ln: int) -> None:
        if node.intrinsic:
            for a in node.args:
                self.walk_expr(a, ln)
            args = [
                CallArg(a.keyword if isinstance(a, KeywordArgument) else None, None, a)
                for a in node.args
            ]
        else:
            args = self.walk_actual_args(node.args, ln)
        self.rec.events.append(FunctionRef(node.name, node.intrinsic, ln, args, node))

    def walk_actual_args(self, actuals: list[Expression], ln: int) -> list[CallArg]:
        args = []
        for a in actuals:
            keyword = a.keyword if isinstance(a, KeywordArgument) else None
            value = a.value if isinstance(a, KeywordArgument) else a
            path = None
            external = isinstance(value, Identifier) and (
                self.scope.resolve(value.value.split("%")[0]).kind
                is SymbolKind.EXTERNAL
            )
            # an external name of unknown kind may be data in an argument position
            if external or self.is_designator(value):
                path = self.walk_designator(value, "arg", ln, context="arg")
            else:
                self.walk_expr(value, ln)
            args.append(CallArg(keyword, path, a))
        return args

    # --- designators ---
    def is_designator(self, expr: Expression) -> bool:
        match expr:
            case Identifier():
                sym = self.scope.resolve(expr.value.split("%")[0])
                return is_data_ref(sym, expr.value)
            case FuncExpression():
                return isinstance(expr.classify(self.scope), ArrayRef)
            case FieldAccessExpression():
                return True
        return False

    def resolve_designator(
        self, expr: Expression, ln: int
    ) -> tuple[str, str, Origin, Optional[str]]:
        """
        Reads all subscripts, then returns (path, base, origin, via) for the
        designator without recording an access to it.
        """
        comps = flatten_designator(expr)
        for _, subs in comps:
            for s in subs:
                self.walk_expr(s, ln)
        names = [n for n, _ in comps]
        sym = self.scope.resolve(names[0], context=f"@{ln}")
        via = None
        if sym.kind is SymbolKind.ALIAS:
            via = names[0]
            names = sym.target.split("%") + names[1:]
        path = "%".join(names)
        return path, names[0], sym.origin, via

    def walk_designator(
        self, expr: Expression, status: str, ln: int, context: str = "expr"
    ) -> str:
        path, base, origin, via = self.resolve_designator(expr, ln)
        self.rec.events.append(Access(path, base, status, ln, origin, via, context))
        return path


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def is_data_ref(sym: Symbol, name: str) -> bool:
    """
    Whether `name` (resolved to `sym`) refers to data. A name of unknown
    kind (from an unavailable module) is data when it has components.
    """
    if sym.is_data or sym.origin is Origin.ASSOCIATE:
        return True
    return sym.kind is SymbolKind.EXTERNAL and "%" in name


def flatten_designator(expr: Expression) -> list[tuple[str, list[Expression]]]:
    """x%y(i)%z(j) -> [("x", []), ("y", [i]), ("z", [j])]"""
    match expr:
        case Identifier():
            return [(n, []) for n in expr.value.split("%")]
        case FuncExpression():
            comps = flatten_designator(expr.function)
            name, _ = comps[-1]
            comps[-1] = (name, list(expr.args))
            return comps
        case FieldAccessExpression():
            return flatten_designator(expr.left) + flatten_designator(expr.field)
    raise SemanticError(f"Not a variable designator: {expr}")


def spec_statements(block: BlockStatement):
    """Top-level specification statements (also inside preprocessor blocks)."""
    for stmt in block.statements:
        if isinstance(
            stmt,
            (
                UseStatement,
                VariableDecl,
                ImplicitNoneStatement,
                ImportStatement,
                NameListStatement,
                IntrinsicStatement,
                DataStatement,
            ),
        ):
            yield stmt
        elif isinstance(stmt, MacroIf):
            yield from spec_statements(stmt.body)
            for branch in stmt.branches:
                yield from spec_statements(branch.body)


def make_declarations(stmt: VariableDecl) -> list[Declaration]:
    intent = None
    attr_names: list[str] = []
    attr_dim = 0
    if stmt.attrs:
        for attr in stmt.attrs.attrs:
            attr_names.append(attr.get_name())
            if isinstance(attr, FuncExpression):
                if attr.get_name() == "intent" and attr.args:
                    intent = str(attr.args[0]).replace(" ", "").lower()
                elif attr.get_name() == "dimension":
                    attr_dim = len(attr.args)
    return [
        Declaration(
            name=ent.token.literal,
            ln=stmt.lineno,
            type=str(stmt.var_type),
            intent=intent,
            dim=len(ent.bounds) if ent.bounds else attr_dim,
            attrs=list(attr_names),
            node=stmt,
        )
        for ent in stmt.entities
    ]
