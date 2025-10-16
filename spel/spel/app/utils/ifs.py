from functools import lru_cache

from django.db import transaction
from django.db.models import Exists, OuterRef, Prefetch

from scripts.fortran_parser import lexer
from scripts.fortran_parser.evaluate_ifs import eval_if_condition
from scripts.fortran_parser.spel_ast import (ExpressionStatement,
                                             PrefixExpression)
from scripts.fortran_parser.spel_parser import Parser
from scripts.types import LineTuple, LogicalLineIterator

from ..models import (FlatIf, FlatIfNamelistVar, IfEvaluationByHash,
                      IntrinsicGlobals, NamelistVariable)


def build_env_from_active(active_data) -> dict[str, str]:
    """
    active_data keys are IntrinsicGlobals.var_id (as strings) → overrides.
    For all NamelistVariables, take override if present, else IntrinsicGlobals.value.
    Return env: var_name -> Fortran-ish literal string.
    """
    # Pull the variables of interest with their module/name/defaults
    rows = NamelistVariable.objects.select_related("active_var_id").values_list(
        "active_var_id__var_id",
        "active_var_id__var_name",
        "active_var_id__value",
    )

    env: dict[str, str] = {}

    for var_id, var_name, default_value in rows:
        key = str(var_id)
        # Use override if present; otherwise default from IntrinsicGlobals.value
        val = active_data.get(key, default_value).lower()
        env[var_name] = val
    return env


def _get_expr_value(expr):
    if isinstance(expr, PrefixExpression):
        if expr.operator == "-":
            return -1 * expr.right_expr.value
        else:
            return expr.right_expr.value
    else:
        # should jsut be a xxLiteral
        return expr.value


@lru_cache(maxsize=1_000)
def parse_condition_ast(norm_cond: str):
    print(f"parsing {norm_cond}")
    li = LogicalLineIterator([LineTuple(line=norm_cond, ln=1)])
    lex = lexer.Lexer(li)
    p = Parser(lex=lex)
    program = p.parse_program()
    return program.statements.pop()


@transaction.atomic
def recompute_if_evals_for_hash(config_hash: str, active_data) -> int:
    """
    Recompute and store FlatIf evaluations for the given config hash.
    Returns number of rows written.
    """
    env = build_env_from_active(active_data)

    flatifs = (
        FlatIf.objects.select_related("subroutine")
        .prefetch_related(
            Prefetch(
                "flatifnamelistvar_set",
                queryset=FlatIfNamelistVar.objects.select_related("namelist_var"),
            )
        )
        .only("flatif_id", "subroutine_id", "start_ln", "end_ln", "condition")
    )

    IfEvaluationByHash.objects.filter(config_hash=config_hash).delete()

    lines = [f"{k} = {v}" for k, v in env.items()]
    lex = lexer.Lexer(LogicalLineIterator([LineTuple(line=s, ln=1) for s in lines]))
    parser = Parser(lex=lex)
    program = parser.parse_program()

    eval_env = {
        stmt.expression.left_expr.value: _get_expr_value(stmt.expression.right_expr)
        for stmt in program.statements
    }
    batch = []
    for fi in flatifs.iterator(chunk_size=1000):
        cond_ast = parse_condition_ast(fi.condition)
        is_active = bool(eval_if_condition(cond_ast, eval_env))
        batch.append(
            IfEvaluationByHash(
                config_hash=config_hash,
                flatif_id=fi.pk,
                subroutine_id=fi.subroutine_id,
                start_ln=fi.start_ln,
                end_ln=fi.end_ln,
                is_active=is_active,
            )
        )

    IfEvaluationByHash.objects.bulk_create(batch, ignore_conflicts=True)
    return len(batch)


def filter_calltree_by_hash(qs, config_hash: str):
    """
    Hide calltree edges whose lineno falls inside any INACTIVE IF range
    for the *parent* subroutine under the given config hash.
    qs: queryset over SubroutineCalltree
    """
    inactive_edges = IfEvaluationByHash.objects.filter(
        config_hash=config_hash,
        subroutine_id=OuterRef("parent_subroutine_id"),
        is_active=False,
        start_ln__lte=OuterRef("lineno"),
        end_ln__gte=OuterRef("lineno"),
    )
    return qs.annotate(in_inactive=Exists(inactive_edges)).filter(in_inactive=False)


def filter_dtype_vars_by_hash(qs, config_hash: str):
    """
    Hide dtype/global-var rows whose ln falls inside any INACTIVE IF range
    for the row's subroutine under the given config hash.
    qs: queryset over SubroutineActiveGlobalVars (must have fields: subroutine_id, ln)
    """
    inactive_var_ranges = IfEvaluationByHash.objects.filter(
        config_hash=config_hash,
        subroutine_id=OuterRef("subroutine_id"),
        is_active=False,
        start_ln__lte=OuterRef("ln"),
        end_ln__gte=OuterRef("ln"),
    )
    return qs.annotate(in_inactive=Exists(inactive_var_ranges)).filter(
        in_inactive=False
    )
