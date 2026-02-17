from collections import defaultdict
from functools import lru_cache

from ..models import CallsiteBinding, PropagatedEffectByLn, SubroutineArgs, SubroutineCalltree
from .ifs import filter_access_lns_by_hash, filter_calltree_by_hash

SubName = str
Dtype = str
RW = tuple[int, str, SubName]

def reachable_subroutine_ids(root_subroutine, cfg_hash: str) -> set[int]:
    root_id = root_subroutine.subroutine_id
    seen: set[int] = {root_id}
    frontier: list[int] = [root_id]

    while frontier:
        # All edges whose *parent* is in the current frontier, already pruned
        qs = SubroutineCalltree.objects.filter(parent_subroutine__in=frontier)
        qs = filter_calltree_by_hash(qs, cfg_hash)

        child_ids = list(qs.values_list("child_subroutine", flat=True).distinct())
        # De-dup and avoid cycles
        new_ids = [cid for cid in child_ids if cid not in seen]
        if not new_ids:
            break
        seen.update(new_ids)
        frontier = new_ids

    return seen


def get_propagated_elmtypes(
    reachable_ids: set[int],
    root_sub_id: int,
    cfg,
) -> dict[Dtype, list[RW]]:
    # Figure out which globals are Propagated to children via arguments.
    # 1) Internal edges: caller is in the subtree rooted at `root_sub_id`
    internal_qs = PropagatedEffectByLn.objects.filter(
        call_site__parent_subroutine__subroutine_id__in=reachable_ids,
    )

    # 2) Incoming edges: calls *into* root (elm_drv -> laketemperature)
    into_root_qs = PropagatedEffectByLn.objects.filter(
        call_site__child_subroutine__subroutine_id=root_sub_id,
    )
    # Union and deduplicate
    prop_qs = (
        (internal_qs | into_root_qs)
        .select_related(
            "call_site",
            "binding",
        )
        .order_by(
            "call_site__child_subroutine__subroutine_name",
            "lineno",
        )
        .distinct()
    )

    prop_qs = filter_access_lns_by_hash(
        qs=prop_qs,
        sub_field="call_site__parent_subroutine",
        lineno_field="call_site__lineno",
        config_hash=cfg,
    )
    prop_qs = filter_access_lns_by_hash(
        qs=prop_qs,
        sub_field="call_site__child_subroutine",
        lineno_field="lineno",
        config_hash=cfg,
    )
    # for row in prop_qs:
    #     print(
    #         "prop_id=", row.prop_id,
    #         " parent_sub=", row.call_site.parent_subroutine_id,
    #         " child_sub=", row.call_site.child_subroutine_id,
    #         " call_ln=", row.call_site.lineno,
    #         " acc_ln=", row.lineno,
    #         " in_inactive=", row.in_inactive,
    #     )
    tmp_statuses: dict[Dtype, list[RW]] = defaultdict(list)
    for row in prop_qs:
        instances = instances_for_propagated_row(row, cfg)
        if "c13_col_cf" in instances:
            print(row)

        # Only care about rows that correspond to some instance%member
        if not instances:
            continue

        for inst in instances:
            full_ref = f"{inst}%{row.member_path}"
            tmp_statuses[full_ref].append(
                (row.lineno, row.status, row.call_site.child_subroutine.subroutine_name)
            )

    return tmp_statuses


@lru_cache(maxsize=None)
def trace_var_to_instances(
    sub_id: int,
    var_name: str,
    scope: str,
    cfg:str,
) -> tuple[str, ...]:
    """
    Does the variable `var_name` (with given `scope`) in subroutine `sub_id`
    ultimately come from the ELMTYPE instance `instance_name`?

    Handles arbitrary depth:
      root (ELMTYPE inst)
        -> child1(ARG inst)
            -> child2(ARG inst_dummy)
                -> ...

    Uses CallsiteBinding where this subroutine appears as the *callee*,
    to see what callers pass into its arguments.
    """

    # Base case: this variable is an ELMTYPE-instance in this subroutine
    if scope == "ELMTYPE":
        if var_name == "this":
            print(f"ERROR - {var_name} scoped as ELMTYPE:\n{sub_id}, {scope}")
        return (var_name,)

    # Local variables never trace back to a top-level ELMTYPE instance
    if scope == "LOCAL":
        return ()

    if scope != "ARG":
        # Unknown / Unexpected scope
        return ()

    # scope == "ARG": var_name is an argument name of this subroutine
    # 1) Find the SubroutineArgs rows for this arg in this subroutine
    arg_ids = list(
        SubroutineArgs.objects.filter(
            subroutine_id=sub_id,
            arg_name=var_name,
        ).values_list("arg_id", flat=True)
    )
    if not arg_ids:
        return ()

    # 2) For each callsite where this subroutine is the *callee* and this arg
    #    is the dummy, see what the *caller* binds it to (ELMTYPE, ARG, LOCAL)
    parent_bindings = CallsiteBinding.objects.filter(
        call__child_subroutine_id=sub_id,
        dummy_arg_id__in=arg_ids,
    ).select_related("parent_subroutine")
    parent_bindings = filter_access_lns_by_hash(
        qs=parent_bindings,
        sub_field="call__parent_subroutine_id",
        lineno_field="call__lineno",
        config_hash=cfg,
    )

    instances: set[str] = set()
    for b in parent_bindings:
        parent_sub_id = b.parent_subroutine_id
        parent_var_name = b.var_name
        parent_scope = scope_for_binding(b)

        instances.update(
            trace_var_to_instances(
                parent_sub_id,
                parent_var_name,
                parent_scope,
                cfg,
            )
        )

    return tuple(sorted(instances))


def propagated_row_matches_instance(
    row: PropagatedEffectByLn,
    instance_name: str,
    member_name: str,
    check_all: bool,
    cfg_hash:str,
) -> bool:
    """
    Given a PropagatedEffectByLn row, decide if it corresponds to
    `instance_name%member_name` when traced back up the call tree.
    """

    # Only consider rows with the right member_path
    if not check_all and row.member_path != member_name:
        return False

    parent_sub = row.call_site.parent_subroutine
    sub_id = parent_sub.subroutine_id

    instances = trace_var_to_instances(
        sub_id=sub_id,
        var_name=str(row.var_name),
        scope=str(row.scope),
        cfg=cfg_hash,
    )

    return instance_name in instances


def instances_for_propagated_row(
    row: PropagatedEffectByLn,
    cfg,
) -> set[str]:
    """
    For this propagated row, return all ELMTYPE instances that the
    (var_name, scope) can ultimately come from.
    """
    parent_sub = row.call_site.parent_subroutine
    sub_id = parent_sub.subroutine_id

    return set(
        trace_var_to_instances(
            sub_id=sub_id,
            var_name=str(row.var_name),
            scope=str(row.scope),
            cfg=cfg,
        )
    )


def scope_for_binding(b: CallsiteBinding) -> str:
    """
    Returns the effective scope for this binding, using PropagatedEffectByLn
    as the source of truth. Assumes all propagated rows for a binding
    agree on scope.
    """
    scopes = set(
        PropagatedEffectByLn.objects.filter(binding=b)
        .values_list("scope", flat=True)
        .distinct()
    )
    if not scopes:
        # Fallback: no propagated rows for this binding; treat as unknown
        return ""

    if len(scopes) > 1:
        return sorted(scopes)[0]

    return scopes.pop()


def combine_many_statuses(statuses: list[str]) -> str:
    """
    Combine read/write statuses in program order.

    Rules:
    - If the first access is a pure write ('w'), overall status is 'w'
      regardless of later reads.
    - Otherwise, overall status is the union of all accesses.
    """

    if not statuses:
        return ""

    # First access determines input-ness
    first = statuses[0]

    if first == "w":
        return "w"

    # Otherwise, fall back to union logic
    perms = set()
    for s in statuses:
        perms |= set(s)

    return "".join(sorted(perms))


