import json
from collections import defaultdict

from django import template
from django.conf import settings
from django.contrib import messages
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
from django.db import connection, models, transaction
from django.db.models import Case, IntegerField, QuerySet, Value, When
from django.db.models.functions import Concat
from django.db.models.query import Prefetch
from django.http import HttpResponseBadRequest
from django.shortcuts import HttpResponse, get_object_or_404, redirect, render
from django.template.loader import render_to_string
from django.views import View
from django.views.decorators.http import require_http_methods

from .calltree import (
    annotate_tree,
    create_calltree_from_sub,
    get_calltree_for_var,
    print_node,
    prune_tree,
)
from .gen_html import build_tree_html
from .models import (
    ArgAccess,
    CallsiteBinding,
    ConfigProfile,
    FlatIf,
    FlatIfNamelistVar,
    IfEvaluationByHash,
    IntrinsicGlobals,
    ModuleDependency,
    Modules,
    NamelistVariable,
    PresetConfig,
    PropagatedEffectByLn,
    SubroutineActiveGlobalVars,
    SubroutineArgs,
    SubroutineCalltree,
    SubroutineElmtypesByConfig,
    SubroutineIntrinsicGlobals,
    Subroutines,
    TypeDefinitions,
    UserTypeInstances,
)
from .signals import cache_tree, retrieve_cache_tree
from .utils.configs import get_active_config
from .utils.ifs import (
    compute_if_evals_for_hash,
    debug_inactive_ifs,
    filter_access_lns_by_hash,
    filter_calltree_by_hash,
)
from .utils.populate_config_tables import propagate_bindings
from .utils.tables import VIEWS_TABLE_DICT
from .utils.view_helper import (
    combine_many_statuses,
    propagated_row_matches_instance,
    reachable_subroutine_ids,
)

register = template.Library()

FORTRAN_TYPES = {"complex", "character", "integer", "real", "logical"}


def execute(statement):
    with connection.cursor() as cur:
        cur.execute(statement)


def var_active_search(request):
    q = (request.POST.get("search") or "").strip()
    LIMIT = 12
    items: list[dict] = []

    if not q:
        return _render_var_results([], q)

    if "%" in q:
        inst_part, field_part = _split_instance_field(q)
        items = _search_instance_fields(inst_part, field_part, limit=LIMIT)
    else:
        items = _search_instances_and_globals(q, limit=LIMIT)

    return _render_var_results(items, q)


def _render_var_results(items: list[dict], query: str) -> HttpResponse:
    html = render_to_string(
        "partials/search/var_active_search.html",
        {"items": items, "query": query},
    )
    return HttpResponse(html)


def _startswith_annotate(qs: QuerySet, field_name: str, needle: str) -> QuerySet:
    """Annotate with integer 'startswith': 0 if field starts with needle, 1 otherwise."""
    return qs.annotate(
        startswith=Case(
            When(**{f"{field_name}__istartswith": needle}, then=Value(0)),
            default=Value(1),
            output_field=IntegerField(),
        )
    )


def _split_instance_field(q: str) -> tuple[str, str]:
    """Split 'inst%field' into (instance_part, field_part)."""
    inst_part, field_part = q.split("%", 1)
    return inst_part.strip(), field_part.strip()


def _search_instances_and_globals(needle: str, limit: int) -> list[dict]:
    inst_items = _instance_items(needle, limit)
    glob_items = _global_items(needle, limit)
    merged = [*inst_items, *glob_items]
    merged.sort(key=lambda x: (x["startswith"], x["label"].lower()))
    return merged[:limit]


def _instance_items(needle: str, limit: int) -> list[dict]:
    qs = UserTypeInstances.objects.select_related(
        "inst_module", "instance_type"
    ).filter(instance_name__icontains=needle)
    qs = _startswith_annotate(qs, "instance_name", needle).order_by(
        "startswith", "instance_name", "inst_module__module_name"
    )[:limit]

    return [
        {
            "kind": "instance",
            "startswith": getattr(r, "startswith", 1),
            "label": f"{r.inst_module.module_name}::{r.instance_name}%",
            "qualified": f"{r.instance_name}%",
        }
        for r in qs
    ]


def _global_items(needle: str, limit: int) -> list[dict]:
    qs = IntrinsicGlobals.objects.select_related("gv_module").filter(
        var_name__icontains=needle
    )
    qs = _startswith_annotate(qs, "var_name", needle).order_by(
        "startswith", "var_name", "gv_module__module_name"
    )[:limit]

    return [
        {
            "kind": "global",
            "startswith": getattr(g, "startswith", 1),
            "label": f"{g.gv_module.module_name}::{g.var_name}",
            "qualified": f"{g.var_name}",
        }
        for g in qs
    ]


def _search_instance_fields(inst_part: str, field_part: str, limit: int) -> list[dict]:
    """Search derived-type fields for instances matching inst_part%field_part."""
    if not inst_part:
        return []

    insts = list(
        _startswith_annotate(
            UserTypeInstances.objects.select_related(
                "inst_module", "instance_type"
            ).filter(instance_name__icontains=inst_part),
            "instance_name",
            inst_part,
        ).order_by("startswith", "instance_name", "inst_module__module_name")[:limit]
    )

    if not insts:
        return []

    type_ids = {i.instance_type_id for i in insts}
    defs_qs = TypeDefinitions.objects.filter(user_type_id__in=type_ids)
    if field_part:
        defs_qs = defs_qs.filter(member_name__icontains=field_part)

    defs_qs = _startswith_annotate(defs_qs, "member_name", field_part or "")
    defs_qs = defs_qs.select_related("user_type", "type_module").order_by(
        "startswith", "member_name"
    )[: 5 * limit]

    defs_by_type: dict[int, list[TypeDefinitions]] = {}
    for d in defs_qs:
        defs_by_type.setdefault(d.user_type_id, []).append(d)

    items: list[dict] = []
    for inst in insts:
        for d in defs_by_type.get(inst.instance_type_id, []):
            label = (
                f"{inst.inst_module.module_name}::{inst.instance_name}%{d.member_name}"
            )
            items.append(
                {
                    "kind": "field",
                    "startswith": getattr(d, "startswith", 1),
                    "label": label,
                    "qualified": f"{inst.instance_name}%{d.member_name}",
                }
            )
            if len(items) >= limit:
                return _sorted_items(items)
    return _sorted_items(items)


def _sorted_items(items: list[dict]) -> list[dict]:
    """Sort results consistently by (startswith, label)."""
    return sorted(items, key=lambda x: (x["startswith"], x["label"].lower()))


def sub_active_search(request):
    q = (request.POST.get("search") or "").strip()

    if not q:
        html = render_to_string(
            "partials/search/sub_active_search.html",
            {
                "query": q,
                "results": [],
            },
        )
        return HttpResponse(html)

    # Prioritize names that start with the query, then fall back to icontains
    qs = (
        Subroutines.objects.select_related("module")  # pulls Modules in one query
        .annotate(
            startswith=Case(
                When(subroutine_name__istartswith=q, then=Value(0)),
                default=Value(1),
                output_field=IntegerField(),
            ),
            display_label=Concat("module__module_name", Value("::"), "subroutine_name"),
        )
        .filter(subroutine_name__icontains=q)  # search on subroutine only
        .order_by("startswith", "subroutine_name", "module__module_name")[:10]
    )

    html = render_to_string(
        "partials/search/sub_active_search.html",
        {
            "results": qs,
            "query": q,
        },
    )
    return HttpResponse(html)


def view_call_tree(request):
    if request.method == "POST":
        sub_name = request.POST.get("Subroutine")
    else:
        sub_name = "elm_drv"
    tree_list = create_calltree_from_sub(sub_name)
    html_tree = build_tree_html(tree_list)
    context = {"tree": html_tree}
    return render(request, "partials/call_tree.html", context)


def subcall(request):
    if request.method == "POST":
        variable = request.POST.get("Variable")
        if "%" in variable:
            instance, member = variable.split("%")
        else:
            # Assume just an instance.
            instance = variable
            member = ""
    else:
        instance = "bounds"
        member = "begc"
    tree_list, all = get_calltree_for_var(instance, member)

    html_tree = build_tree_html(tree_list)
    context = {
        "tree": html_tree,
        "all": all,
    }
    if request.method == "POST":
        return render(request, "partials/subcall_partial.html", context)
    return render(request, "partials/subcall_partial.html", context)


@require_http_methods(["GET", "POST"])
def view_table(request, table_name):
    """
    Generic Function for printing an SQL table, substituting
    the foreign keys with as specifcied in the table definiton
    """

    table = VIEWS_TABLE_DICT[table_name]
    if not table:
        return HttpResponse(b"Table not found", status=404)

    model = table["name"]
    display_fields = table["fields"]
    title = table["title"]
    if request.method == "POST":
        sort_by = request.POST.get("sort_by", None)
    else:
        sort_by = None

    foreign_keys = [
        field.name
        for field in model._meta.get_fields()
        if isinstance(field, models.ForeignKey)
    ]
    all_objects = model.objects.select_related(*foreign_keys).all()

    if sort_by:
        sort_field = display_fields[sort_by]
        all_objects = all_objects.order_by(sort_field.replace(".", "__"))

    rows = []
    for obj in all_objects:
        row = []
        for field_name in display_fields.values():
            parts = field_name.split(".")
            value = getattr(obj, parts[0], None)
            if len(parts) > 1:
                for attr in parts[1:]:
                    value = getattr(value, attr, None)

            row.append(value)
        rows.append(row)

    context = {
        "all_objects": rows,
        "field_names": display_fields,
        "table_name": table_name,
        "title": table["title"],
    }
    # Check if the request is coming from HTMX (for partial table response)
    if request.headers.get("HX-Request"):
        return render(request, "partials/dynamic_table.html", context)

    return render(request, "partials/table_view.html", context)


def render_calltree(request, sub_name):
    tree = create_calltree_from_sub(sub_name)

    # retrieve config
    active = get_active_config(request)  # {"data": {...}, "hash": "...", "label": ...}
    cfg_hash = active["hash"]
    active_data = active["data"]
    compute_if_evals_for_hash(cfg_hash, active_data)

    # Resolve subroutine (the "parent" for calltree and the scope for dtype vars)
    subroutine = (
        Subroutines.objects.filter(subroutine_name=sub_name)
        .select_related("module")
        .first()
    )
    # Build transitive reachable set under active IFs
    reachable_ids = reachable_subroutine_ids(subroutine, cfg_hash)
    reachable_names = set(
        Subroutines.objects.filter(subroutine_id__in=reachable_ids).values_list(
            "subroutine_name", flat=True
        )
    )
    pruned_tree = prune_tree(
        tree, hit_set=reachable_names, traversable_set=reachable_names
    )
    annotate_tree(pruned_tree)

    context = {
        "tree": pruned_tree,
        "sub_name": sub_name,
        "statuses": {},
    }
    return render(request, "partials/calltree.html", context)


def debug_pruning(parent_id: int, cfg_hash: str):
    base = SubroutineCalltree.objects.filter(parent_subroutine=parent_id)
    if base.count() == 0:
        return {}
    parent_sub = Subroutines.objects.get(subroutine_id=parent_id)
    kept = filter_calltree_by_hash(base, cfg_hash)
    return {
        "parent_sub": parent_sub.subroutine_name,
        "edges_total": base.count(),
        "edges_kept": kept.count(),
        "children_kept": list(
            kept.values_list("child_subroutine__subroutine_name", flat=True).distinct()
        ),
    }


def sub_view(request, sub_name):
    context = {"sub_name": sub_name}
    return render(request, "sub_view.html", context)


def trace_view(request, var_name):
    context = {"var_name": var_name}
    return render(request, "trace_view.html", context)


def type_details(request, type_name):
    context = {"type_name": type_name, "details": "Testing type details"}

    return render(
        request,
        "partials/type_details.html",
        context,
    )


def subroutine_details(request, sub_name: str):
    # Active config (preset or user)
    active = get_active_config(request)  # {"data": {...}, "hash": "...", "label": ...}
    cfg_hash = active["hash"]
    active_data = active["data"].copy()
    num = compute_if_evals_for_hash(cfg_hash, active_data)
    num = propagate_bindings(cfg_hash)

    # Resolve subroutine (the "parent" for calltree and the scope for dtype vars)
    subroutine: Subroutines = (
        Subroutines.objects.filter(subroutine_name=sub_name)
        .select_related("module")
        .first()
    )
    if not subroutine:
        return HttpResponse(b"Subroutine not found.", status=404)

    module = subroutine.module.module_name

    reachable_ids = reachable_subroutine_ids(subroutine, cfg_hash)
    # for id in reachable_ids:
    #     debug_inactive_ifs(sub_id=id, config_hash=cfg_hash)

    # reachable_names = set(
    #     Subroutines.objects.filter(subroutine_id__in=reachable_ids).values_list(
    #         "subroutine_name", flat=True
    #     )
    # )
    # ---------- Calltree (filter out edges in inactive IF ranges) ----------
    # inactive = any IfEvaluationByHash range (for this parent) that covers the edge.lineno and is_active == False
    # --- Calltree (parent == this subroutine) ---
    calltree_qs = (
        SubroutineCalltree.objects.filter(parent_subroutine=subroutine)
        .select_related("child_subroutine__module")
        .order_by("lineno", "child_subroutine__subroutine_name")
    )
    calltree_qs = filter_calltree_by_hash(calltree_qs, cfg_hash)

    # Preserve duplicates & order; include lineno for UI
    callees = [
        f"L{edge.lineno} call {edge.child_subroutine.module.module_name}::{edge.child_subroutine.subroutine_name}"
        for edge in calltree_qs
    ]

    # When SubroutineElmtypesByConfig is populated, the config ifs are already filtered
    dtype_qs = SubroutineElmtypesByConfig.objects.filter(subroutine=subroutine).values(
        "instance__instance_type__user_type_name",
        "instance__instance_name",
        "member__member_type",
        "member__member_name",
        "status",
        "ln",
        "subroutine_id",
    )
    # dtype_qs = filter_access_lns_by_hash(
    #     qs=dtype_qs,
    #     sub_field="subroutine_id",
    #     lineno_field="ln",
    #     config_hash=cfg_hash,
    # ).order_by("ln")

    groups = defaultdict(list)
    share_ln = defaultdict(list)
    for row in dtype_qs.iterator(chunk_size=1000):
        type_name = row["instance__instance_type__user_type_name"]
        instance = row["instance__instance_name"]
        if "c13" in instance:
            print(row)
        member_type = row["member__member_type"]
        member_name = row["member__member_name"]
        status = row["status"]
        ln = row["ln"]
        if member_name is not None:
            groups[(instance, type_name)].append((member_type, member_name, status, ln))
            share_ln[ln].append((instance, member_name, status))

    def type_label(x: str):
        if x not in FORTRAN_TYPES:
            return f"type({x})"
        return x

    args = [
        (type_label(v.arg_type), v.arg_name, v.dim)
        for v in subroutine.subroutine_args.all()
    ]

    # propagated_vars = get_propagated_elmtypes(
    #     reachable_ids=reachable_ids,
    #     root_sub_id=subroutine.subroutine_id,
    #     cfg=cfg_hash,
    # )
    #
    # instance_set: set[str] = {x.split("%", 1)[0] for x in propagated_vars}
    # instances_qs = UserTypeInstances.objects.filter(
    #     instance_name__in=instance_set
    # ).select_related("instance_type")

    # instance_qs_by_name = {obj.instance_name: obj for obj in instances_qs}
    # needed_pairs = set()
    # for pair in propagated_vars.keys():
    #     inst, member_path = pair.split("%", 1)
    #     if not member_path.strip():
    #         continue
    #     inst_obj = instance_qs_by_name[inst]
    #     needed_pairs.add((inst_obj.instance_type.user_type_id, member_path))
    # needed_type_ids = {x[0] for x in needed_pairs}
    # needed_members = {x[1] for x in needed_pairs}
    #
    # typedef_qs = TypeDefinitions.objects.filter(
    #     user_type__user_type_id__in=needed_type_ids,
    #     member_name__in=needed_members,
    # ).select_related("user_type")

    # type_by_id = {
    #     (obj.user_type.user_type_id, obj.member_name): obj for obj in typedef_qs
    # }
    # for pair, statuses in propagated_vars.items():
    #     inst, member_name = pair.split("%", 1)
    #     if not member_name.strip():
    #         continue
    #     type_id = instance_qs_by_name[inst].instance_type.user_type_id
    #     type_obj = type_by_id[(type_id, member_name)]
    #     type_name = type_obj.user_type.user_type_name
    #     member_type = type_obj.member_type
    #     for ln, status, _ in statuses:
    #         groups[(inst, type_name)].append((member_type, member_name, status, ln))

    grouped = []
    for instance, type_name in sorted(groups.keys(), key=lambda k: (k[0], k[1])):
        entries = sorted(groups[(instance, type_name)], key=lambda x: x[1])
        sts_by_member = defaultdict(list)
        for ent in entries:
            sts_by_member[f"{ent[0]}::{ent[1]}"].append(ent[2])
        overall_sts_by_member = {
            name: combine_many_statuses(sts) for name, sts in sts_by_member.items()
        }
        grouped.append(
            {
                "instance": instance,
                "type_name": type_name,
                "members": [
                    {
                        "type": key.split("::")[0],
                        "name": key.split("::")[1],
                        "status": rw,
                    }
                    for key, rw in overall_sts_by_member.items()
                ],
            }
        )

    globals_prefetch = Prefetch(
        "subroutine_global_vars",
        queryset=(
            SubroutineIntrinsicGlobals.objects.select_related(
                "gv_id", "gv_id__gv_module"
            )
            .prefetch_related("gv_id__namelist")
            .order_by("gv_id__var_name")
        ),
    )
    # Pull links with the prefetch now
    sub_with_globals = (
        Subroutines.objects.filter(pk=subroutine.pk)
        .prefetch_related(globals_prefetch)
        .first()
    )

    globals_nml, globals_non = [], []
    for link in sub_with_globals.subroutine_global_vars.all():
        g = link.gv_id
        label = f"{g.var_type} {g.var_name}"
        if list(g.namelist.all()):
            globals_nml.append(label)
        else:
            globals_non.append(label)

    context = {
        "subroutine": subroutine.subroutine_name,
        "module": module,
        "args": args,
        "callees": callees,
        "dtype_var_groups": grouped,
        "globals": globals_non,
        "nml_vars": globals_nml,
    }

    return render(request, "partials/subroutine_details.html", context)


def home(request):
    return redirect("sub_view", sub_name="elm_drv")


def trace_vars(request, key: str):
    """ """

    active = get_active_config(request)
    cfg_hash = active["hash"]

    # Compute reachable set (under active IFs) from your fixed root
    root_sub = "elm_drv"
    root_obj = Subroutines.objects.get(subroutine_name=root_sub)
    reachable_ids = reachable_subroutine_ids(root_obj, cfg_hash)

    reachable_names = set(
        Subroutines.objects.filter(subroutine_id__in=reachable_ids).values_list(
            "subroutine_name", flat=True
        )
    )

    if "%" in key:
        status_by_sub = trace_dtype_var(key, cfg_hash, reachable_ids, reachable_names)
        hitset = set(status_by_sub.keys())
        inst_name, member_name = key.split("%", 1)
        member_name = "%" + member_name
    else:
        hitset = trace_intrinsic_var(key)
        inst_name = key
        member_name = ""
        status_by_sub = {}

    full_tree = retrieve_cache_tree(sub_names=["elm_drv"])
    if full_tree is None:
        full_tree = create_calltree_from_sub("elm_drv")
        cache_tree(full_tree)

    pruned_tree = prune_tree(full_tree, hitset)
    annotate_tree(pruned_tree)

    return render(
        request,
        "partials/trace_details/trace_tree.html",
        {
            "tree": pruned_tree,
            "instance": inst_name,
            "member": member_name,
            "statuses": status_by_sub,
        },
    )


def trace_intrinsic_var(key: str):
    var_type = ""
    if len(key.split()) == 2:
        var_type, var_name = key.split()
    else:
        var_name = key
    var_obj = IntrinsicGlobals.objects.get(var_name=var_name)
    rows_qs = (
        SubroutineIntrinsicGlobals.objects.filter(gv_id=var_obj)
        .select_related("subroutine")
        .values_list("sub_id__subroutine_name")
        .distinct()
    )
    return {x[0] for x in rows_qs}


def trace_dtype_var(
    key: str, cfg_hash: str, reachable_ids: set[int], reachable_names: set[str]
) -> dict:
    """ """

    status_by_sub = {}
    inst_name, member_name = key.split("%", 1)
    # instance = get_object_or_404(UserTypeInstances, instance_name=inst_name)
    # member = get_object_or_404(
    #     TypeDefinitions,
    #     member_name=member_name,
    #     user_type=instance.instance_type,
    # )


    arg_qs = (
        ArgAccess.objects.filter(
            subroutine__subroutine_id__in=reachable_ids,
            member_path=member_name,
        )
        .values_list("subroutine__subroutine_name", "ln")
        .distinct()
    )
    member_access_by_sub: dict[str, set[int]] = defaultdict(set)
    for t in arg_qs:
        member_access_by_sub[t[0]].add(t[1])

    # Rows where this member is referenced AND its ln is NOT covered by inactive IFs
    rows_qs = (
        SubroutineElmtypesByConfig.objects.filter(var_name=inst_name, member_path=member_name)
        .select_related("subroutine")
        .order_by("subroutine__subroutine_name", "ln")
    )
    # rows_qs = filter_access_lns_by_hash(
    #     qs=rows_qs,
    #     sub_field="subroutine_id",
    #     lineno_field="ln",
    #     config_hash=cfg_hash,
    # ).order_by("ln")

    triples = rows_qs.values_list(
        "subroutine__subroutine_name",
        "status",
        "ln",
    ).distinct()

    # Aggregate in Python
    tmp_statuses = defaultdict(list)  # sub -> [status,...]

    for sub_name, st, ln in triples:
        if sub_name not in reachable_names:
            continue
        tmp_statuses[sub_name].append(st)

    for sub_name, sts in tmp_statuses.items():
        status_by_sub[sub_name] = combine_many_statuses(sts)

    return status_by_sub


def _infer_dtype(var_type: str) -> str:
    """
    Map your IntrinsicGlobals.var_type strings to a coarse dtype.
    Adjust mapping to your real values (Fortran types, etc.).
    """
    t = (var_type or "").lower()
    if any(k in t for k in ["logical", "bool"]):
        return "bool"
    if any(k in t for k in ["int", "integer"]):
        return "int"
    if any(k in t for k in ["real", "double", "float"]):
        return "float"
    return "str"


def _coerce_value(dtype: str, raw: str):
    if dtype == "bool":
        return str(raw).lower() in {"1", "true", "yes", "on", "y", "t"}
    if dtype == "int":
        return int(raw)
    if dtype == "float":
        return float(raw)
    return str(raw)


@login_required
def config_editor(request, config_id: int):
    config = get_object_or_404(ConfigProfile, id=config_id, owner=request.user)
    # Pull all configurable vars
    nml = NamelistVariable.objects.select_related(
        "active_var_id",
        "active_var_id__gv_module",
    ).all()

    rows = []
    data = config.data or {}
    for nv in nml:
        ig: IntrinsicGlobals = nv.active_var_id
        dtype = _infer_dtype(str(ig.var_type))
        key = str(ig.var_id)  # use var_id as JSON key to avoid rename issues
        # Fallback to intrinsic default/value if not overridden
        raw = str(ig.value)
        if "=" in raw:
            raw = raw.split("=")[1]
        current = data.get(key, raw)

        # Normalize checkbox truthy value for bool UI
        if dtype == "bool":
            current = str(current).lower() in {"1", "true", "yes", "on", "y", "t"}

        rows.append(
            {
                "ig": ig,
                "key": key,
                "dtype": dtype,
                "current": current,
            }
        )

    return render(
        request,
        "configs/editor.html",
        {
            "config": config,
            "rows": rows,
        },
    )


@login_required
@transaction.atomic
def set_config_value(request, config_id: int, var_id: int):
    """
    HTMX endpoint to set a single config value. Expects form field 'value'.
     - Coerces to the right type based on IntrinsicGlobals.var_type
     - Saves into ConfigProfile.data JSON by var_id
     - Returns the updated row partial for HTMX swap, including error message if coercion fails.
    """
    if request.method != "POST":
        return HttpResponseBadRequest(b"POST required")

    config = get_object_or_404(ConfigProfile, id=config_id, owner=request.user)
    ig = get_object_or_404(IntrinsicGlobals, var_id=var_id)

    dtype = _infer_dtype(ig.var_type)
    raw = request.POST.get("value", "")

    try:
        # Special case checkbox unchecked: HTMX won't send "value"
        if dtype == "bool" and "value" not in request.POST:
            coerced = False
        else:
            coerced = _coerce_value(dtype, raw)
    except Exception as e:
        # Re-render the row with error message
        return render(
            request,
            "configs/row.html",
            {
                "config": config,
                "ig": ig,
                "key": str(ig.var_id),
                "dtype": dtype,
                "current": raw,
                "error": str(e),
            },
            status=400,
        )

    # Save override into JSON by var_id
    data = dict(config.data or {})
    data[str(ig.var_id)] = coerced
    config.data = data
    config.save(update_fields=["data"])

    # Return the updated row (HTMX swap)
    return render(
        request,
        "configs/row.html",
        {
            "config": config,
            "ig": ig,
            "key": str(ig.var_id),
            "dtype": dtype,
            "current": coerced,
        },
    )


@require_http_methods(["POST"])
def recompute_if_evals(request, config_id=None):
    """
    If config_id is provided: recompute for that user config (must belong to the user).
    Otherwise: recompute for the current active config (preset or user). If nothing
    is active, get_active_config() should fall back to the default preset.
    """
    if config_id is not None:
        # lock to owner; if you want admins to recompute any, relax this check
        cfg = get_object_or_404(ConfigProfile, id=config_id, owner=request.user)
        data = cfg.data or {}
        cfg_hash = cfg.config_hash
        label = cfg.name
    else:
        active = get_active_config(
            request
        )  # {"data": {...}, "hash": "...", "label": "..."}
        data = active["data"] or {}
        cfg_hash = active["hash"]
        label = active["label"]
        if not cfg_hash:
            return HttpResponseBadRequest("No active configuration to recompute.")

    written = compute_if_evals_for_hash(active["hash"], active["data"])
    return HttpResponse(f"Recomputed {written} IFs for {active['label']}")


def config_start(request):
    if not request.user.is_authenticated:
        # send to login, and return here after
        login_url = f"{settings.LOGIN_URL}?next=/configs/"
        return redirect(login_url)
    cfg, _ = ConfigProfile.objects.get_or_create(
        owner=request.user, name="Default", defaults={"data": {}}
    )
    return redirect("config_editor", config_id=cfg.id)


@require_http_methods(["POST"])
def select_config_htmx(request):
    """
    HTMX endpoint. Expects form field 'config' with value:
      - 'preset:<slug>' OR
      - 'user:<id>'
    Sets session and returns the updated picker partial.
    Also triggers a global 'configChanged' event with the new hash+label.
    """
    raw = request.POST.get("config", "")
    if raw.startswith("preset:"):
        slug = raw.split(":", 1)[1]
        p = PresetConfig.objects.filter(slug=slug).first()
        if not p:
            return HttpResponseBadRequest(b"Unknown preset")
        request.session["active_config"] = {"type": "preset", "slug": p.slug}

    elif raw.startswith("user:"):
        try:
            cid = int(raw.split(":", 1)[1])
        except ValueError:
            return HttpResponseBadRequest(b"Bad user config id")
        c = ConfigProfile.objects.filter(id=cid, owner=request.user).first()
        if not c:
            return HttpResponseBadRequest(b"Config not found or not yours")
        request.session["active_config"] = {"type": "user", "id": c.id}

    else:
        return HttpResponseBadRequest(b"Bad value")

    # Re-render the picker
    resp = config_picker(request)

    # Add HX-Trigger so other parts of the page can refresh
    active = get_active_config(request)
    trigger = {
        "configChanged": {
            "hash": active["hash"],
            "label": active["label"],
            "kind": active["kind"],
        }
    }
    if isinstance(resp, HttpResponse):
        resp["HX-Trigger"] = json.dumps(trigger)
    return resp


def config_picker(request):
    """Return the dropdown partial with current active + all options."""
    active = get_active_config(request)
    presets = PresetConfig.objects.order_by("-is_default", "name")
    user_configs = (
        ConfigProfile.objects.filter(owner=request.user).order_by("name")
        if request.user.is_authenticated
        else []
    )
    preset_tag = [(p, active["label"] == p.name) for p in presets]
    user_cfgs = [(c, active["label"] == c.name) for c in user_configs]

    return render(
        request,
        "configs/selector.html",
        {
            "active": active,
            "presets": preset_tag,
            "user_configs": user_cfgs,
        },
    )


def select_preset(request, slug):
    preset = get_object_or_404(PresetConfig, slug=slug)
    request.session["active_config"] = {"type": "preset", "slug": preset.slug}
    return redirect(request.GET.get("next") or request.META.get("HTTP_REFERER") or "/")


@login_required
def select_user_config(request, config_id):
    cfg = get_object_or_404(ConfigProfile, id=config_id, owner=request.user)
    request.session["active_config"] = {"type": "user", "id": cfg.id}
    num = compute_if_evals_for_hash()
    return redirect(request.GET.get("next") or request.META.get("HTTP_REFERER") or "/")


class SignupView(View):
    template_name = "registration/signup.html"

    def get(self, request):
        if not getattr(settings, "REGISTRATION_OPEN", True):
            messages.error(request, "Registration is closed.")
            return redirect("login")
        return render(request, self.template_name, {"form": UserCreationForm()})

    def post(self, request):
        if not getattr(settings, "REGISTRATION_OPEN", True):
            messages.error(request, "Registration is closed.")
            return redirect("login")

        invite = request.POST.get("invite_code", "").strip()
        required = getattr(settings, "INVITE_CODE", "")
        if required and invite != required:
            form = UserCreationForm(request.POST)
            form.add_error(None, "Invalid invite code.")
            return render(request, self.template_name, {"form": form})

        form = UserCreationForm(request.POST)
        if not form.is_valid():
            return render(request, self.template_name, {"form": form})

        user = form.save()
        login(request, user)  # auto-login
        messages.success(request, "Welcome!")
        next_url = request.GET.get("next") or settings.LOGIN_REDIRECT_URL or "/"
        return redirect(next_url)


def nml_values(request):
    active = get_active_config(request)
    data = active["data"] or {}

    nml = NamelistVariable.objects.select_related(
        "active_var_id", "active_var_id__gv_module"
    ).all()
    rows = []
    for nv in nml:
        ig: IntrinsicGlobals = nv.active_var_id
        key = str(ig.var_name)
        effective = data.get(key, ig.value)
        source = "config" if key in data else "default"
        rows.append(
            {
                "module": ig.gv_module.module_name,
                "name": ig.var_name,
                "value": effective,
                "source": source,
            }
        )

    rows.sort(key=lambda r: (r["module"].lower(), r["name"].lower()))
    config_rows = [r for r in rows if r["source"] == "config"]
    default_rows = [r for r in rows if r["source"] == "default"]

    return render(
        request,
        "configs/values_modal.html",
        {
            "active_label": active["label"],
            "active_kind": active["kind"],
            "config_rows": config_rows,
            "default_rows": default_rows,
        },
    )
