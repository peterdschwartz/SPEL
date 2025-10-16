import json
from collections import defaultdict, deque
from functools import reduce

from django import template
from django.conf import settings
from django.contrib import messages
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
from django.db import connection, models, transaction
from django.db.models.query import Prefetch
from django.http import HttpResponseBadRequest
from django.shortcuts import HttpResponse, get_object_or_404, redirect, render
from django.views import View
from django.views.decorators.http import require_http_methods

from .calltree import (create_calltree_from_sub, get_calltree_for_var,
                       prune_tree)
from .gen_html import build_tree_html
from .models import (ConfigProfile, FlatIf, FlatIfNamelistVar,
                     IfEvaluationByHash, IntrinsicGlobals, ModuleDependency,
                     Modules, NamelistVariable, PresetConfig,
                     SubroutineActiveGlobalVars, SubroutineArgs,
                     SubroutineCalltree, SubroutineIntrinsicGlobals,
                     Subroutines, TypeDefinitions, UserTypeInstances)
from .signals import cache_tree, retrieve_cache_tree
from .utils.configs import get_active_config
from .utils.ifs import (filter_calltree_by_hash, filter_dtype_vars_by_hash,
                        recompute_if_evals_for_hash)

register = template.Library()
# import module_calltree
TYPE_DEFAULT_DICT = {
    "id": "",
    "module": "",
    "type_name": "",
    "member": "",
    "member_type": "",
    "dim": "",
    "bounds": "",
    "active": "",
}

MODS_DEFAULT_DICT = {
    "id": "",
    "subroutine": "",
    "variable_name": "",
    "status": "",
}

MOD_DEPENDENCY_DEFAULT_DICT = {
    "id": "",
    "module_name": "",
    "dependency": "",
    "object_used": "",
}

VARS_DEFAULT_DICT = {
    "id": "",
    "module": "",
    "name": "",
    "type": "",
    "dim": "",
}

TABLE_NAME_LOOKUP = {
    "subroutine_active_global_vars": MODS_DEFAULT_DICT,
    "user_types": TYPE_DEFAULT_DICT,
    "module_dependency": MOD_DEPENDENCY_DEFAULT_DICT,
    "variables": VARS_DEFAULT_DICT,
}

VIEWS_TABLE_DICT = {
    "subroutines": {
        "name": Subroutines,
        "html": "subroutines.html",
        "fields": {
            "Id": "subroutine_id",
            "Module": "module.module_name",
            "Subroutine": "subroutine_name",
        },
        "title": "Table of Subroutines",
    },
    "modules": {
        "name": Modules,
        "html": "modules.html",
        "fields": {
            "Id": "module_id",
            "Module": "module_name",
        },
        "title": "Table of Modules",
    },
    "subroutine_calltree": {
        "name": SubroutineCalltree,
        "html": "subroutine_calltree.html",
        "fields": {
            "Id": "parent_id",
            "Parent Sub": "parent_subroutine.subroutine_name",
            "Child Sub": "child_subroutine.subroutine_name",
            "Lineno": "lineno",
        },
        "title": "Table of Subroutine Call Tree",
    },
    "types": {
        "name": TypeDefinitions,
        "html": "types.html",
        "fields": {
            "Id": "define_id",
            "Module": "type_module.module_name",
            "Type Name": "user_type.user_type_name",
            "Member Type": "member_type",
            "Member Name": "member_name",
            "Dim": "dim",
            "Bounds": "bounds",
        },
        "title": "Table of Type Definitions",
    },
    "dependency": {
        "name": ModuleDependency,
        "html": "dep.html",
        "fields": {
            "Id": "dependency_id",
            "Module": "module.module_name",
            "Dependent Mod": "dep_module.module_name",
            "Used object": "object_used",
        },
        "title": "Table of Module Dependencies",
    },
    "instances": {
        "name": UserTypeInstances,
        "html": "instances.html",
        "fields": {
            "Id": "instance_id",
            "Module": "instance_type.module.module_name",
            "Type Name": "instance_type.user_type_name",
            "Instance Name": "instance_name",
        },
        "title": "Table of User Type Instances",
    },
    "subroutineargs": {
        "name": SubroutineArgs,
        "html": "subroutineargs.html",
        "fields": {
            "Id": "arg_id",
            "Subroutine": "subroutine.subroutine_name",
            "Arg Type": "arg_type",
            "Arg Name": "arg_name",
            "Dim": "dim",
        },
        "title": "Table of Subroutine Arguments",
    },
    "activeglobalvars": {
        "name": SubroutineActiveGlobalVars,
        "html": "active_global_vars.html",
        "fields": {
            "Id": "variable_id",
            "Subroutine": "subroutine.subroutine_name",
            "Inst": "instance.instance_name",
            "Member": "member.member_name",
            "Status": "status",
            "ln": "ln",
        },
        "title": "Table of Global Vars by Subroutine",
    },
    "intrinsicglobalvars": {
        "name": IntrinsicGlobals,
        "html": "intrinsic_gvs.html",
        "fields": {
            "Id": "var_id",
            "Module": "gv_module.module",
            "Type": "var_type",
            "Name": "var_name",
            "Dim": "dim",
            "Bounds": "bounds",
            "Default": "value",
        },
        "title": "Table of Intrinsic type Globals",
    },
    "subroutineintrinsicglobals": {
        "name": SubroutineIntrinsicGlobals,
        "html": "active_intrinsics.html",
        "fields": {
            "Id": "sub_gv_id",
            "Module": "sub_id.module.module_name",
            "Subroutine": "sub_id.subroutine_name",
            "Type": "gv_id.var_type",
            "Name": "gv_id.var_name",
        },
        "title": "Table of Active Intrinsic Globals",
    },
    "namelistvars": {
        "name": NamelistVariable,
        "html": "namelist",
        "fields": {
            "Id": "nml_id",
            "Module": "active_var_id.gv_module.module_name",
            "Type": "active_var_id.var_type",
            "Name": "active_var_id.var_name",
            "Dim": "active_var_id.dim",
            "Bounds": "active_var_id.bounds",
            "Default": "active_var_id.value",
        },
        "title": "Table of Namelist Variables",
    },
    "flat_ifs": {
        "name": FlatIf,
        "html": "flat_ifs",
        "fields": {
            "Id": "flatif_id",
            "Module": "subroutine.module.module_name",
            "Subroutine": "subroutine.subroutine_name",
            "Start": "start_ln",
            "End": "end_ln",
            "Cond.": "condition",
        },
        "title": "Table of If Blocks",
    },
    "if_nml": {
        "name": FlatIfNamelistVar,
        "html": "if_nml",
        "fields": {
            "Id": "id",
            "Module": "flatif.subroutine.module.module_name",
            "Subroutine": "flatif.subroutine.subroutine_name",
            "Start": "flatif.start_ln",
            "End": "flatif.end_ln",
            "Name": "namelist_var.active_var_id.var_name",
            "Default": "namelist_var.active_var_id.value",
            "Cond.": "flatif.condition",
        },
        "title": "Table of Ifs with NML Conditions",
    },
}


def execute(statement):
    with connection.cursor() as cur:
        cur.execute(statement)


def modules_calltree(request):
    if request.method == "POST":
        data = request.POST.get("mod")
        tree = get_module_calltree(data)
    else:
        return render(request, "modules_calltree.html", {})

    return render(request, "modules_calltree.html", {"tree": tree})


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
    tree_html = build_tree_html(tree)
    context = {"tree": tree_html}
    return render(request, "partials/calltree.html", context)


def sub_view(request, sub_name):
    context = {"sub_name": sub_name}
    return render(request, "sub_view.html", context)


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

    # (Optional) lazily populate IF cache if missing
    # if not IfEvaluationByHash.objects.filter(config_hash=cfg_hash).exists():
        # Safe to skip if you prefer explicit recompute via button
    recompute_if_evals_for_hash(cfg_hash, active["data"])

    # Resolve subroutine (the "parent" for calltree and the scope for dtype vars)
    subroutine = (
        Subroutines.objects.filter(subroutine_name=sub_name)
        .select_related("module")
        .first()
    )
    if not subroutine:
        return HttpResponse(b"Subroutine not found.", status=404)

    module = subroutine.module.module_name

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

    # --- DType vars (scoped to this subroutine) ---
    dtype_qs = (
        SubroutineActiveGlobalVars.objects.filter(subroutine=subroutine)
        .select_related("instance__instance_type", "member")
        .order_by("ln")
    )
    dtype_qs = filter_dtype_vars_by_hash(dtype_qs, cfg_hash)

    args = [(v.arg_type, v.arg_name, v.dim) for v in subroutine.subroutine_args.all()]

    dtype_vars = [
        (
            v.instance.instance_type.user_type_name,
            v.instance.instance_name,
            v.member.member_type,
            v.member.member_name,
            v.status,
        )
        for v in subroutine.subroutine_dtype_vars.all()
    ]
    dtype_vars.sort(key=lambda x: x[1])
    groups = defaultdict(list)
    for type_name, instance, member_type, member_name, status in dtype_vars:
        groups[(instance, type_name)].append((member_name, status, member_type))

    # order: instance asc, then member asc
    grouped = []
    for instance, type_name in sorted(groups.keys(), key=lambda k: (k[0], k[1])):
        entries = sorted(groups[(instance, type_name)], key=lambda x: x[0])
        grouped.append(
            {
                "instance": instance,  # e.g. 'veg_ef'
                "type_name": type_name,  # e.g. 'vegetation_energy_flux'
                "members": [
                    {"type": mt, "name": m, "status": s} for m, s, mt in entries
                ],
            }
        )

    # ---------- Globals / namelist flags (unchanged build) ----------
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


def fake(request, table):
    table = VIEWS_TABLE_DICT[table]
    # print(table["name"])
    return render(request, "query_variables.html", {"table": table["dict"]})


def trace_dtype_var(request, key: str):
    """ """
    # Active config
    active = get_active_config(request)
    cfg_hash = active["hash"]

    # Optionally ensure cache exists (if you don't always click 'Recompute IFs'):
    # ensure_if_cache(cfg_hash, active["data"] or {})

    inst_name = member_name = None

    # Compute reachable set (under active IFs) from your fixed root
    root_sub = "elm_drv"
    reachable = _reachable_subroutines_from(root_sub, cfg_hash)
    status_by_sub = {}
    if "%" in key:
        inst_name, member_name = key.split("%", 1)
        instance = get_object_or_404(UserTypeInstances, instance_name=inst_name)
        member = get_object_or_404(
            TypeDefinitions,
            member_name=member_name,
            user_type=instance.instance_type,
        )

        # Rows where this member is referenced AND its ln is NOT covered by inactive IFs
        rows_qs = (
            SubroutineActiveGlobalVars.objects.filter(instance=instance, member=member)
            .select_related("subroutine")
            .order_by("subroutine__subroutine_name" ,"ln")
        )
        rows_qs = filter_dtype_vars_by_hash(rows_qs, cfg_hash)
        triples = rows_qs.values_list("subroutine__subroutine_name", "status", "ln").distinct()

        # Aggregate in Python
        tmp_statuses = defaultdict(list)  # sub -> [status,...]

        for sub_name, st, ln in triples:
            if sub_name not in reachable:
                continue
            tmp_statuses[sub_name].append(st)

        # Compute overall status per sub
        for sub_name, sts in tmp_statuses.items():
            overall = combine_many_statuses(sts, _combine_status)
            status_by_sub[sub_name] = overall
            # (optional) sort lines by ln; they’re already ordered, but ensure stability:
        hitset = set(status_by_sub.keys())
    else:
        var_obj = get_object_or_404(IntrinsicGlobals, var_name=key)
        rows_qs = (
            SubroutineIntrinsicGlobals.objects.filter(gv_id=var_obj)
            .select_related("subroutine")
            .values_list("sub_id__subroutine_name")
            .distinct()
        )
        hitset = set(rows_qs)

    full_tree = retrieve_cache_tree(sub_names=["elm_drv"])
    if full_tree is None:
        full_tree = create_calltree_from_sub("elm_drv")
        cache_tree(full_tree)

    pruned_tree = prune_tree(full_tree, hitset)

    return render(
        request,
        "partials/trace_details/trace_tree.html",
        {
            "tree": pruned_tree,
            "instance": inst_name,
            "member": member_name,
        },
    )


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
        "active_var_id", "active_var_id__gv_module"
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

    written = recompute_if_evals_for_hash(active["hash"], active["data"])
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
    print(f"presets: {preset_tag}")

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
    active = get_active_config(
        request
    )  # {"data": {...}, "label": "...", "kind": "..."}
    data = active["data"] or {}

    nml = NamelistVariable.objects.select_related(
        "active_var_id", "active_var_id__gv_module"
    ).all()

    rows = []
    for nv in nml:
        ig: IntrinsicGlobals = nv.active_var_id
        key = str(ig.var_id)
        effective = data.get(key, ig.value)
        source = "config" if key in data else "default"
        rows.append(
            {
                "module": ig.gv_module.module_name,
                "name": ig.var_name,
                "value": effective,
                "source": source,  # "config" or "default"
            }
        )

    rows.sort(key=lambda r: (r["module"].lower(), r["name"].lower()))

    return render(
        request,
        "configs/values_modal.html",
        {
            "active_label": active["label"],
            "active_kind": active["kind"],
            "rows": rows,
            "json_payload": data,
        },
    )


# --- small helper: compute reachable subroutines under active IFs ---
def _reachable_subroutines_from(root_sub_name: str, cfg_hash: str) -> set[str]:
    """BFS using only calltree edges that are NOT inside inactive IF ranges."""
    # start node
    root = (
        Subroutines.objects.filter(subroutine_name=root_sub_name)
        .only("subroutine_id", "subroutine_name")
        .first()
    )
    print(f"root: {root}")
    if not root:
        return set()

    # active edges out of any parent
    edges = SubroutineCalltree.objects.select_related(
        "parent_subroutine", "child_subroutine"
    ).order_by("parent_subroutine_id", "lineno")
    edges = filter_calltree_by_hash(edges, cfg_hash)
    print(f"edges: {edges}")

    # build adjacency list by parent id
    adj = defaultdict(list)
    name_by_id = {}
    for e in edges:
        adj[e.parent_subroutine_id].append(e.child_subroutine_id)
        name_by_id[e.parent_subroutine_id] = e.parent_subroutine.subroutine_name
        name_by_id[e.child_subroutine_id] = e.child_subroutine.subroutine_name

    # BFS
    seen_ids = set([root.subroutine_id])
    q = deque([root.subroutine_id])
    while q:
        u = q.popleft()
        for v in adj.get(u, []):
            if v not in seen_ids:
                seen_ids.add(v)
                q.append(v)

    # map back to names
    reachable = {name_by_id.get(i) for i in seen_ids if name_by_id.get(i)}
    return reachable

def combine_many_statuses(statuses, combine_status):
    """Reduce a list like ['r','rw','w'] -> overall (e.g., 'rw')."""
    if not statuses:
        return ""
    return reduce(combine_status, statuses)

def _combine_status(s1: str, s2: str) -> str:
    """
    # Convert each status string to a set of permissions
    # Return 'rw' if both permissions are present; otherwise 'r' or 'w'
    """
    perms = set(s1) | set(s2)
    return "".join(sorted(perms))
