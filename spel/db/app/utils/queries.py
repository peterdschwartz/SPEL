"""
Read-side helpers shared by the views: the step tree, variable references,
config-filtered accesses and per-configuration categories. Everything that
walks the whole trace is cached per (snapshot, configuration hash).
"""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional

from django.core.cache import cache
from django.db.models import Q

from spel.scripts.db_export.filters import filter_members, filters_overlap  # noqa: F401

from ..models import (
    FieldState,
    UserTypes,
    IntrinsicGlobals,
    Snapshot,
    SubroutineActiveGlobalVars,
    SubroutineCalltree,
    SubroutineIntrinsicGlobals,
    Subroutines,
    TraceAccess,
    FilterRelation,
    TraceNode,
    TypeDefinitions,
    UserTypeInstances,
)
from .ifs import Reach, inactive_ranges, is_inactive, reachable

CACHE_SECONDS = 600


# ---------------------------------------------------------------- step tree
@dataclass
class Node:
    order: int
    pk: int
    sub_pk: int
    name: str
    label: str  # module::name
    sub_id: str
    depth: int
    size: int
    cut: bool
    analyzed: bool
    call_ln: Optional[int]
    active: bool = True
    children: list = field(default_factory=list)
    marks: dict = field(default_factory=dict)  # variable key -> status (trace mode)
    count: int = 1  # occurrences merged into this node (same callee under the same parent)
    orders: list = field(default_factory=list)  # trace orders of every merged occurrence

    @property
    def uid(self) -> str:
        return f"n{self.order}"


def load_nodes(snapshot_id: int) -> list[tuple]:
    """(pk, order, parent_order, sub_pk, name, module, sub_id, depth, size, cut, analyzed, call_ln) in order."""
    key = f"nodes:{snapshot_id}"
    rows = cache.get(key)
    if rows is None:
        rows = list(
            TraceNode.objects.filter(snapshot_id=snapshot_id)
            .order_by("order")
            .values_list(
                "pk", "order", "parent__order", "subroutine_id", "subroutine__subroutine_name",
                "subroutine__module__module_name", "subroutine__sub_id", "depth", "size", "cut", "analyzed",
                "call_site__lineno",
            )
        )
        cache.set(key, rows, CACHE_SECONDS)
    return rows


def node_subs(snapshot_id: int) -> dict[int, int]:
    """trace order -> subroutine pk."""
    return {r[1]: r[3] for r in load_nodes(snapshot_id)}


def top_level(snapshot_id: int) -> dict[int, int]:
    """trace order -> order of its depth-1 ancestor (the elm_drv call it belongs to)."""
    out, stack = {}, []
    for row in load_nodes(snapshot_id):
        order, depth = row[1], row[7]
        del stack[depth:]
        stack.append(order)
        out[order] = stack[1] if len(stack) > 1 else order
    return out


def merge_repeats(nodes: list[Node]) -> list[Node]:
    """
    One node per callee under a parent: repeated calls of the same routine are
    merged into the first occurrence (count, orders, children, size), recursively.
    """
    out: list[Node] = []
    first_of: dict[int, Node] = {}
    for n in nodes:
        first = first_of.get(n.sub_pk)
        if first is None:
            first_of[n.sub_pk] = n
            out.append(n)
            continue
        first.count += 1
        first.orders.extend(n.orders)
        first.children.extend(n.children)
        first.size += n.size
        first.active = first.active or n.active
        first.cut = first.cut and n.cut
    for n in out:
        n.children = merge_repeats(n.children)
    return out


def build_tree(
    snapshot_id: int,
    reach: Optional[Reach],
    root_sub_pk: Optional[int] = None,
    root_order: Optional[int] = None,
    keep_inactive: bool = False,
    merge: bool = True,
) -> list[Node]:
    """
    The step tree (or the subtree(s) rooted at `root_sub_pk`'s first occurrence /
    `root_order`), with nodes the configuration does not reach dropped (or
    kept with active=False). With `merge`, repeated calls of one callee under a
    parent become a single node (Node.count / Node.orders).
    """
    rows = load_nodes(snapshot_id)
    nodes: dict[int, Node] = {}
    roots: list[Node] = []
    lo, hi = 0, len(rows)
    if root_order is not None or root_sub_pk is not None:
        start = next((r for r in rows if (r[1] == root_order) or (root_order is None and r[3] == root_sub_pk)), None)
        if start is None:
            return []
        lo, hi = start[1], start[1] + start[8]
    for row in rows:
        pk, order, parent_order, sub_pk, name, module, sub_id, depth, size, cut, analyzed, call_ln = row
        if order < lo or order >= hi:
            continue
        active = reach is None or order in reach.orders
        if not active and not keep_inactive:
            continue
        node = Node(order, pk, sub_pk, name, f"{module or 'lib'}::{name}", sub_id, depth, size, cut, analyzed, call_ln, active, orders=[order])
        nodes[order] = node
        parent = nodes.get(parent_order) if parent_order is not None else None
        if parent is None or order == lo:
            roots.append(node)
        else:
            parent.children.append(node)
    return merge_repeats(roots) if merge else roots


def occurrence_index(snapshot_id: int) -> dict[int, tuple[int, int]]:
    """trace order -> (k, n): the k-th of n calls of that routine by the same parent."""
    groups: dict[tuple, list[int]] = defaultdict(list)
    for row in load_nodes(snapshot_id):
        groups[(row[2], row[3])].append(row[1])
    out = {}
    for orders in groups.values():
        for k, o in enumerate(orders, 1):
            out[o] = (k, len(orders))
    return out


MAX_EDGE_VARS = 40
# Routines the box diagram hides by default: book-keeping that clutters the data flow.
BOOKKEEPING = ("summary", "zerodwt", "balance", "get_clump_bounds", "get_proc_bounds", "get_proc_clumps", "interp", "budget", "acc_vars", "accvars")


def dependency_graph(
    snapshot_id: int,
    reach: Optional[Reach],
    ranges: dict,
    root_sub_pk: Optional[int] = None,
    keep_inactive: bool = False,
) -> Optional[dict]:
    """
    The direct callees of a routine (its first occurrence; the step's root by
    default) with the data dependencies between them.

    A callee's *outputs* are the paths written anywhere in its subtree, its
    *inputs* the paths it reads before its subtree writes them. An edge runs from
    an earlier callee to a later one whose inputs meet the earlier's outputs;
    `direct` marks the edges left after transitive reduction (the rest are
    implied by chains of direct ones). `level` is the longest dependency chain
    above a node, the row of the diagram. Calls and accesses the configuration
    prunes are left out; with `keep_inactive` the pruned calls stay as nodes
    (`active=False`) without accesses. `bookkeeping` flags routines matching
    BOOKKEEPING, which the diagram hides by default (it recomputes levels and
    direct edges on the nodes it shows).
    """
    rows = load_nodes(snapshot_id)
    if not rows:
        return None
    start = rows[0] if root_sub_pk is None else next((r for r in rows if r[3] == root_sub_pk), None)
    if start is None:
        return None
    lo, hi = start[1], start[1] + start[8]
    kids = []
    for row in rows:
        pk, order, parent_order, sub_pk, name, module, sub_id, depth, size, cut, analyzed, call_ln = row
        if order <= lo or order >= hi or parent_order != lo:
            continue
        active = reach is None or order in reach.orders
        if not active and not keep_inactive:
            continue
        kids.append({"id": order, "order": order, "end": order + size, "sub": sub_id, "name": name,
                     "label": f"{module or 'lib'}::{name}", "size": size, "cut": cut, "analyzed": analyzed,
                     "active": active, "call_ln": call_ln, "n_children": 0, "sub_pk": sub_pk,
                     "bookkeeping": any(pat in name for pat in BOOKKEEPING)})
    child_orders = {k["order"] for k in kids}
    for row in rows:
        if row[2] in child_orders:
            next(k for k in kids if k["order"] == row[2])["n_children"] += 1
    # k-th of n calls of the same routine among the displayed callees
    per_sub: dict[int, list] = defaultdict(list)
    for k in kids:
        per_sub[k["sub_pk"]].append(k)
    for group in per_sub.values():
        for i, k in enumerate(group, 1):
            k["k"], k["n"] = i, len(group)
    subs = node_subs(snapshot_id)
    starts = [k["order"] for k in kids]
    n = len(kids)
    # path -> the filters ("*" = any element) its reads / writes run under
    writes: list[dict[str, set]] = [defaultdict(set) for _ in kids]
    inputs: list[dict[str, set]] = [defaultdict(set) for _ in kids]
    if kids:
        qs = (
            TraceAccess.objects.filter(snapshot_id=snapshot_id, trace_node__order__gte=starts[0], trace_node__order__lt=hi)
            .order_by("order")
            .values_list("trace_node__order", "path", "status", "ln", "filter")
        )
        for order, path, status, ln, filt in qs:
            i = bisect_right(starts, order) - 1
            k = kids[i]
            if order >= k["end"] or not k["active"]:
                continue
            if reach is not None and order not in reach.orders:
                continue
            if is_inactive(ranges, subs.get(order), ln):
                continue
            key = filter_members(filt)
            if "r" in status and not (path in writes[i] and "*" in writes[i][path]):
                inputs[i][path].add(key)
            if "w" in status:
                writes[i][path].add(key)
    disjoint = disjoint_filters(snapshot_id)
    edges: dict[tuple[int, int], set] = {}
    excluded: dict[tuple[int, int], set] = {}
    succ: dict[int, set] = defaultdict(set)
    for j in range(n):
        for i in range(j):
            shared, dropped = set(), set()
            for path in writes[i].keys() & inputs[j].keys():
                if filters_overlap(writes[i][path], inputs[j][path], disjoint):
                    shared.add(path)
                else:
                    dropped.add(path)
            if shared:
                edges[(i, j)] = shared
                succ[i].add(j)
            if dropped:
                excluded[(i, j)] = dropped
    # transitive reduction: an edge i->j is implied if j is reachable from another successor of i
    downstream: dict[int, set] = {}
    for i in reversed(range(n)):
        acc: set = set()
        for j in succ[i]:
            acc |= {j} | downstream[j]
        downstream[i] = acc
    level = [0] * n
    for j in range(n):
        preds = [i for i in range(j) if (i, j) in edges]
        level[j] = 1 + max((level[i] for i in preds), default=-1)
    for i, k in enumerate(kids):
        k["level"] = level[i]
        k["n_in"], k["n_out"] = len(inputs[i]), len(writes[i])
        del k["end"], k["sub_pk"]
    out_edges = []
    for (i, j), shared in sorted(edges.items()):
        direct = not any(j in downstream[m] for m in succ[i] if m != j)
        names = sorted(shared)
        out_edges.append({"s": kids[i]["id"], "t": kids[j]["id"], "n": len(names), "vars": names[:MAX_EDGE_VARS], "direct": direct,
                          "n_excluded": len(excluded.get((i, j), ()))})
    only_excluded = [{"s": kids[i]["id"], "t": kids[j]["id"], "n": len(v), "vars": sorted(v)[:MAX_EDGE_VARS]}
                     for (i, j), v in sorted(excluded.items()) if (i, j) not in edges]
    root_node = {"order": start[1], "sub": start[6], "name": start[4], "label": f"{start[5] or 'lib'}::{start[4]}", "size": start[8]}
    return {"root": root_node, "nodes": kids, "edges": out_edges, "excluded": only_excluded}


def disjoint_filters(snapshot_id: int) -> set[tuple[str, str]]:
    """Unordered pairs of filter members that cannot share an element (FilterRelation, both orders)."""
    key = f"disjoint:{snapshot_id}"
    out = cache.get(key)
    if out is None:
        out = set()
        for a, b in FilterRelation.objects.filter(snapshot_id=snapshot_id, relation="disjoint").values_list("filter_a", "filter_b"):
            out.add((a, b))
            out.add((b, a))
        cache.set(key, out, CACHE_SECONDS)
    return out


def fuzzy_score(needle: str, text: str) -> Optional[int]:
    """
    Higher is better; None if `needle` is not a subsequence of `text`. Substring
    matches rank first (prefix best), then subsequences with bonuses for runs and
    matches at word boundaries (`%`, `_`, `::`). Case-insensitive.
    """
    needle, text = needle.lower(), text.lower()
    if not needle:
        return 0
    i = text.find(needle)
    if i >= 0:
        return 1000 - 4 * i - len(text) + (300 if i == 0 else 0)
    score, pos, prev = 100, 0, -2
    for ch in needle:
        i = text.find(ch, pos)
        if i < 0:
            return None
        if i == prev + 1:
            score += 6
        if i == 0 or text[i - 1] in "%_:":
            score += 4
        prev, pos = i, i + 1
    return score - len(text) // 3


def fuzzy_filter(needle: str, items: Iterable, key, limit: Optional[int] = None) -> list:
    """Items whose `key(item)` fuzzily matches, best first (stable for equal scores)."""
    scored = []
    for n, item in enumerate(items):
        sc = fuzzy_score(needle, key(item))
        if sc is not None:
            scored.append((-sc, n, item))
    scored.sort(key=lambda t: (t[0], t[1]))
    out = [t[2] for t in scored]
    return out[:limit] if limit else out


def reg_tags(regs: list) -> list[dict]:
    """
    One badge per registration kind (`hist`, `restart`, ...) for a member that
    may be registered several times (per instance, under different guards):
    the count, whether every one is inactive, and a title listing them.
    """
    by_kind: dict[str, list] = {}
    for r in regs:
        by_kind.setdefault(r.kind, []).append(r)
    out = []
    for kind, items in by_kind.items():
        lines = []
        for r in items:
            line = f"{r.name} {r.long_name}".strip() + (f" [{r.units}]" if r.units else "")
            if r.guards:
                line += " if " + " and ".join(r.guards)
            if r.inactive:
                line += " (inactive)"
            lines.append(line)
        out.append({"kind": kind, "n": len(items), "inactive": all(r.inactive for r in items), "title": "\n".join(lines)})
    return out


def member_rows(snapshot: Snapshot, instance: Optional[UserTypeInstances], user_type: UserTypes, needle: str = "") -> list[dict]:
    """
    Rows of a type's members for the picker / type page: component, field state
    of `instance` (if any), registrations, and the trace key.
    """
    from ..models import Registration

    states = {fs.member_id: fs for fs in FieldState.objects.filter(snapshot=snapshot, instance=instance)} if instance else {}
    regs: dict[int, list] = {}
    reg_q = Q(user_type=user_type)
    if instance is not None:
        reg_q |= Q(instance=instance)
    for r in Registration.objects.filter(snapshot=snapshot).filter(reg_q).select_related("member"):
        if r.member_id:
            regs.setdefault(r.member_id, []).append(r)
    members = list(TypeDefinitions.objects.filter(user_type=user_type).select_related("component_type").order_by("position"))
    if needle:
        members = fuzzy_filter(needle, members, lambda m: m.member_name)
    rows = []
    for m in members:
        fs = states.get(m.pk)
        rows.append({
            "m": m, "state": fs, "regs": regs.get(m.pk, []), "tags": reg_tags(regs.get(m.pk, [])),
            "key": f"{instance.instance_name}%{m.member_name}" if instance else "",
            "category": fs.effective_category if fs else "",
        })
    return rows


def first_occurrence(snapshot_id: int, sub_pk: int) -> Optional[int]:
    return next((r[1] for r in load_nodes(snapshot_id) if r[3] == sub_pk), None)


# ---------------------------------------------------------- variable refs
@dataclass
class VarRef:
    key: str  # inst%member or module::name
    label: str
    kind: str  # inst | global
    instance: Optional[UserTypeInstances] = None
    member: Optional[TypeDefinitions] = None
    global_var: Optional[IntrinsicGlobals] = None
    state: Optional[FieldState] = None

    @property
    def filter(self) -> Q:
        if self.kind == "inst":
            return Q(instance=self.instance, member=self.member)
        return Q(global_var=self.global_var)


def parse_var(snapshot: Snapshot, key: str) -> Optional[VarRef]:
    """`inst%member`, `module::inst%member`, `name` or `module::name` -> VarRef, or None."""
    key = (key or "").strip().lower()
    if not key:
        return None
    module, _, rest = key.rpartition("::")
    if "%" in rest:
        inst_name, _, member_path = rest.partition("%")
        insts = UserTypeInstances.objects.filter(inst_module__snapshot=snapshot, instance_name=inst_name)
        if module:
            insts = insts.filter(inst_module__module_name=module)
        inst = insts.select_related("instance_type", "inst_module").first()
        if inst is None:
            return None
        member = TypeDefinitions.objects.filter(user_type=inst.instance_type, member_name=member_path).first()
        if member is None:
            return None
        state = FieldState.objects.filter(snapshot=snapshot, instance=inst, member=member).first()
        return VarRef(f"{inst.instance_name}%{member.member_name}", f"{inst.instance_name}%{member.member_name}", "inst", inst, member, None, state)
    gq = IntrinsicGlobals.objects.filter(snapshot=snapshot, var_name=rest)
    if module:
        gq = gq.filter(gv_module__module_name=module)
    gvar = gq.select_related("gv_module").first()
    if gvar is None:
        return None
    state = FieldState.objects.filter(snapshot=snapshot, global_var=gvar).first()
    return VarRef(f"{gvar.gv_module.module_name}::{gvar.var_name}", gvar.var_name, "global", None, None, gvar, state)


def parse_vars(snapshot: Snapshot, keys: Iterable[str]) -> list[VarRef]:
    out, seen = [], set()
    for k in keys:
        ref = parse_var(snapshot, k)
        if ref is not None and ref.key not in seen:
            seen.add(ref.key)
            out.append(ref)
    return out


# ------------------------------------------------------- config-filtered rows
def trace_rows(snapshot: Snapshot, ref: VarRef, reach: Optional[Reach], ranges: dict) -> list[dict]:
    """TraceAccess rows of one variable under a configuration, in step order."""
    subs = node_subs(snapshot.pk)
    rows = (
        TraceAccess.objects.filter(ref.filter, snapshot=snapshot)
        .order_by("order")
        .values("trace_node__order", "status", "ln", "order", "path", "resolved_from", "conditional")
    )
    out = []
    for r in rows:
        order = r["trace_node__order"]
        if reach is not None and order not in reach.orders:
            continue
        if is_inactive(ranges, subs.get(order), r["ln"]):
            continue
        out.append({"c": order, "s": r["status"], "ln": r["ln"], "order": r["order"], "path": r["path"], "from": r["resolved_from"], "cond": r["conditional"]})
    return out


def arcs_for(marks: list[dict]) -> list[dict]:
    """
    Write -> next read ("when is it needed next"), and the carried arc from the
    last write to the first read when the step reads before it writes.
    """
    arcs = []
    reads = [i for i, m in enumerate(marks) if m["s"] in ("r", "rw")]
    for i, m in enumerate(marks):
        if m["s"] not in ("w", "rw"):
            continue
        nxt = next((j for j in reads if j > i), None)
        if nxt is not None:
            arcs.append({"from": i, "to": nxt, "kind": "next-read"})
    if marks and marks[0]["s"] in ("r", "rw"):
        last_w = max((i for i, m in enumerate(marks) if m["s"] in ("w", "rw")), default=None)
        if last_w is not None:
            arcs.append({"from": last_w, "to": 0, "kind": "carried"})
    return arcs


def categorize(accesses: list[tuple[str, bool]], history: bool) -> str:
    """Same rule as the exporter (spel.scripts.db_export.fieldstate.categorize)."""
    from spel.scripts.db_export.fieldstate import categorize as _categorize

    return _categorize(accesses, history)


def config_categories(snapshot: Snapshot, cfg_hash: str) -> dict[tuple, str]:
    """
    Category of every field/global under a configuration (accesses in inactive
    branches and unreachable routines dropped): ("i", instance_pk, member_pk) or
    ("g", global_pk) -> category.
    """
    key = f"categories:{snapshot.pk}:{cfg_hash}"
    out = cache.get(key)
    if out is not None:
        return out
    reach = reachable(snapshot.pk, cfg_hash)
    ranges = inactive_ranges(snapshot.pk, cfg_hash)
    subs = node_subs(snapshot.pk)
    statuses: dict[tuple, list[tuple[str, bool]]] = defaultdict(list)
    rows = (
        TraceAccess.objects.filter(snapshot=snapshot).order_by("order")
        .values_list("instance_id", "member_id", "global_var_id", "trace_node__order", "status", "ln", "conditional")
    )
    for inst, member, gvar, order, status, ln, cond in rows:
        if order not in reach.orders or is_inactive(ranges, subs.get(order), ln):
            continue
        k = ("i", inst, member) if inst is not None else ("g", gvar)
        if k[0] == "i" and member is None:
            continue
        statuses[k].append((status, cond))
    history = {}
    for inst, member, gvar, hist in FieldState.objects.filter(snapshot=snapshot).values_list("instance_id", "member_id", "global_var_id", "history"):
        history[("i", inst, member) if inst is not None else ("g", gvar)] = hist
    out = {k: categorize(v, history.get(k, False)) for k, v in statuses.items()}
    for k, hist in history.items():
        out.setdefault(k, "unused")
    cache.set(key, out, CACHE_SECONDS)
    return out


def state_key(fs: FieldState) -> tuple:
    return ("i", fs.instance_id, fs.member_id) if fs.instance_id else ("g", fs.global_var_id)


# -------------------------------------------------------- routine details
def merge_status(statuses: Iterable[str]) -> str:
    out = set()
    for s in statuses:
        out |= set(s)
    return "".join(sorted(out))


def routine_accesses(sub: Subroutines, ranges: dict) -> tuple[list[dict], list[dict]]:
    """
    (derived-type groups, intrinsic globals) the routine accesses directly,
    minus lines in inactive if branches. Groups: {instance, type, members: [{name, status, lines, key}]}.
    """
    groups: dict[str, dict] = {}
    rows = (
        SubroutineActiveGlobalVars.objects.filter(subroutine=sub)
        .select_related("instance", "instance__instance_type", "member")
        .order_by("instance__instance_name", "member__member_name", "ln")
    )
    for r in rows:
        if is_inactive(ranges, sub.pk, r.ln):
            continue
        inst = r.instance.instance_name if r.instance else r.path.split("%")[0]
        tname = r.instance.instance_type.user_type_name if r.instance else "?"
        g = groups.setdefault(inst, {"instance": inst, "type": tname, "members": {}})
        mname = r.member.member_name if r.member else "%".join(r.path.split("%")[1:])
        m = g["members"].setdefault(mname, {"name": mname, "statuses": [], "lines": [], "key": f"{inst}%{mname}", "linked": r.member is not None})
        m["statuses"].append(r.status)
        m["lines"].append(r.ln)
    out = []
    for g in groups.values():
        members = []
        for m in g["members"].values():
            m["status"] = merge_status(m["statuses"])
            members.append(m)
        g["members"] = members
        out.append(g)
    globals_: dict[str, dict] = {}
    for r in SubroutineIntrinsicGlobals.objects.filter(sub_id=sub).select_related("gv_id", "gv_id__gv_module").order_by("gv_ln"):
        if is_inactive(ranges, sub.pk, r.gv_ln):
            continue
        name = f"{r.gv_id.gv_module.module_name}::{r.gv_id.var_name}" if r.gv_id else r.path
        g = globals_.setdefault(name, {"key": name, "name": r.gv_id.var_name if r.gv_id else r.path, "module": r.gv_id.gv_module.module_name if r.gv_id else "",
                                      "type": r.gv_id.var_type if r.gv_id else "", "namelist": bool(r.gv_id and r.gv_id.namelist_group),
                                      "parameter": bool(r.gv_id and r.gv_id.parameter), "statuses": [], "lines": []})
        g["statuses"].append(r.status)
        g["lines"].append(r.gv_ln)
    for g in globals_.values():
        g["status"] = merge_status(g["statuses"])
    return out, sorted(globals_.values(), key=lambda g: (g["namelist"] is False, g["name"]))


def routine_callees(sub: Subroutines, ranges: dict) -> list[dict]:
    out = []
    for c in SubroutineCalltree.objects.filter(parent_subroutine=sub).select_related("child_subroutine", "child_subroutine__module").order_by("lineno", "position"):
        out.append({
            "ln": c.lineno,
            "name": c.child_subroutine.subroutine_name if c.child_subroutine else c.callee_name,
            "sub_id": c.child_subroutine.sub_id if c.child_subroutine else "",
            "kind": c.kind,
            "resolved": c.child_subroutine is not None,
            "analyzed": bool(c.child_subroutine and c.child_subroutine.analyzed),
            "gated": is_inactive(ranges, sub.pk, c.lineno),
        })
    return out


def find_subroutine(snapshot: Snapshot, name: str) -> Optional[Subroutines]:
    """By `module::name`, or by bare name (the analyzed one if several)."""
    name = (name or "").lower()
    qs = Subroutines.objects.filter(snapshot=snapshot).select_related("module")
    if "::" in name:
        return qs.filter(sub_id=name).first()
    return qs.filter(subroutine_name=name).order_by("-analyzed", "sub_id").first()
