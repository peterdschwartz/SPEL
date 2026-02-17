from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Optional

from django.conf import settings

from .models import SubroutineActiveGlobalVars, SubroutineCalltree, Subroutines

DB_NAME = settings.DATABASES["default"]["NAME"]

modules = {}


class Node:
    __slots__ = ("name", "children", "is_hit","id","uid")

    def __init__(
        self,
        name: str,
        id : str,
        is_hit: bool = False,
        children=None
    ):
        self.name: str = name
        self.children: list[Node] = []
        if children:
            self.children = children
        self.is_hit: bool = is_hit
        self.id: str = id
        self.uid = ""

    def __repr__(self):
        if self.children:
            str_ = "/".join([str(c) for c in self.children])
        else:
            str_ = ""
        return f"{self.id} -> {str_}"

    def __str__(self):
        return f"{self.id}"

def print_node(node: Node, level:int=0):
    """Recursively prints the tree in a hierarchical format."""
    if level == 0:
        print("CallTree for ", node.id)
    indent = "|--" * level
    print(f"{indent}>{node.id} {node.uid}")

    for child in node.children:
        print_node(child, level + 1)

def annotate_tree(tree: list[Node], prefix="ctree"):
    """
    Annotate a pruned call tree with a unique 'uid' per *occurrence*.
    The uid is a path, e.g., 'ctree-0-3-1'. Runs in O(N).
    Returns the same structure, modified in-place (and also returns it).
    """
    path = []
    def walk(nodes, depth=0):
        for i, node in enumerate(nodes):
            kids = node.children
            node_uid = "-".join([prefix] + list(map(str, path + [i])))
            node.uid = node_uid

            path.append(i)
            walk(kids,depth+1)
            path.pop()
    walk(tree,0)
    return

def prune_tree(
    roots: list[Node],
    hit_set: set[str],
    traversable_set: Optional[set[str]] = None,
) -> list[Node]:

    seen_map: dict[str, Optional[Node] | object] = {}
    IN_PROGRESS = object()

    def may_descend(name: str) -> bool:
        return traversable_set is None or name in traversable_set

    def search_children(node: Node) -> Optional[Node]:
        val = seen_map.get(node.id, None)
        if val is not None:
            if val is IN_PROGRESS:
                new_node = (
                    Node(node.name, id=node.id, is_hit=(node.name in hit_set))
                    if (node.name in hit_set)
                    else None
                )
                seen_map[node.id] = new_node
                return new_node
            else:
                return deepcopy(val)

        # HARD-STOP: if we cannot descend here, we decide solely on self-hit
        if not may_descend(node.name):
            kept = node.name in hit_set
            seen_map[node.id] = (
                Node(node.name,id=node.id,is_hit=True, children=[]) if kept else None
            )
            return deepcopy(seen_map[node.id])

        kept_children: list[Node] = []
        seen_map[node.id] = IN_PROGRESS

        for child in node.children:
            kc = search_children(child)
            if kc is not None:
                kept_children.append(kc)

        is_hit = node.name in hit_set
        if is_hit or kept_children:
            new_node = Node(name=node.name,id=node.id, is_hit=is_hit)
            new_node.children = kept_children.copy()
            seen_map[node.id] = new_node
            return new_node

        seen_map[node.id] = None
        return None

    ptree: list[Node] = []
    for root in roots:
        new_root = search_children(root)
        if new_root is not None:
            ptree.append(new_root)

    return ptree


def get_subroutine_details(instance, member, mode):
    from django.db.models import F

    """
    Function that queries database
    """
    # Filter subroutines not in the call tree
    if mode == "head":
        excluded_subroutines = SubroutineCalltree.objects.values_list(
            "child_subroutine", flat=True
        )
    else:
        excluded_subroutines = []

    if instance != "":
        if member != "":
            # Query the database with partial matches
            results = (
                SubroutineActiveGlobalVars.objects.filter(
                    instance__instance_name=instance,
                    member__member_name__contains=member,  # Partial match for member_name
                )
                .exclude(subroutine__in=list(excluded_subroutines))
                .values(
                    sub=F("subroutine__subroutine_name"),
                    inst=F("instance__instance_name"),
                    m=F("member__member_name"),
                    rw=F("status"),
                )
            )
        else:
            results = (
                SubroutineActiveGlobalVars.objects.filter(
                    instance__instance_name__contains=instance,
                )
                .exclude(subroutine__in=list(excluded_subroutines))
                .values(
                    sub=F("subroutine__subroutine_name"),
                    inst=F("instance__instance_name"),
                    m=F("member__member_name"),
                    rw=F("status"),
                )
            )
    elif instance == "" and member == "":
        results = SubroutineActiveGlobalVars.objects.all().values(
            sub=F("subroutine__subroutine_name"),
            inst=F("instance__instance_name"),
            m=F("member__member_name"),
            rw=F("status"),
        )
    return results


def build_calltree(root_subroutine_name, active_subs=None):
    """
    Build a call tree starting from the subroutine with name root_subroutine_name.
    If active_subs is provided (a set or list of subroutine names),
    only include children whose names are in active_subs.
    """
    from app.models import SubroutineCalltree, Subroutines

    try:
        root_sub = Subroutines.objects.get(subroutine_name=root_subroutine_name)
    except Subroutines.DoesNotExist:
        return None

    # Create a mapping: parent_subroutine_id -> list of child Subroutines
    edges = SubroutineCalltree.objects.select_related(
        "parent_subroutine", "child_subroutine"
    ).all()
    calltree_map = {}
    for edge in edges:
        parent_id = edge.parent_subroutine.subroutine_name
        calltree_map.setdefault(parent_id, []).append(edge.child_subroutine)

    # Build the tree using a queue (BFS) to avoid recursion.
    root_node = Node(root_sub.subroutine_name)
    queue = [(root_sub, root_node)]
    visited = set()

    while queue:
        current_sub, current_node = queue.pop(0)
        # Avoid cycles:
        if current_sub.subroutine_name in visited:
            continue
        visited.add(current_sub.subroutine_name)
        children = calltree_map.get(current_sub.subroutine_name, [])
        for child in children:
            child_node = Node(child.subroutine_name)
            current_node.children.append(child_node)
            queue.append((child, child_node))

    def prune_tree(node):
        # If active_subs is None, we don't filter.
        if active_subs is None:
            return True
        # Check if the current node is active.
        contains_active = node.name in active_subs
        pruned_children = []
        for child in node.children:
            # Recursively prune children.
            if prune_tree(child):
                pruned_children.append(child)
                contains_active = True
        node.children = pruned_children
        return contains_active

    prune_tree(root_node)
    return root_node


def create_calltree_from_sub(sub_name: str) -> list[Node]:
    try:
        root_sub = Subroutines.objects.get(subroutine_name=sub_name)
    except Subroutines.DoesNotExist:
        return None

    # Pull just what we need, already sorted by parent then lineno
    edges = SubroutineCalltree.objects.values(
        "parent_subroutine__subroutine_name",
        "child_subroutine__subroutine_name",
        "lineno",
    ).order_by(
        "parent_subroutine__subroutine_name",
        "lineno",
        "child_subroutine__subroutine_name",
    )

    calltree_map: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for e in edges:
        p = e["parent_subroutine__subroutine_name"]
        c = e["child_subroutine__subroutine_name"]
        ln = e["lineno"]
        calltree_map[p].append((c, ln))

    root_id = f"{sub_name}@L0"
    root = Node(sub_name,id=root_id)
    queue = [(sub_name, root)]

    # Build the tree using a queue (BFS) to avoid recursion.
    # Expand each subroutine at most once to avoid infinite loops on recursion.
    expanded: set[str] = set()

    while queue:
        cur_id, cur_node = queue.pop(0)
        if cur_id in expanded:
            continue
        expanded.add(cur_id)
        cur_name = cur_id.split("@")[0]

        # children are already ordered by lineno due to the queryset order_by
        for child_name, ln in calltree_map.get(cur_name, []):
            id = f"{child_name}@L{ln}"
            child_node = Node(child_name,id=id)
            cur_node.children.append(child_node)

            # enqueue for expansion unless we've already expanded that subroutine elsewhere
            if id not in expanded:
                queue.append((id, child_node))

    return [root]


def get_calltree_for_var(instance, member, active_only=True):
    tree = []

    if active_only:
        active_vars_query = get_subroutine_details(instance, member, mode="")
    parent_subs = get_subroutine_details(instance, member, mode="head")

    parent_sub_names = []
    [
        parent_sub_names.append(row["sub"])
        for row in parent_subs
        if row["sub"] not in parent_sub_names
    ]

    active_subs = []
    if active_vars_query:
        [
            active_subs.append(s["sub"])
            for s in active_vars_query
            if s["sub"] not in active_subs
        ]

    active_vars_table = []
    for item in active_vars_query:
        row = [
            item["sub"],
            f'{item["inst"]}%{item["m"]}',
            item["rw"],
        ]
        active_vars_table.append(row)

    for sub_name in parent_sub_names:
        root = build_calltree(sub_name, active_subs)
        tree.append(root)

    return tree, active_vars_table
