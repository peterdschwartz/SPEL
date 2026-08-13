from collections import defaultdict
from typing import NamedTuple

from scripts.types import LineTuple, ReadWrite

from ..models import (
    ArgAccess,
    PropagatedEffectByLn,
    SubroutineActiveGlobalVars,
    SubroutineCalltree,
    SubroutineElmtypesByConfig,
    Subroutines,
    UserTypeInstances,
)
from .ifs import filter_access_lns_by_hash
from .view_helper import combine_many_statuses, trace_var_to_instances, reachable_subroutine_ids


class AccessKey(NamedTuple):
    subroutine_id: int
    inst_id: int
    member_id: int
    call_ln: int
    var_name: str
    member_path: str
    indirect: bool


def propagate_bindings(cfg_hash) -> int:
    qs = SubroutineElmtypesByConfig.objects.filter(config_hash=cfg_hash)
    if qs.exists():
        # SubroutineElmtypesByConfig.objects.filter(config_hash=cfg_hash).delete()
        return qs.count()

    elm_drv = Subroutines.objects.get(subroutine_name="elm_drv")
    reachable_ids = reachable_subroutine_ids(root_subroutine=elm_drv, cfg_hash=cfg_hash)

    prop_qs = (
        PropagatedEffectByLn.objects.filter(
            call_site__parent_subroutine__subroutine_id__in=reachable_ids,
        )
        .select_related("call_site", "binding")
        .order_by("call_site__child_subroutine__subroutine_name", "lineno")
    )

    # I believe I need two filter passes.
    #   First, to remove callsites in the parent that are in dead if branches
    #   Second, to remove accesses in the child that are in dead if branches
    # I think the first is needed because if a subroutine may still be in reachable_ids but used multiple times
    # and some of those times could be in dead if branches
    prop_qs = filter_access_lns_by_hash(
        qs=prop_qs,
        sub_field="call_site__parent_subroutine",
        lineno_field="call_site__lineno",
        config_hash=cfg_hash,
    )
    prop_qs = filter_access_lns_by_hash(
        qs=prop_qs,
        sub_field="call_site__child_subroutine",
        lineno_field="lineno",
        config_hash=cfg_hash,
    )

    call_lns_list = SubroutineCalltree.objects.filter(
        parent_subroutine__in=reachable_ids, child_subroutine__in=reachable_ids
    ).values_list(
        "parent_subroutine__subroutine_name",
        "child_subroutine__subroutine_name",
        "lineno",
    )

    call_lns_by_parent: dict[str, set[int]] = defaultdict(set)
    for t in call_lns_list:
        call_lns_by_parent[f"{t[0]}"].add(t[2])
    # overall_status_for_parent:  list[SubroutineElmtypesByConfig] = []
    agg_for_parent: dict[AccessKey, list[ReadWrite]] = defaultdict(list)
    batch: list[SubroutineElmtypesByConfig] = []
    child_batch: dict[AccessKey, str] = {}
    for row in prop_qs:
        parent_sub = row.call_site.parent_subroutine
        child_sub = row.call_site.child_subroutine
        call_ln = row.call_site.lineno
        var_name = row.var_name
        member_path = row.member_path
        access_ln = row.lineno
        member = row.first_member
        member_id = member.define_id if member else None
        lns = call_lns_by_parent[str(child_sub.subroutine_name)]
        indirect = bool(access_ln in lns)

        if row.scope == "ELMTYPE":
            instance = row.from_elmtype
            child_key = AccessKey(
                subroutine_id=child_sub.subroutine_id,
                inst_id=instance.instance_id,
                member_id=member_id,
                call_ln=access_ln,
                var_name=var_name,
                member_path=member_path,
                indirect=indirect,
            )

            child_batch[child_key] = row.status

            parent_key = AccessKey(
                subroutine_id=parent_sub.subroutine_id,
                inst_id=instance.instance_id,
                member_id=member_id,
                call_ln=call_ln,
                var_name=var_name,
                member_path=member_path,
                indirect=True,
            )
            agg_for_parent[parent_key].append(
                ReadWrite(
                    status=row.status,
                    ln=access_ln,
                    line=LineTuple(
                        ln=-1,
                        line="",
                    ),
                )
            )
        elif row.scope == "ARG":
            arg = row.from_arg
            instances = trace_var_to_instances(
                sub_id=parent_sub.subroutine_id,
                var_name=str(var_name),
                scope=str(row.scope),
                cfg=cfg_hash,
            )
            for inst in instances:
                inst_obj = UserTypeInstances.objects.filter(instance_name=inst).first()
                child_key = AccessKey(
                    subroutine_id=child_sub.subroutine_id,
                    inst_id=inst_obj.instance_id,
                    member_id=member_id,
                    call_ln=access_ln,
                    var_name=var_name,
                    member_path=member_path,
                    indirect=indirect,
                )

                child_batch[child_key] = row.status

                parent_key = AccessKey(
                    subroutine_id=parent_sub.subroutine_id,
                    inst_id=inst_obj.instance_id,
                    member_id=member_id,
                    call_ln=call_ln,
                    var_name=var_name,
                    member_path=member_path,
                    indirect=True,
                )
                agg_for_parent[parent_key].append(
                    ReadWrite(
                        status=row.status,
                        ln=access_ln,
                        line=LineTuple(
                            ln=-1,
                            line="",
                        ),
                    )
                )

    # Aggregate statuses
    for parent_key, statuses in agg_for_parent.items():
        overall = combine_many_statuses(
            [x.status for x in sorted(statuses, key=lambda rw: rw.ln)]
        )
        batch.append(
            SubroutineElmtypesByConfig(
                config_hash=cfg_hash,
                subroutine_id=parent_key.subroutine_id,
                instance_id=parent_key.inst_id,
                member_id=parent_key.member_id,
                status=overall,
                ln=parent_key.call_ln,
                var_name=parent_key.var_name,
                member_path=parent_key.member_path,
                indirect=True,
            )
        )

    batch.extend(
        [
            SubroutineElmtypesByConfig(
                config_hash=cfg_hash,
                subroutine_id=key.subroutine_id,
                instance_id=key.inst_id,
                member_id=key.member_id,
                status=status,
                ln=key.call_ln,
                var_name=key.var_name,
                member_path=key.member_path,
                indirect=key.indirect,
            )
            for key, status in child_batch.items()
        ]
    )

    elm_qs = SubroutineActiveGlobalVars.objects.filter(
        subroutine__subroutine_id__in=reachable_ids,
    ).order_by("subroutine", "ln")

    elm_qs = filter_access_lns_by_hash(
        qs=elm_qs,
        sub_field="subroutine",
        lineno_field="ln",
        config_hash=cfg_hash,
    )

    for row in elm_qs:
        batch.append(
            SubroutineElmtypesByConfig(
                config_hash=cfg_hash,
                subroutine=row.subroutine,
                instance=row.instance,
                member=row.member,
                status=row.status,
                ln=row.ln,
                var_name=row.instance.instance_name,
                member_path=row.member.member_name,
            )
        )

    SubroutineElmtypesByConfig.objects.bulk_create(batch)

    return len(batch)
