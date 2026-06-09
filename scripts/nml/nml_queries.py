from db.app.models import PresetConfig, SubroutineElmtypesByConfig, Subroutines
from db.app.utils.view_helper import reachable_subroutine_ids
from scripts.analyze_subroutines import Subroutine
from scripts.config import database_app, options
from scripts.fortran_modules import FortranModule
from scripts.types import LineTuple, ReadWrite

ModDict = dict[str, FortranModule]


def get_preset_hash(preset_name: str = ""):
    if not preset_name:
        preset: PresetConfig = PresetConfig.objects.get(is_default=True)
    else:
        preset: PresetConfig = PresetConfig.objects.get(name=preset_name)
    return str(preset.preset_hash)


def query_active_variables(sub_dict: dict[str, Subroutine]) -> bool:
    """
    Query the existing database to get only the active variables for the functional unit test
    - Modifies sub_dict in place
    """
    from pprint import pprint

    root_subs = [sub for sub in sub_dict.values() if sub.unit_test_function]
    reachable_ids: set[int] = set()
    default_hash = get_preset_hash()

    for root in root_subs:
        sub_name = root.name
        mod_name = root.module
        root_obj = Subroutines.objects.get(
            subroutine_name=sub_name,
            module__module_name=mod_name,
        )
        reachable_ids.update(
            reachable_subroutine_ids(
                root_subroutine=root_obj,
                cfg_hash=default_hash,
            )
        )

    # Get variable types
    dtype_qs = (
        SubroutineElmtypesByConfig.objects.filter(
            config_hash=default_hash,
            subroutine__subroutine_id__in=reachable_ids,
        )
        .order_by(
            "subroutine__module__module_name",
            "subroutine__subroutine_name",
            "instance__instance_name",
            "member__member_name",
            "ln",
        )
        .values_list(
            "subroutine__module__module_name",
            "subroutine__subroutine_name",
            "instance__instance_name",
            "member__member_name",
            "status",
            "ln",
        )
    )

    if not dtype_qs:
        return False

    for mod, subname, inst_name, member, status, ln in dtype_qs:
        if member is None:
            continue
        sub = sub_dict[f"{mod}::{subname}"]
        rw = ReadWrite(status=status, ln=ln, line=LineTuple(line="", ln=ln))
        sub.elmtype_access_by_ln.setdefault(f"{inst_name}%{member}", []).append(rw)

    return True
