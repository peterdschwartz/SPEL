import csv

from app.models import (
    CallsiteBinding,
    Modules,
    PropagatedEffectByLn,
    SubroutineArgs,
    SubroutineCalltree,
    Subroutines,
    TypeDefinitions,
    UserTypeInstances,
    UserTypes,
)
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Updates CallsiteBindings Table"

    def add_arguments(self, parser):
        parser.add_argument("csv_file", type=str, help="Path to CSV file.")

    def handle(self, *args, **options):
        csv_file = options["csv_file"]
        with open(csv_file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                parent_module = row["parent_module"]
                parent_sub = row["parent_sub"]
                child_module = row["child_module"]
                child_sub = row["child_sub"]
                call_ln = row["call_ln"]
                nested_level = row["nested_level"]
                bound_member = row["bound_member"]
                scope = row["scope"]
                var_name = row["var_name"]
                member_path = row["member_path"]
                status = row["status"]
                rw_ln = row["rw_ln"]
                dummy_arg = row["dummy_arg"]
                type_name = row["type_name"]
                type_mod = row["type_mod"]
                inst_mod = row["inst_mod"]

                p_mod_obj = Modules.objects.get(module_name=parent_module)
                p_sub_obj = Subroutines.objects.get(
                    module=p_mod_obj,
                    subroutine_name=parent_sub,
                )
                c_mod_obj = Modules.objects.get(module_name=child_module)
                c_sub_obj = Subroutines.objects.get(
                    module=c_mod_obj,
                    subroutine_name=child_sub,
                )

                calltree_obj = SubroutineCalltree.objects.get(
                    parent_subroutine=p_sub_obj,
                    child_subroutine=c_sub_obj,
                    lineno=call_ln,
                )

                dummy_arg_obj = SubroutineArgs.objects.get(
                    subroutine=c_sub_obj,
                    arg_name=dummy_arg,
                )


                if scope == "ELMTYPE":
                    arg_obj = None
                    local_obj = None
                    type_mod_obj = Modules.objects.get(module_name=type_mod)
                    type_obj = UserTypes.objects.get(
                        module=type_mod_obj, user_type_name=type_name
                    )
                    inst_mod_obj = Modules.objects.get(module_name=inst_mod)

                    instance_obj = UserTypeInstances.objects.get(
                        inst_module=inst_mod_obj,
                        instance_type=type_obj,
                        instance_name=var_name,
                    )

                    member_obj = TypeDefinitions.objects.filter(
                        type_module=type_mod_obj,
                        user_type=type_obj,
                        member_name=member_path.split("%", 1)[0],
                    ).first()
                elif scope == "ARG":
                    instance_obj = None
                    local_obj = None
                    member_obj = None
                    arg_obj = SubroutineArgs.objects.get(subroutine=p_sub_obj,arg_name=var_name)
                elif scope == "LOCAL":
                    instance_obj = None
                    member_obj = None
                    arg_obj = None
                    local_obj = None

                call_binding_obj = CallsiteBinding.objects.get(
                    parent_subroutine=p_sub_obj,
                    call=calltree_obj,
                    dummy_arg=dummy_arg_obj,
                    var_name=var_name,
                    member_path_str=bound_member,
                    nested_level=nested_level,
                )

                prop_obj = PropagatedEffectByLn.objects.update_or_create(
                    call_site=calltree_obj,
                    var_name=var_name,
                    member_path=member_path,
                    lineno=rw_ln,
                    status=status,
                    scope=scope,
                    binding=call_binding_obj,
                    from_elmtype=instance_obj,
                    from_arg=arg_obj,
                    from_local=local_obj,
                    first_member=member_obj,
                )
