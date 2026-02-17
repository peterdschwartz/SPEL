import csv

from app.models import (
    Modules,
    SubroutineActiveGlobalVars,
    Subroutines,
    TypeDefinitions,
    UserTypeInstances,
    UserTypes,
)
from django.core.management.base import BaseCommand, sys


class Command(BaseCommand):
    help = "Update SubroutineActiveGlobalVars from CSV file."

    def add_arguments(self, parser):
        parser.add_argument("csv_file", type=str, help="Path to CSV file.")

    def handle(self, *args, **options):
        csv_file = options["csv_file"]
        with open(csv_file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                subroutine_name = row["subroutine"].strip()
                inst_module = row["inst_mod"].strip()
                instance_name = row["inst_name"].strip()
                sub_module_name = row["sub_module"].strip()
                type_module_name = row["type_module"].strip()
                user_type_name = row["inst_type"].strip()
                member_type = row["member_type"].strip()
                member_name = row["member_name"].strip()
                status = row["status"].strip()
                ln = row["ln"].strip()

                sub_mod_obj = Modules.objects.get(module_name=sub_module_name)
                type_mod_obj = Modules.objects.get(module_name=type_module_name)
                inst_mod_obj = Modules.objects.get(module_name=inst_module)

                # Lookup the Subroutine record.
                subroutine_obj = Subroutines.objects.get(
                        module=sub_mod_obj, subroutine_name=subroutine_name
                    )

                inst_type_obj = UserTypes.objects.get(
                        module=type_mod_obj, user_type_name=user_type_name
                    )

                # Lookup the UserTypeInstances record.
                instance_obj = UserTypeInstances.objects.get(
                        inst_module=inst_mod_obj,
                        instance_type=inst_type_obj,
                        instance_name=instance_name,
                    )

                # Lookup the TypeDefinitions record.
                type_def_obj = TypeDefinitions.objects.get(
                        type_module=type_mod_obj,
                        user_type=inst_type_obj,
                        member_type=member_type,
                        member_name=member_name,
                    )

                # Update or create the SubroutineActiveGlobalVars record.
                obj, created = SubroutineActiveGlobalVars.objects.update_or_create(
                    subroutine=subroutine_obj,
                    instance=instance_obj,
                    member=type_def_obj,
                    status=status,
                    ln=ln,
                )

        self.stdout.write(self.style.SUCCESS("Global Derived Types Update complete."))
