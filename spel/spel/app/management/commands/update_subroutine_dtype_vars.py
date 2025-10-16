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
                instance_name = row["inst_name"].strip()
                sub_module_name = row["sub_module"].strip()
                type_module_name = row["type_module"].strip()
                user_type_name = row["inst_type"].strip()
                member_type = row["member_type"].strip()
                member_name = row["member_name"].strip()
                status = row["status"].strip()
                ln = row["ln"].strip()

                try:
                    sub_mod_obj = Modules.objects.get(module_name=sub_module_name)
                except Modules.DoesNotExist:
                    self.stdout.write(
                        self.style.ERROR(f"Module {sub_module_name} not found.")
                    )
                    sys.exit(1)

                try:
                    type_mod_obj = Modules.objects.get(module_name=type_module_name)
                except Modules.DoesNotExist:
                    self.stdout.write(
                        self.style.ERROR(f"Module {type_module_name} not found.")
                    )
                    sys.exit(1)

                # Lookup the Subroutine record.
                try:
                    subroutine_obj = Subroutines.objects.get(
                        module=sub_mod_obj, subroutine_name=subroutine_name
                    )
                except Subroutines.DoesNotExist:
                    self.stdout.write(
                        self.style.ERROR(f"Subroutine {subroutine_name} not found.")
                    )
                    sys.exit(1)

                try:
                    inst_type_obj = UserTypes.objects.get(
                        module=type_mod_obj, user_type_name=user_type_name
                    )
                except UserTypes.DoesNotExist:
                    self.stdout.write(
                        self.style.ERROR(f"Type {user_type_name} not found.")
                    )
                    sys.exit(1)

                # Lookup the UserTypeInstances record.
                try:
                    instance_obj = UserTypeInstances.objects.get(
                        inst_module=type_mod_obj,
                        instance_type=inst_type_obj,
                        instance_name=instance_name,
                    )
                except UserTypeInstances.DoesNotExist:
                    self.stdout.write(
                        self.style.ERROR(f"Instance {instance_name} not found.")
                    )
                    sys.exit(1)

                # Lookup the TypeDefinitions record.
                try:
                    type_def_obj = TypeDefinitions.objects.get(
                        type_module=type_mod_obj,
                        user_type=inst_type_obj,
                        member_type=member_type,
                        member_name=member_name,
                    )
                except TypeDefinitions.DoesNotExist:
                    self.stdout.write(
                        self.style.ERROR(
                            f"TypeDefinition not found for module {module_name}, user_type {user_type_name}, member_type {member_type}, member_name {member_name}."
                        )
                    )
                    sys.exit(1)

                # Update or create the SubroutineActiveGlobalVars record.
                obj, created = SubroutineActiveGlobalVars.objects.update_or_create(
                    subroutine=subroutine_obj,
                    instance=instance_obj,
                    member=type_def_obj,
                    defaults={"status": status, "ln": ln},
                )

        self.stdout.write(self.style.SUCCESS("Global Derived Types Update complete."))
