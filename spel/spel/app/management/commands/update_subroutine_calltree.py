import csv
import os

from app.models import Modules, SubroutineCalltree, Subroutines
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Update Modules and ModuleDependency with new data from a CSV file."

    def add_arguments(self, parser):
        parser.add_argument(
            "csv_file", type=str, help="The path to the CSV file containing new data."
        )

    def handle(self, *args, **options):
        csv_file = options["csv_file"]
        if not os.path.exists(csv_file):
            self.stdout.write(self.style.ERROR(f"CSV file not found: {csv_file}"))
            return

        with open(csv_file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                mod_parent = row.get("mod_parent")
                parent_sub = row.get("parent_subroutine")
                mod_child = row.get("mod_child")
                child_sub = row.get("child_subroutine")
                ln = row.get("ln")
                args = row.get("args")

                try:
                    parent_mod_obj = Modules.objects.get(module_name=mod_parent)
                except Modules.DoesNotExist:
                    self.stdout.write(
                        self.style.ERROR(f"Module {mod_parent} not found.")
                    )
                    continue

                try:
                    child_mod_obj = Modules.objects.get(module_name=mod_child)
                except Modules.DoesNotExist:
                    self.stdout.write(
                        self.style.ERROR(f"Module {mod_child} not found.")
                    )
                    continue

                try:
                    parent_sub_obj = Subroutines.objects.get(
                        module=parent_mod_obj, subroutine_name=parent_sub
                    )
                except Subroutines.DoesNotExist:
                    self.stdout.write(
                        self.style.ERROR(f"Subroutine {parent_sub} not found.")
                    )
                    continue

                try:
                    child_sub_obj = Subroutines.objects.get(
                        module=child_mod_obj, subroutine_name=child_sub
                    )
                except Subroutines.DoesNotExist:
                    self.stdout.write(
                        self.style.ERROR(f"Subroutine {child_sub} not found.")
                    )
                    continue

                calltree_obj, created = SubroutineCalltree.objects.update_or_create(
                    parent_subroutine=parent_sub_obj,
                    child_subroutine=child_sub_obj,
                    lineno=ln,
                    defaults={"args": args},
                )
        self.stdout.write(self.style.SUCCESS("Call Tree Data update complete."))
