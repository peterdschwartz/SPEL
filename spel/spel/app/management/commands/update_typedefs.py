import csv
import os

from app.models import Modules, TypeDefinitions, UserTypes
from django.core.management.base import BaseCommand, sys


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
                module_name = row.get("module").strip()
                type_name = row.get("user_type_name").strip()
                member_type = row.get("member_type").strip()
                member_name = row.get("member_name").strip()
                dim = row.get("dim").strip()
                bounds = row.get("bounds").strip()
                dim = int(dim)
                # Lookup the Module record.
                try:
                    module_obj = Modules.objects.get(module_name=module_name)
                except Modules.DoesNotExist:
                    self.stdout.write(
                        self.style.ERROR(f"Module {module_name} not found.")
                    )
                    sys.exit(1)

                user_type_obj, created = UserTypes.objects.update_or_create(
                    module=module_obj,
                    user_type_name=type_name,
                )

                obj, created = TypeDefinitions.objects.update_or_create(
                    type_module=module_obj,
                    user_type=user_type_obj,
                    member_type=member_type,
                    member_name=member_name,
                    defaults={
                        "dim": dim,
                        "bounds": bounds,
                    },
                )
        self.stdout.write(self.style.SUCCESS("Type Defs Data update complete."))
