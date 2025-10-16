import csv
import os

from app.models import Modules, UserTypeInstances, UserTypes
from django.core.management.base import BaseCommand, sys


class Command(BaseCommand):
    help = "Update Type Instances with new data from a CSV file."

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
                inst_name = row.get("instance_name").strip()
                try:
                    module_obj = Modules.objects.get(module_name=module_name)
                except Modules.DoesNotExist:
                    self.stdout.write(
                        self.style.ERROR(f"Module {module_name} not found.")
                    )
                    sys.exit(1)

                try:
                    user_type_obj = UserTypes.objects.get(
                        module=module_obj,
                        user_type_name=type_name,
                    )
                except UserTypes.DoesNotExist:
                    self.stdout.write(self.style.ERROR(f"Type {type_name} not found."))
                    input("continue?")
                    continue

                inst_obj, created = UserTypeInstances.objects.update_or_create(
                    inst_module=module_obj,
                    instance_type=user_type_obj,
                    instance_name=inst_name,
                )

        self.stdout.write(self.style.SUCCESS(" Type Insts Data update complete."))
