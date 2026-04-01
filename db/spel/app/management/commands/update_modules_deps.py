import csv
import os

from app.models import ModuleDependency, Modules
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
                module_name = row.get("module_name").strip()
                dep_module_name = row.get("dep_module_name").strip()
                object_used = row.get("object_used").strip()

                # Update or create the module record
                module, _ = Modules.objects.update_or_create(
                    module_name=module_name,
                    defaults={"module_name": module_name},
                )
                dep_module, _ = Modules.objects.update_or_create(
                    module_name=dep_module_name,
                    defaults={"module_name": dep_module_name},
                )

                # Update or create the dependency record
                dependency, created = ModuleDependency.objects.update_or_create(
                    module=module,
                    dep_module=dep_module,
                    object_used=object_used,
                    defaults={"object_used": object_used},
                )
        self.stdout.write(self.style.SUCCESS("Module Dependency update complete."))
