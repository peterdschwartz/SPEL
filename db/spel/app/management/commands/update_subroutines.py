import csv
import os

from app.models import Modules, Subroutines
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Update Subroutines table with new data from a CSV file."

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
                module = row.get("module")
                sub_name = row.get("subroutine")

                try:
                    mod_obj = Modules.objects.get(module_name=module)
                except Modules.DoesNotExist:
                    self.stdout.write(self.style.ERROR(f"Module {module} not found."))
                    continue

                sub_obj, created = Subroutines.objects.update_or_create(
                    module=mod_obj,
                    subroutine_name=sub_name,
                )
