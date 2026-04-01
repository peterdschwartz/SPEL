import csv
import os

from app.models import Modules
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Update Modules and ModuleDependency with new data from a CSV file."

    def add_arguments(self, parser):
        parser.add_argument(
            "csv_file",
            type=str,
            help="The path to the CSV file containing new data.",
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
                # Update or create the module record
                module, _ = Modules.objects.update_or_create(module_name=module_name)

        self.stdout.write(self.style.SUCCESS("Module update complete."))
