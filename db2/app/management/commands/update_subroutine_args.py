import csv
import os

from app.models import SubroutineArgs, Subroutines
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Update SubroutineArgs with new data from a CSV file."

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
                module_name = row.get("module")
                subroutine = row.get("subroutine")
                arg_type = row.get("arg_type")
                arg_name = row.get("arg_name")
                dim = int(row.get("dim"))

                try:
                    subroutine_obj = Subroutines.objects.get(
                        module__module_name=module_name, subroutine_name=subroutine
                    )
                except Subroutines.DoesNotExist:
                    self.stdout.write(
                        self.style.ERROR(f"Subroutine {subroutine} not found.")
                    )
                    continue

                arg_obj, created = SubroutineArgs.objects.update_or_create(
                    subroutine=subroutine_obj,
                    arg_name=arg_name,
                    defaults={
                    "arg_type":arg_type,
                    "dim":dim,
                    },
                )
        self.stdout.write(self.style.SUCCESS("Subroutine Args Data update complete."))
