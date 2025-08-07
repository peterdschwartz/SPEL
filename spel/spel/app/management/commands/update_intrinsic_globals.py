import csv

from app.models import (
    IntrinsicGlobals,
    Modules,
    SubroutineIntrinsicGlobals,
    Subroutines,
)
from django.core.management.base import BaseCommand, sys


class Command(BaseCommand):
    help = "Updates intrinsic global variable table."

    def add_arguments(self, parser):
        parser.add_argument("csv_file", type=str, help="Path to CSV file.")

    def handle(self, *args, **options):
        csv_file = options["csv_file"]
        with open(csv_file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                module = row["module"]
                var_name = row["var_name"]
                var_type = row["var_type"]
                dim = row["dim"]
                bounds = row["bounds"]
                value = row["value"]
                sub_mod = row["sub_module"]
                sub_name = row["sub_name"]

                try:
                    mod_obj = Modules.objects.get(module_name=module)
                except Modules.DoesNotExist:
                    self.stdout.write(self.style.ERROR(f"Module {module} not found."))
                    sys.exit(1)

                try:
                    sub_obj = Subroutines.objects.get(subroutine_name=sub_name)
                except Subroutines.DoesNotExist:
                    self.stdout.write(
                        self.style.ERROR(f"Subroutine {sub_name} not found!")
                    )
                    sys.exit(1)

                var_obj, _ = IntrinsicGlobals.objects.update_or_create(
                    gv_module=mod_obj,
                    var_name=var_name,
                    defaults={
                        "dim": dim,
                        "var_type": var_type,
                        "bounds": bounds,
                        "value": value,
                    },
                )

                sub_var_obj, _ = SubroutineIntrinsicGlobals.objects.update_or_create(
                    sub_id=sub_obj, gv_id=var_obj
                )
