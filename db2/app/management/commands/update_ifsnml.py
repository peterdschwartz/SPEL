import csv

from app.models import (
    FlatIf,
    FlatIfNamelistVar,
    IntrinsicGlobals,
    Modules,
    NamelistVariable,
    Subroutines,
)
from django.core.management.base import BaseCommand, sys
from django.db.models.fields.json import json


class Command(BaseCommand):
    help = "Updates FlatIf, FlatIfNamelistVar and NamelistVariable model tables from CSV file."

    def add_arguments(self, parser):
        parser.add_argument("csv_file", type=str, help="Path to CSV file.")

    def handle(self, *args, **options):
        csv_file = options["csv_file"]
        with open(csv_file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sub_modname = row["sub_module"]
                subroutine = row["subroutine"]
                nml_var_name = row["nml_var_name"]
                nml_var_type = row["nml_var_type"]
                nml_var_dim = row["nml_var_dim"]
                nml_var_bounds = row["nml_var_bounds"]
                nml_var_module = row["nml_var_module"]
                if_start = row["if_start"]
                if_end = row["if_end"]
                if_cond = row["if_cond"]
                val = row["value"]

                try:
                    sub_mod_obj = Modules.objects.get(module_name=sub_modname)
                except Modules.DoesNotExist:
                    self.stdout.write(
                        self.style.ERROR(f"Module {sub_modname} not found.")
                    )
                    sys.exit(1)
                try:
                    nml_mod_obj = Modules.objects.get(module_name=nml_var_module)
                except Modules.DoesNotExist:
                    self.stdout.write(
                        self.style.ERROR(f"Module {nml_var_module} not found.")
                    )
                    sys.exit(1)

                # Lookup the Subroutine record.
                try:
                    subroutine_obj = Subroutines.objects.get(
                        module=sub_mod_obj,
                        subroutine_name=subroutine,
                    )
                except Subroutines.DoesNotExist:
                    self.stdout.write(
                        self.style.ERROR(f"Subroutine {subroutine} not found.")
                    )
                    sys.exit(1)

                var_obj = IntrinsicGlobals.objects.get(
                    gv_module=nml_mod_obj,
                    dim=nml_var_dim,
                    var_type=nml_var_type,
                    var_name=nml_var_name,
                    bounds=nml_var_bounds,
                )
                # Update or create the NamelistVariable record
                nml_obj, _ = NamelistVariable.objects.update_or_create(
                    active_var_id=var_obj,
                )

                # Update or create the FlatIfs Record
                ifs_obj, _ = FlatIf.objects.update_or_create(
                    subroutine=subroutine_obj,
                    start_ln=if_start,
                    end_ln=if_end,
                    defaults={
                        "condition": json.loads(if_cond),
                        "active": True,
                    },
                )
                # Update Nml - if lookup
                nml_if_obj, _ = FlatIfNamelistVar.objects.update_or_create(
                    flatif=ifs_obj, namelist_var=nml_obj
                )
