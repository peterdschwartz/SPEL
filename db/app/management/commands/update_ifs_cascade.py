import csv

from app.models import (
    CascadeDependence,
    CascadePair,
    FlatIf,
    FlatIFCascadeVar,
    IntrinsicGlobals,
    Modules,
    NamelistVariable,
    Subroutines,
)
from django.core.management.base import BaseCommand, sys
from django.db.models.fields.json import json


def sync_cascades():
    from scripts.nml.namelist_cascade import NML_CASCADES

    for cascade_var, dep in NML_CASCADES.items():
        nml_mod, nml_var = dep.trigger.split("::")
        nml_mod_obj = Modules.objects.get(module_name=nml_mod)
        nml_obj = NamelistVariable.objects.get(
            active_var_id__gv_module=nml_mod_obj,
            active_var_id__var_name=nml_var,
        )
        cascade_obj, _ = CascadeDependence.objects.update_or_create(
            trigger_var=nml_obj,
            cascade_var=cascade_var,
        )

        for pair in dep.pairs:
            pair_obj, _ = CascadePair.objects.update_or_create(
                dependence=cascade_obj,
                nml_val=pair.nml_val,
                cascade_val=pair.cascade_val,
            )

    return


class Command(BaseCommand):
    help = "Updates FlatIf, FlatIfNamelistVar and NamelistVariable model tables from CSV file."

    def add_arguments(self, parser):
        parser.add_argument("csv_file", type=str, help="Path to CSV file.")

    def handle(self, *args, **options):
        csv_file = options["csv_file"]
        sync_cascades()
        with open(csv_file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sub_modname = row["sub_module"]
                subroutine = row["subroutine"]
                nml_var_name = row["nml_var_name"]
                nml_var_module = row["nml_var_module"]
                if_start = row["if_start"]
                if_end = row["if_end"]
                if_cond = row["if_cond"]
                cascade_var = row["cascade_var"]

                nml_mod_obj = Modules.objects.get(module_name=nml_var_module)

                # Lookup the Subroutine record.
                subroutine_obj = Subroutines.objects.get(
                    module__module_name=sub_modname,
                    subroutine_name=subroutine,
                )

                nml_obj = NamelistVariable.objects.get(
                    active_var_id__gv_module=nml_mod_obj,
                    active_var_id__var_name=nml_var_name,
                )

                ifs_obj, _ = FlatIf.objects.update_or_create(
                    subroutine=subroutine_obj,
                    start_ln=if_start,
                    end_ln=if_end,
                    defaults={
                        "condition": json.loads(if_cond),
                        "active": True,
                    },
                )
                cascade_obj = CascadeDependence.objects.get(
                    trigger_var=nml_obj,
                    cascade_var=cascade_var,
                )
                ifs_cv_obj, _ = FlatIFCascadeVar.objects.update_or_create(
                    flatif_cv=ifs_obj,
                    cascade=cascade_obj,
                )
