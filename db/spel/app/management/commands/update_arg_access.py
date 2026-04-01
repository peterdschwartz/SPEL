import csv

from django.core.management.base import BaseCommand, sys

from app.models import ArgAccess, Modules, SubroutineArgs, Subroutines


class Command(BaseCommand):
    help = "Updates FlatIf, FlatIfNamelistVar and NamelistVariable model tables from CSV file."

    def add_arguments(self, parser):
        parser.add_argument("csv_file", type=str, help="Path to CSV file.")

    def handle(self, *args, **options):
        csv_file = options["csv_file"]
        with open(csv_file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                module = row["module"]
                subname = row["subroutine"]
                arg = row["dummy_arg"]
                member_path = row["member_path"]
                ln = row["ln"]
                status = row["status"]

                mod_obj = Modules.objects.get(module_name=module)
                sub_obj = Subroutines.objects.get(module=mod_obj,subroutine_name=subname)
                arg_obj = SubroutineArgs.objects.get(subroutine=sub_obj,arg_name=arg)
                arg_access_obj = ArgAccess.objects.update_or_create(
                    subroutine=sub_obj,
                    arg=arg_obj,
                    member_path=member_path,
                    ln=ln,
                    status=status,
                )
