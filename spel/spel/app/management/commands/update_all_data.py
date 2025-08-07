# myapp/management/commands/update_all_data.py
import glob
import os

from django.core.management import call_command
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Update Modules, ModuleDependency, and SubroutineActiveGlobalVars from CSV files."

    def add_arguments(self, parser):
        parser.add_argument(
            "--modules-csv",
            action="store_true",
            required=False,
            help="Path to the CSV file for Modules and ModuleDependency updates.",
        )
        parser.add_argument(
            "--typedef-csv",
            action="store_true",
            required=False,
            help="Path to CSV file for TypeDefinitions",
        )
        parser.add_argument(
            "--instances-csv",
            action="store_true",
            required=False,
            help="Path to CSV file for UserTypeInstances",
        )
        parser.add_argument(
            "--sub-args-csv",
            action="store_true",
            required=False,
            help="Path to the CSV file for SubroutineArgs updates.",
        )
        parser.add_argument(
            "--active-globals-csv",
            action="store_true",
            required=False,
            help="Path to the CSV file for SubroutineActiveGlobalVars updates.",
        )
        parser.add_argument(
            "--calltree-csv",
            action="store_true",
            required=False,
            help="Path to the CSV file for SubroutineCalltree updates.",
        )
        parser.add_argument(
            "--subroutines_csv",
            action="store_true",
            required=False,
            help="Path to the CSV file for Subroutine updates.",
        )
        parser.add_argument(
            "--nml_csv",
            action="store_true",
            required=False,
            help="Path to the CSV file for Ifs/NML updates.",
        )
        parser.add_argument(
            "--intrinsic_csv",
            action="store_true",
            required=False,
            help="Path to the CSV file for intrinsic type globals updates.",
        )
        parser.add_argument(
            "--all",
            action="store_true",
            required=False,
            help="If set, automatically search the default CSV directory for all CSV files.",
        )

    def handle(self, *args, **options):

        default_dir = (
            os.path.dirname(__file__) + "/csv/"
        )  # Replace with your hardcoded directory path
        self.stdout.write(f"Searching for CSV files in {default_dir}...")
        # Get all CSV files in the default directory
        files = glob.glob(os.path.join(default_dir, "*.csv"))
        # Prepare a mapping dictionary
        csv_files = {}
        for f in files:
            filename = os.path.basename(f).lower()
            if "module_deps" in filename:
                csv_files["modules_csv"] = f
            elif "type_defs" in filename:
                csv_files["typedef_csv"] = f
            elif "user_type_instances" in filename:
                csv_files["instances_csv"] = f
            elif "subroutine_args" in filename:
                csv_files["sub_args_csv"] = f
            elif "active_dtype_vars" in filename:
                csv_files["active_globals_csv"] = f
            elif "subroutine_calltree" in filename:
                csv_files["calltree_csv"] = f
            elif "subroutines" in filename:
                csv_files["subroutines_csv"] = f
            elif "nml_ifs" in filename:
                csv_files["nml_csv"] = f
            elif "intrinsic_globals" in filename:
                csv_files["intrinsic_csv"] = f
        # Update options with the detected file paths.
        if options["all"]:
            options.update(csv_files)
            for key, path in csv_files.items():
                self.stdout.write(f"Found {key}: {path}")
        else:
            for key in csv_files:
                if options[key]:
                    options[key] = csv_files[key]
                    print("csv:", csv_files[key])
        input("Continue?")

        modules_csv = options.get("modules_csv")
        subroutines_csv = options.get("subroutines_csv")
        typedef_csv = options.get("typedef_csv", None)
        instances_csv = options.get("instances_csv", None)
        active_globals_csv = options.get("active_globals_csv", None)
        sub_args_csv = options.get("sub_args_csv", None)
        sub_calltree = options.get("calltree_csv", None)
        nml_csv = options.get("nml_csv", None)
        intrinsic_csv = options.get("intrinsic_csv", None)

        if modules_csv:
            self.stdout.write("Updating Modules and ModuleDependency...")
            call_command("update_modules_deps", modules_csv)
        if typedef_csv:
            self.stdout.write("Updating TypeDefinitions...")
            call_command("update_typedefs", typedef_csv)
        if instances_csv:
            self.stdout.write("Updating UserTypeInstances...")
            call_command("update_type_insts", instances_csv)
        if subroutines_csv:
            self.stdout.write("Updating Subroutines...")
            call_command("update_subroutines", subroutines_csv)
        if sub_calltree:
            self.stdout.write("Updating SubroutineCalltree...")
            call_command("update_subroutine_calltree", sub_calltree)
        if active_globals_csv:
            self.stdout.write("Updating SubroutineActiveGlobalVars...")
            call_command("update_subroutine_dtype_vars", active_globals_csv)
        if sub_args_csv:
            self.stdout.write("Updating SubroutineArgs...")
            call_command("update_subroutine_args", sub_args_csv)
        if intrinsic_csv:
            self.stdout.write("Updating Intrinsic Globals...")
            call_command("update_intrinsic_globals", intrinsic_csv)
        if nml_csv:
            self.stdout.write("Updating If and NML Tables...")
            call_command("update_ifsnml", nml_csv)
