import json
import re
from pathlib import Path

from django.core.management import call_command

from app.models import  NamelistVariable, PresetConfig
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.utils.text import slugify


class Command(BaseCommand):
    help = "Import/update PresetConfig rows from .cfg files, filtering to known NamelistVariables"

    def add_arguments(self, parser):
        parser.add_argument(
            "--path",
            default="app/management/commands/presets",
            help="Directory containing .cfg preset files",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Parse and report without writing to DB",
        )
        parser.add_argument(
            "--default",
            metavar="SLUG",
            help="Mark the given slug as default (will unset others)",
        )

    @transaction.atomic
    def handle(self, *args, **opts):
        path = Path(opts["path"]).resolve()
        if not path.exists() or not path.is_dir():
            raise CommandError(f"Preset path does not exist or is not a dir: {path}")

        # Build mapping of "allowed" var names -> IntrinsicGlobals.var_id (only those in NamelistVariable)
        allowed_qs = NamelistVariable.objects.select_related(
            "active_var_id"
        ).values_list("active_var_id__var_name")

        nml_set = {x[0] for x in allowed_qs}

        self.stdout.write(f"nml_set: {nml_set}")
        total_files = 0
        upserts = 0

        for cfg_path in sorted(path.glob("*.cfg")):
            total_files += 1
            slug = slugify(cfg_path.stem)
            name = cfg_path.stem

            parsed = {}
            with cfg_path.open("r", encoding="utf-8") as f:
                for ln_no, line in enumerate(f, start=1):
                    line = line.strip()
                    temp = line.split("=")
                    if len(temp) != 2:
                        self.stdout.write(
                            self.style.ERROR(
                                f"{line} malformed namelist option in {name}"
                            )
                        )
                        continue

                    key, raw = temp
                    key_norm = key.strip()
                    if key_norm not in nml_set:
                        continue
                    parsed[key_norm] = raw.strip()

            if opts["dry_run"]:
                self.stdout.write(
                    f"[DRY] {cfg_path.name}: {len(parsed)} keys -> slug={slug}"
                )
                self.stdout.write(json.dumps(parsed, indent=2, ensure_ascii=False))
                continue

            obj, created = PresetConfig.objects.update_or_create(
                slug=slug,
                defaults={"name": name, "data": parsed},
            )
            upserts += 1
            self.stdout.write(
                self.style.SUCCESS(
                    f"{'Created' if created else 'Updated'} preset '{name}' (slug={slug}) with {len(parsed)} vars."
                )
            )

        # Optionally mark a default
        default = opts.get("default") if opts.get("default") else 'CNPRDCTCBC' 
        if not opts["dry_run"]:
            sel =  default.lower()
            # unset others, set selected
            PresetConfig.objects.exclude(slug=sel).update(is_default=False)
            cnt = PresetConfig.objects.filter(slug=sel).update(is_default=True)
            if cnt == 0:
                self.stdout.write(
                    self.style.WARNING(f"Default slug '{sel}' not found.")
                )
            else:
                self.stdout.write(
                    self.style.SUCCESS(f"Marked '{sel}' as default preset.")
                )

        if not opts["dry_run"]:
            call_command("update_config_preset_tables")
        self.stdout.write(
            self.style.NOTICE(f"Processed {total_files} files; upserts: {upserts}.")
        )
