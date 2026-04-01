
import sys
from app.models import PresetConfig
from django.core.management.base import BaseCommand

from ...utils.ifs import compute_if_evals_for_hash
from ...utils.populate_config_tables import propagate_bindings


class Command(BaseCommand):
    help = "Populate SubroutineElmtypesByConfig for each of the presets"

    def handle(self, *args, **opts):
        presets = PresetConfig.objects.all()
        for preset in presets:
            data = preset.data
            cfg = preset.preset_hash
            num = compute_if_evals_for_hash(config_hash=cfg, active_data=data)
            if num == 0:
                sys.exit(f"Error {preset.name} has empty if table")
            num = propagate_bindings(cfg_hash=cfg)
            print(f"{preset.name} has {num} entries on SubroutineElmtypesByConfig")

