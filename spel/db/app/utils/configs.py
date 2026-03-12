from django.utils import timezone
from datetime import timedelta
from django.db import models

from ..models import ConfigProfile, PresetConfig, SubroutineElmtypesByConfig

MAX_CONFIG_HASHES = 100
EVICT_OLDER_THAN_MINUTES = 60

def get_active_config(request):
    sel = request.session.get("active_config")

    # 1) If nothing selected, pick default preset
    if not sel:
        preset = (
            PresetConfig.objects.filter(is_default=True).first()
            or PresetConfig.objects.first()
        )
        if preset:
            request.session["active_config"] = {"type": "preset", "slug": preset.slug}
            return {
                "kind": "preset",
                "data": preset.data,
                "label": preset.name,
                "hash": preset.preset_hash,
            }
        return {"kind": "none", "data": {}, "label": "None", "hash": "none"}

    # 2) Preset selected (public)
    if sel.get("type") == "preset":
        preset = PresetConfig.objects.filter(slug=sel.get("slug")).first()
        if preset:
            return {
                "kind": "preset",
                "data": preset.data,
                "label": preset.name,
                "hash": preset.preset_hash,
            }

    # 3) User config selected (requires login to change/edit, but can view)
    if sel.get("type") == "user":
        cfg = ConfigProfile.objects.filter(id=sel.get("id")).first()
        if cfg:
            return {
                "kind": "user",
                "data": cfg.data or {},
                "label": cfg.name,
                "hash": cfg.user_hash,
            }

    request.session.pop("active_config", None)
    return get_active_config(request)


def touch_config_hash(config_hash: str):
    SubroutineElmtypesByConfig.objects.filter(config_hash=config_hash).update(
        last_used_at=timezone.now()
    )


def enforce_config_hash_limit():
    now = timezone.now()
    cutoff = now - timedelta(minutes=EVICT_OLDER_THAN_MINUTES)

    # Distinct hashes + min(last_used_at) for each hash
    hashes_qs = (
        SubroutineElmtypesByConfig.objects
        .values("config_hash")
        .annotate(
            min_last_used=models.Min("last_used_at"),
        )
        .order_by("min_last_used")
    )

    hashes = list(hashes_qs)

    if len(hashes) <= MAX_CONFIG_HASHES:
        return

    # Only evict hashes that are "cold"
    cold_hashes = [h for h in hashes if h["min_last_used"] < cutoff]

    # How many need to go to get back under the limit?
    excess = len(hashes) - MAX_CONFIG_HASHES
    if excess <= 0:
        return

    # Evict the oldest cold hashes up to `excess`
    to_drop = [h["config_hash"] for h in cold_hashes[:excess]]
    if not to_drop:
        return  

    SubroutineElmtypesByConfig.objects.filter(
        config_hash__in=to_drop
    ).delete()


