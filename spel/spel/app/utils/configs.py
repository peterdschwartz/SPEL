# utils/configs.py

from ..models import ConfigProfile, PresetConfig


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
