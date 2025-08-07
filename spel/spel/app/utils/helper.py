import hashlib
import json


def compute_config_hash(data: dict) -> str:
    """
    Canonical SHA256 of config data.
    - Sort keys
    - Compact separators
    - Ensure JSON-serializable (lists/bools/numbers/strings)
    """
    canonical = json.dumps(data or {}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
