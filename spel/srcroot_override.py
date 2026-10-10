"""
`--srcroot` for any spel command.

spel.scripts.config resolves E3SM_SRCROOT (and everything derived from it)
once, at import, from SPEL_E3SM_SRCROOT. So the option is applied to the
environment before anything imports the config; child processes (`spel
validate` runs `spel create`s) inherit it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence

ENV_VAR = "SPEL_E3SM_SRCROOT"
OPTION = "--srcroot"


def find_srcroot(argv: Sequence[str]) -> Optional[str]:
    """The value of the last `--srcroot X` / `--srcroot=X` before a `--`."""
    found = None
    for i, arg in enumerate(argv):
        if arg == "--":
            break
        if arg == OPTION and i + 1 < len(argv):
            found = argv[i + 1]
        elif arg.startswith(OPTION + "="):
            found = arg.split("=", 1)[1]
    return found


def apply_srcroot(argv: Sequence[str]) -> Optional[Path]:
    """Point SPEL at the E3SM checkout given with --srcroot (if any)."""
    value = find_srcroot(argv)
    if value is None:
        return None
    # `--srcroot=~/x` reaches us unexpanded (the shell only expands a leading ~)
    root = Path(value).expanduser().resolve()
    if not (root / "components/elm/src").is_dir():
        raise SystemExit(f"spel: error: {OPTION} {value}: no components/elm/src in {root}")
    os.environ[ENV_VAR] = str(root)
    return root
