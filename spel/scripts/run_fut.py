"""Build and run a functional unit test, then validate it against ELM's data."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Sequence


def run_unit_test(case_dir: Path, exe_args: Sequence[str] = (), validate: bool = True) -> int:
    """
    `spel run`: configure/build (check_config.sh), run build/elmtest, then
    (validate=True) compare its fut-outputs with ELM's spel-outputs and check
    the static access analysis (validate_access.validate_case).
    Returns the validation exit code (0 = pass).
    """
    from spel.scripts.config import input_data_dir
    from spel.scripts.validate_access import validate_case

    case_dir = Path(case_dir).resolve()
    subprocess.run(["./check_config.sh"], check=True, cwd=case_dir)
    subprocess.run(["./build/elmtest", *exe_args], check=True, cwd=case_dir)
    if not validate:
        return 0
    return validate_case(case_dir, Path(input_data_dir) / case_dir.name)
