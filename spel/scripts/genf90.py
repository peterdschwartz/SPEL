"""
genf90 templates (`*.F90.in`) as analyzable sources.

E3SM generates dynVarMod, dynVarTimeInterpMod, dynVarTimeUninterpMod, ncdio_pio, restUtilMod
and the shr_assert/shr_frz/shr_infnan modules from genf90 templates at build time; a checkout
has only the templates, so those modules were missing for the analysis and their users
(dyncropFileMod, dynpftFileMod) had to be excluded (`edit_files.bad_modules`). The cost showed
on 2026-10-30: `dynSubgrid_driver` calls `dyncrop_interp`, whose host writes (crop landunit and
column weights, fertilizer) nobody synced to the device because the analysis had no events for
the call.

`generate(outdir)` runs the checkout's genf90.pl on every template under ELM_SRC and SHR_SRC
into `outdir` (skipping up-to-date outputs), dropping the `# N "template"` line markers genf90
emits: the preprocessor maps line markers to the file being read, so the lines under a marker
naming the template would be dropped ("Missing END MODULE"). `fortran_modules.get_filename_from_module`
searches `dynamic_globals.genf90_dir` after the checkout, so the generated modules resolve
like any other; modules that stay in `bad_modules` (ncdio_pio, shr_assert_mod, shr_infnan_mod)
stay excluded.
"""

from __future__ import annotations

import logging
import re
import subprocess as sp
from pathlib import Path
from typing import Optional

from spel.scripts.config import E3SM_SRCROOT, ELM_SRC, SHR_SRC

LINE_MARKER = re.compile(r'^#\s*\d+\s+"')
SCRIPT_CANDIDATES = ("cime/CIME/non_py/externals/genf90/genf90.pl", "cime/src/externals/genf90/genf90.pl")


def genf90_script(srcroot: Path = E3SM_SRCROOT) -> Optional[Path]:
    """The checkout's genf90.pl, or None."""
    for rel in SCRIPT_CANDIDATES:
        p = Path(srcroot) / rel
        if p.is_file():
            return p
    return None


def templates(elm_src: Path = ELM_SRC, shr_src: Path = SHR_SRC) -> list[Path]:
    """Every `*.F90.in` of the ELM and share sources (external models left out)."""
    out = [p for p in Path(elm_src).rglob("*.F90.in") if "external_models" not in p.parts]
    out += list(Path(shr_src).glob("*.F90.in"))
    return sorted(out)


def strip_markers(text: str) -> str:
    return "\n".join(l for l in text.split("\n") if not LINE_MARKER.match(l))


def generate(outdir: Path, logger: Optional[logging.Logger] = None,
             srcroot: Path = E3SM_SRCROOT, files: Optional[list[Path]] = None) -> list[Path]:
    """
    Generate `<outdir>/<stem>.F90` for each template (default: `templates()`); outputs newer
    than their template are kept. Returns the generated paths (empty when the checkout has no
    genf90.pl: the templates then stay unavailable, as before).
    """
    log = logger or logging.getLogger("SPEL")
    script = genf90_script(srcroot)
    if script is None:
        log.warning(f"no genf90.pl under {srcroot}: genf90 templates stay unavailable to the analysis")
        return []
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    done: list[Path] = []
    for tpl in (files if files is not None else templates()):
        out = outdir / tpl.name[: -len(".in")]
        if out.exists() and out.stat().st_mtime >= tpl.stat().st_mtime:
            done.append(out)
            continue
        res = sp.run(["perl", str(script), str(tpl)], capture_output=True, text=True)
        if res.returncode != 0 or not res.stdout.strip():
            log.warning(f"genf90 failed on {tpl}: {res.stderr.strip()[:200]}")
            continue
        out.write_text(strip_markers(res.stdout))
        done.append(out)
    log.info(f"genf90: {len(done)} generated modules in {outdir}")
    return done
