#!/bin/bash
# Save a finished case's restart set as a reusable CIME refcase, keyed by compset:
#   $DIN_LOC_ROOT/spel-refcases/<compset alias>/<case>/<refdate>/
# Start a new case from it with: uELM_casegen.sh --refcase <that dir> ...
set -euo pipefail

usage() {
   cat <<USAGE
Usage: $(basename "$0") CASEDIR [--date YYYY-MM-DD] [--root DIR] [--force]
  CASEDIR       finished CIME case (reads its RUNDIR rpointer files)
  --date D      restart date to save (default: date in rpointer.lnd)
  --root DIR    store root (default: \$DIN_LOC_ROOT/spel-refcases)
  --force       overwrite an existing saved refcase
The saved directory is printed as 'REFCASE=<path>'.
USAGE
}

casedir="" date="" root="" force=0
while [ $# -gt 0 ]; do
   case "$1" in
      --date) date=$2; shift ;;
      --root) root=$2; shift ;;
      --force) force=1 ;;
      -h|--help) usage; exit 0 ;;
      -*) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
      *) casedir=$1 ;;
   esac
   shift
done
[ -n "$casedir" ] || { usage >&2; exit 1; }
casedir=$(realpath "$casedir")
cd "$casedir"
q() { ./xmlquery --value "$1"; }

case_name=$(q CASE)
rundir=$(q RUNDIR)
compset=$(q COMPSET)
srcroot=$(q SRCROOT)
root=${root:-$(q DIN_LOC_ROOT)/spel-refcases}

# compset alias (falls back to the long name)
alias=$(python3 - "$srcroot" "$compset" <<'PY'
import sys, glob, xml.etree.ElementTree as ET
src, lname = sys.argv[1], sys.argv[2]
for f in glob.glob(f"{src}/components/*/cime_config/config_compsets.xml"):
    for c in ET.parse(f).getroot().iter("compset"):
        ln = (c.findtext("lname") or "").strip()
        if ln and lname.startswith(ln):
            print(c.findtext("alias").strip()); sys.exit()
print(lname)
PY
)

lnd_rst=$(head -1 "$rundir/rpointer.lnd" | xargs basename)
if [ -z "$date" ]; then
   date=$(sed -E 's/.*\.r\.([0-9]{4}-[0-9]{2}-[0-9]{2})-[0-9]{5}\.nc/\1/' <<<"$lnd_rst")
fi
tod=$(sed -E 's/.*\.r\.[0-9-]{10}-([0-9]{5})\.nc/\1/' <<<"$lnd_rst")
stamp="${date}-${tod}"
[ "$lnd_rst" == "${case_name}.elm.r.${stamp}.nc" ] || {
   echo "rpointer.lnd points at $lnd_rst, not a ${stamp} restart of ${case_name}" >&2; exit 1; }

dest="$root/$alias/$case_name/$date"
if [ -e "$dest" ]; then
   [ "$force" -eq 1 ] || { echo "$dest exists (use --force)" >&2; exit 1; }
   rm -rf "$dest"
fi
mkdir -p "$dest"

shopt -s nullglob
files=("$rundir"/${case_name}.*.r*.${stamp}.* "$rundir"/rpointer.*)
for f in "${files[@]}"; do cp -p "$f" "$dest/"; done
[ -f "$dest/$lnd_rst" ] || { echo "missing $lnd_rst in $rundir" >&2; exit 1; }

python3 - "$casedir" "$dest" "$alias" "$date" "$tod" <<'PY'
import json, subprocess, sys, datetime
from pathlib import Path
casedir, dest, alias, date, tod = sys.argv[1:]
def q(v):
    r = subprocess.run(["./xmlquery", "--value", v], cwd=casedir, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None
xml = ["CASE", "COMPSET", "GRID", "MACH", "COMPILER", "SRCROOT", "RUN_TYPE", "RUN_STARTDATE",
       "STOP_OPTION", "STOP_N", "CALENDAR", "ATM_NCPL", "ELM_BLDNML_OPTS", "ELM_CONFIG_OPTS",
       "ELM_ACCELERATED_SPINUP", "DATM_MODE", "DATM_CLMNCEP_YR_START", "DATM_CLMNCEP_YR_END",
       "DATM_CLMNCEP_YR_ALIGN", "ATM_DOMAIN_FILE", "LND_DOMAIN_FILE", "ATM_DOMAIN_PATH",
       "LND_DOMAIN_PATH", "DIN_LOC_ROOT"]
src = q("SRCROOT")
git = lambda *a: subprocess.run(["git", "-C", src, *a], capture_output=True, text=True).stdout.strip()
lnd_in = Path(q("RUNDIR"), "lnd_in")
nml = {}
for line in lnd_in.read_text().splitlines() if lnd_in.exists() else []:
    if "=" in line and not line.lstrip().startswith("!"):
        k, v = line.split("=", 1)
        nml[k.strip()] = v.strip()
manifest = {
    "refcase": q("CASE"), "refdate": date, "reftod": tod, "compset_alias": alias,
    "saved": datetime.datetime.now().isoformat(timespec="seconds"),
    "source_casedir": casedir,
    "e3sm": {"commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD")},
    "xml": {v: q(v) for v in xml},
    "lnd_in": nml,
    "user_nl_elm": Path(casedir, "user_nl_elm").read_text(),
    "files": sorted(p.name for p in Path(dest).iterdir()),
}
Path(dest, "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
PY
echo "REFCASE=$dest"
