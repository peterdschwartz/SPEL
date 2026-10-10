#!/bin/bash
# Create (and optionally build/run) a single-site uELM case.
# Defaults reproduce the original hard-coded setup; see --help.
set -euo pipefail

parent_dir=${HOME}/climate-projects
scratch=${parent_dir}/e3sm-scratch
E3SM_SRCROOT=${E3SM_SRCROOT:-$parent_dir/E3SM}
config=cn
casename=canflux
case_group=fut
mach=mylaptop
compiler=gnu
compset=I1850GSWCNPRDCTCBC
res=ELM_USRDAT
user_mods=elm/kilocraft
stop_n=1
stop_option=nyears
debug=TRUE
keep=0
build=0
clean_lnd=0
submit=0
refcase=""
start_date=""

usage() {
   cat <<USAGE
Usage: $(basename "$0") [options]
  --case NAME          case name                 (default: $casename)
  --group NAME         case group subdirectory   (default: $case_group)
  --scratch DIR        scratch root              (default: $scratch)
  --srcroot DIR        E3SM source root          (default: $E3SM_SRCROOT)
  --mach NAME          machine                   (default: $mach)
  --compiler NAME      compiler                  (default: $compiler)
  --compset NAME       compset                   (default: $compset)
  --res NAME           resolution                (default: $res)
  --user-mods DIR      testmods dir, relative to elm testmods_dirs or absolute ('' for none)
                                                 (default: $user_mods)
  --config NAME        cn | fates | test         (default: $config)
  --stop-n N           STOP_N                    (default: $stop_n)
  --stop-option OPT    STOP_OPTION               (default: $stop_option)
  --no-debug           DEBUG=FALSE
  --refcase DIR        hybrid start from a refcase saved by save_refcase.sh
                       (e.g. \$DIN_LOC_ROOT/spel-refcases/<compset>/<case>/<date>)
  --start-date DATE    RUN_STARTDATE for --refcase (default: the refcase date)
  --keep               reuse an existing case instead of recreating it
  --build              run case.setup and case.build
  --clean-lnd          run 'case.build --clean lnd' before building
  --submit             run case.submit
  -h, --help           show this message
The case directory is printed as 'CASEDIR=<path>'.
USAGE
}

while [ $# -gt 0 ]; do
   case "$1" in
      --case) casename=$2; shift ;;
      --group) case_group=$2; shift ;;
      --scratch) scratch=$2; shift ;;
      --srcroot) E3SM_SRCROOT=$2; shift ;;
      --mach) mach=$2; shift ;;
      --compiler) compiler=$2; shift ;;
      --compset) compset=$2; shift ;;
      --res) res=$2; shift ;;
      --user-mods) user_mods=$2; shift ;;
      --config) config=$2; shift ;;
      --stop-n) stop_n=$2; shift ;;
      --stop-option) stop_option=$2; shift ;;
      --no-debug) debug=FALSE ;;
      --refcase) refcase=$(realpath "$2"); shift ;;
      --start-date) start_date=$2; shift ;;
      --keep) keep=1 ;;
      --build) build=1 ;;
      --clean-lnd) clean_lnd=1 ;;
      --submit) submit=1 ;;
      -h|--help) usage; exit 0 ;;
      *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
   esac
   shift
done

src_path=$(realpath "$E3SM_SRCROOT")
casedir="$scratch/$case_group/$casename"
output=$scratch/$case_group/output/$casename

if [ -n "$user_mods" ] && [[ "$user_mods" != /* ]]; then
   user_mods=$src_path/components/elm/cime_config/testdefs/testmods_dirs/$user_mods
fi

set -x
if [ "$keep" -eq 0 ] || [ ! -d "$casedir" ]; then
   rm -rf "$casedir" "$output"

   "$src_path"/cime/scripts/create_newcase --case "${casedir}" \
      --output-root "$output" \
      --mach "$mach" --compiler "$compiler" \
      --compset "$compset" \
      --res "$res" \
      --handle-preexisting-dirs r \
      --srcroot "$src_path" \
      ${user_mods:+--user-mods-dir "$user_mods"}

   cd "${casedir}"
   ./xmlchange JOB_WALLCLOCK_TIME="0:10:00"
   ./xmlchange USER_REQUESTED_WALLTIME="0:10:00"
   ./xmlchange DEBUG=$debug

   if [ "$config" == "fates" ]; then
      ./xmlchange ELM_BLDNML_OPTS=" -bgc fates -no-megan -no-drydep -nutrient cnp -nutrient_comp_pathway eca -soil_decomp century"
      echo "
      hist_nhtfrq=-5000
      hist_mfilt=1
      use_cn=.false.
      fates_parteh_mode = 2
      use_lch4          = .true.
      fates_spitfire_mode = 1
      suplphos = 'NONE'
      suplnitro = 'NONE'
      " > user_nl_elm
   fi

   if [ "$config" == "cn" ]; then
      echo "
      hist_nhtfrq=-5000
      hist_mfilt=1
      use_lch4          = .true.
      nu_com = 'RD'
      " >> user_nl_elm
   fi

   if [ "$config" == "test" ]; then
      echo "
      use_snicar_ad                 = .true.
      use_dust_snow_internal_mixing = .true.
      snow_shape                    = 'koch_snowflake'
      snicar_atm_type            = 'mid-latitude_winter'
      fsnowoptics                   = '\$DIN_LOC_ROOT/lnd/clm2/snicardata/snicar_optics_5bnd_mam_c211006.nc'
      " > user_nl_elm
   fi
fi

cd "${casedir}"
./xmlchange STOP_N="$stop_n"
./xmlchange STOP_OPTION="$stop_option"

if [ -n "$refcase" ]; then
   # refuse a refcase saved from an incompatible configuration
   read -r ref_case ref_date ref_tod < <(python3 - "$refcase" <<'PY'
import json, subprocess, sys
from pathlib import Path
m = json.loads(Path(sys.argv[1], "manifest.json").read_text())
q = lambda v: subprocess.run(["./xmlquery", "--value", v], capture_output=True, text=True).stdout.strip()
keys = ["COMPSET", "LND_DOMAIN_FILE", "ATM_DOMAIN_FILE", "CALENDAR", "ELM_BLDNML_OPTS"]
bad = [f"{k}: refcase {m['xml'].get(k)!r} != case {q(k)!r}" for k in keys
       if (m["xml"].get(k) or "").split() != q(k).split()]
if bad:
    sys.exit("refcase does not match this case:\n  " + "\n  ".join(bad))
print(m["refcase"], m["refdate"], m["reftod"])
PY
)
   ./xmlchange RUN_TYPE=hybrid
   ./xmlchange GET_REFCASE=TRUE
   ./xmlchange RUN_REFDIR="$refcase"
   ./xmlchange RUN_REFCASE="$ref_case"
   ./xmlchange RUN_REFDATE="$ref_date"
   ./xmlchange RUN_REFTOD="$ref_tod"
   ./xmlchange RUN_STARTDATE="${start_date:-$ref_date}"
fi
{ set +x; } 2>/dev/null
echo "CASEDIR=${casedir}"
set -x

if [ "$build" -eq 1 ]; then
   ./case.setup
   if [ "$clean_lnd" -eq 1 ]; then
      ./case.build --clean lnd
   fi
   ./case.build
fi

if [ "$submit" -eq 1 ]; then
   ./case.submit
fi
