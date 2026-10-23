#!/bin/bash
# Create (and optionally build/run) an ELM case.
# Defaults reproduce the original single-site uELM setup; --preset crop-perf
# gives the crop-model performance setup on Perlmutter or Chrysalis. See --help.
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
datm_y1=""; datm_y2=""
ntasks=""
nthrds=""
threaded=0
max_tasks_per_node=""
max_mpitasks_per_node=""
rest_n=""
rest_option=""
queue=""
wallclock="0:10:00"
auto_name=0
share=0
compressed_cft=0
daily_hist=0
remove_zero_weight_urban=0
finidat=""
compact=0
name_suffix=""
case "$(hostname -f)" in
   chr*) default_cft_dir=/lcrc/group/e3sm/$USER/crop_inputdata/crop_reduced_inputdata ;;
   *) default_cft_dir=/pscratch/sd/g/gbisht/crop_reduced_inputdata ;;
esac
compressed_cft_dir=${COMPRESSED_CFT_DIR:-$default_cft_dir}

usage() {
   cat <<USAGE
Usage: $(basename "$0") [options]
  --case NAME          case name                 (default: $casename)
  --group NAME         case group subdirectory   (default: $case_group)
  --scratch DIR        scratch root              (default: $scratch)
  --srcroot DIR        E3SM source root          (default: $E3SM_SRCROOT)
  --mach NAME          machine                   (default: $mach)
  --compiler NAME      compiler, '' for the machine default (default: $compiler)
  --compset NAME       compset                   (default: $compset)
  --with-crops         --compset I1850GSWCNPCROPTOP
  --no-crops           --compset I1850GSWCNPTOP
  --res NAME           resolution; u01 and t-lr25 expand to u01_u01 and
                       t-lr25_t-lr25             (default: $res)
  --user-mods DIR      testmods dir, relative to elm testmods_dirs or absolute ('' for none)
                                                 (default: $user_mods)
  --config NAME        cn | fates | test | none  (default: $config)
  --stop-n N           STOP_N                    (default: $stop_n)
  --stop-option OPT    STOP_OPTION               (default: $stop_option)
  --rest-n N           REST_N                    (default: unchanged)
  --rest-option OPT    REST_OPTION               (default: unchanged)
  --ntasks N           NTASKS                    (default: unchanged)
  --nthrds N           NTHRDS (all components); implies --threaded
  --threaded           FORCE_BUILD_SMP=TRUE (one executable for any NTASKS x NTHRDS)
  --max-tasks-per-node N  MAX_TASKS_PER_NODE (e.g. 128 on pm-cpu: one thread per core)
  --max-mpitasks-per-node N  MAX_MPITASKS_PER_NODE
  --queue NAME         JOB_QUEUE                 (default: unchanged)
  --wallclock T        JOB_WALLCLOCK_TIME and USER_REQUESTED_WALLTIME
                                                 (default: $wallclock)
  --no-debug           DEBUG=FALSE
  --daily-history      hist_nhtfrq=-24, hist_mfilt=5 (part of --preset crop-perf; --no-daily-history undoes it)
  --compressed-cft     t-lr25 only: use the paramfile and surface dataset reduced
                       to the 16 CFTs unique in mergetoclmpft (from
                       \$COMPRESSED_CFT_DIR, default $compressed_cft_dir)
                       and append '.compressed_cft' to an --auto-name case name.
                       Any t-lr25 case takes the t-lr25 domain file and MOSART
                       input from that directory when \$DIN_LOC_ROOT lacks them
  --finidat FILE       finidat in user_nl_elm (e.g. a spun-up restart)
  --compact            apply the SPEL compact layout at build time: `spel compact rewrite
                       --srcroot SRC --reanalyze -o CASE/SourceMods/src.elm` (adds .compact
                       to an --auto-name case name)
  --remove-zero-weight-urban
                       add 'run_zero_weight_urban = .false.' to user_nl_elm
                       (older E3SM hashes lack this namelist option)
  --auto-name          name the case <date>.<res>.<compset>.<mach>.NTASKS<n>.<git hash>
                       (overrides --case)
  --name-suffix S      append '.S' to an --auto-name case name
  --share              chmod -R og+rX the case directory when done
  --preset crop-perf   ELM crop-model performance case: --user-mods '' --config none
                       --no-debug --ntasks 1024 --rest-n 10 --rest-option ndays --stop-n 5
                       --stop-option ndays --queue debug --wallclock 00:20:00 --daily-history
                       --group E3SM_testings --auto-name --share --build, and per machine
                       (from the SPEL machine file, `spel machine vars`; else by hostname)
                         Perlmutter: --mach pm-cpu --compiler '' --scratch \$SCRATCH
                         Chrysalis:  --mach chrysalis --compiler intel
                                     --scratch /lcrc/globalscratch/\$USER
                                     NTHRDS=1 --max-mpitasks-per-node 64 (the
                                     default layout is 32 tasks x 2 threads)
                       Options after it override it; combine with
                       --res and --with-crops/--no-crops.
  --refcase DIR        hybrid start from a refcase saved by save_refcase.sh
                       (e.g. \$DIN_LOC_ROOT/spel-refcases/<compset>/<case>/<date>)
  --start-date DATE    RUN_STARTDATE (with --refcase: default = the refcase date; also for a
                       startup run from a --finidat restart, e.g. 0003-07-01)
  --datm-years Y1 Y2   DATM_CLMNCEP_YR_START/END (ALIGN=1): the forcing years the start date
                       must fall in (model year n reads forcing year Y1+n-1)
  --keep               reuse an existing case instead of recreating it
  --build              run case.setup and case.build
  --no-build           undo --build (after --preset)
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
      --preset)
         case "$2" in
            crop-perf)
               user_mods=""; config=none; debug=FALSE
               ntasks=1024; rest_n=10; rest_option=ndays; stop_n=5; stop_option=ndays
               queue=debug; wallclock=00:20:00; daily_hist=1
               case_group=E3SM_testings
               auto_name=1; share=1; build=1
               # machine defaults from the SPEL machine file (`spel machine vars`:
               # e3sm_machine, scratch, [casegen] compiler/nthrds/max_mpitasks_per_node),
               # else by hostname
               machine_vars=$(command -v spel >/dev/null && spel machine vars 2>/dev/null || true)
               if [ -n "$machine_vars" ]; then
                  eval "$machine_vars"
                  mach=${SPEL_E3SM_MACHINE:?machine file sets no e3sm_machine}
                  scratch=${SPEL_SCRATCH:?machine file sets no scratch}
                  compiler=${SPEL_CASEGEN_COMPILER-$compiler}
                  nthrds=${SPEL_CASEGEN_NTHRDS-$nthrds}
                  [ -n "$nthrds" ] && threaded=1   # a machine-file nthrds means a threaded (FORCE_BUILD_SMP) build
                  max_mpitasks_per_node=${SPEL_CASEGEN_MAX_MPITASKS_PER_NODE-$max_mpitasks_per_node}
               else
                  case "$(hostname -f)" in
                     chr*)
                        mach=chrysalis; compiler=intel; nthrds=1; max_mpitasks_per_node=64
                        scratch=/lcrc/globalscratch/$USER ;;
                     *)
                        mach=pm-cpu; compiler=""
                        scratch=${SCRATCH:?SCRATCH is not set} ;;
                  esac
               fi ;;
            *) echo "Unknown preset: $2" >&2; exit 1 ;;
         esac
         shift ;;
      --mach) mach=$2; shift ;;
      --compiler) compiler=$2; shift ;;
      --compset) compset=$2; shift ;;
      --with-crops) compset=I1850GSWCNPCROPTOP ;;
      --no-crops) compset=I1850GSWCNPTOP ;;
      --res)
         case "$2" in
            u01) res=u01_u01 ;;
            t-lr25) res=t-lr25_t-lr25 ;;
            *) res=$2 ;;
         esac
         shift ;;
      --user-mods) user_mods=$2; shift ;;
      --config) config=$2; shift ;;
      --stop-n) stop_n=$2; shift ;;
      --stop-option) stop_option=$2; shift ;;
      --rest-n) rest_n=$2; shift ;;
      --rest-option) rest_option=$2; shift ;;
      --ntasks) ntasks=$2; shift ;;
      --nthrds) nthrds=$2; threaded=1; shift ;;
      --threaded) threaded=1 ;;
      --max-tasks-per-node) max_tasks_per_node=$2; shift ;;
      --max-mpitasks-per-node) max_mpitasks_per_node=$2; shift ;;
      --no-build) build=0 ;;
      --queue) queue=$2; shift ;;
      --wallclock) wallclock=$2; shift ;;
      --compressed-cft) compressed_cft=1 ;;
      --daily-history) daily_hist=1 ;;
      --no-daily-history) daily_hist=0 ;;
      --remove-zero-weight-urban) remove_zero_weight_urban=1 ;;
      --finidat) finidat=$(realpath -s "$2"); shift ;;
      --compact) compact=1 ;;
      --name-suffix) name_suffix=$2; shift ;;
      --auto-name) auto_name=1 ;;
      --share) share=1 ;;
      --no-debug) debug=FALSE ;;
      --refcase) refcase=$(realpath "$2"); shift ;;
      --start-date) start_date=$2; shift ;;
      --datm-years) datm_y1=$2; datm_y2=$3; shift 2 ;;
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
compressed_cft_paramfile=$compressed_cft_dir/clm_params_31pfts_c261015.nc
compressed_cft_fsurdat=$compressed_cft_dir/surfdata_t-oi6to18-lr25_31pfts_simyr1850_c261015.nc

case "$config" in
   cn|fates|test|none) ;;
   *) echo "Unknown config: $config" >&2; exit 1 ;;
esac

if [ "$compressed_cft" -eq 1 ]; then
   if [ "$res" != t-lr25_t-lr25 ]; then
      echo "--compressed-cft is only supported with --res t-lr25" >&2; exit 1
   fi
   for f in "$compressed_cft_paramfile" "$compressed_cft_fsurdat"; do
      [ -f "$f" ] || { echo "--compressed-cft input file not found: $f" >&2; exit 1; }
   done
fi

if [ "$auto_name" -eq 1 ]; then
   git_hash=$(git -C "$src_path" log -n 1 --format=%h)
   casename=$(date +%Y%m%d).$res.$compset.$mach.NTASKS${ntasks:-default}${nthrds:+x$nthrds}.$git_hash
   if [ "$compressed_cft" -eq 1 ]; then
      casename=$casename.compressed_cft
   fi
   if [ "$compact" -eq 1 ]; then
      casename=$casename.compact
   fi
   if [ -n "$name_suffix" ]; then
      casename=$casename.$name_suffix
   fi
fi

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
      --mach "$mach" ${compiler:+--compiler "$compiler"} \
      --compset "$compset" \
      --res "$res" \
      --handle-preexisting-dirs r \
      --srcroot "$src_path" \
      ${user_mods:+--user-mods-dir "$user_mods"}

   cd "${casedir}"
   ./xmlchange DEBUG=$debug
   if [ -n "$ntasks" ]; then
      ./xmlchange NTASKS="$ntasks"
   fi
   if [ -n "$nthrds" ]; then
      ./xmlchange NTHRDS="$nthrds"
   fi
   if [ "$threaded" -eq 1 ]; then
      ./xmlchange FORCE_BUILD_SMP=TRUE   # case.setup recomputes BUILD_THREADED from NTHRDS unless forced
   fi
   if [ -n "$max_tasks_per_node" ]; then
      ./xmlchange MAX_TASKS_PER_NODE="$max_tasks_per_node"
   fi
   if [ -n "$max_mpitasks_per_node" ]; then
      ./xmlchange MAX_MPITASKS_PER_NODE="$max_mpitasks_per_node"
   fi

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

   if [ "$daily_hist" -eq 1 ]; then
      printf " hist_nhtfrq = -24\n hist_mfilt  = 5\n" >> user_nl_elm
   fi

   if [ "$remove_zero_weight_urban" -eq 1 ]; then
      echo "run_zero_weight_urban = .false." >> user_nl_elm
   fi

   if [ "$compressed_cft" -eq 1 ]; then
      echo "
      paramfile = '$compressed_cft_paramfile'
      fsurdat = '$compressed_cft_fsurdat'
      " >> user_nl_elm
   fi

   if [ "$res" == t-lr25_t-lr25 ]; then
      # t-lr25 inputs the machine's inputdata may lack (e.g. on Chrysalis):
      # taken from $COMPRESSED_CFT_DIR when they are there
      din=$(./xmlquery --value DIN_LOC_ROOT)
      domain=domain.lnd.t-oi6to18-lr25.20261012.nc
      if [ ! -f "$din/share/domains/$domain" ] && [ -f "$compressed_cft_dir/$domain" ]; then
         ./xmlchange ATM_DOMAIN_PATH="$compressed_cft_dir"
         ./xmlchange LND_DOMAIN_PATH="$compressed_cft_dir"
         echo " frivinp_mesh = '$compressed_cft_dir/$domain'" >> user_nl_mosart
      fi
      mosart=MOSART_u.oi6to18.lr25.nc
      if [ ! -f "$din/rof/mosart/$mosart" ] && [ -f "$compressed_cft_dir/$mosart" ]; then
         echo " frivinp_rtm = '$compressed_cft_dir/$mosart'" >> user_nl_mosart
      fi
   fi

   if [ -n "$finidat" ]; then
      echo " finidat = '$finidat'" >> user_nl_elm
   fi

   if [ "$compact" -eq 1 ]; then
      # the compact layout is a generated transform applied per case, not kept in the sources
      command -v spel >/dev/null || { echo "--compact needs spel on PATH" >&2; exit 1; }
      spel compact rewrite --srcroot "$src_path" --reanalyze -o "$casedir/SourceMods/src.elm" \
         || { echo "spel compact rewrite failed" >&2; exit 1; }
   fi
fi

cd "${casedir}"
./xmlchange STOP_N="$stop_n"
./xmlchange STOP_OPTION="$stop_option"
./xmlchange JOB_WALLCLOCK_TIME="$wallclock"
./xmlchange USER_REQUESTED_WALLTIME="$wallclock"
if [ -n "$rest_n" ]; then
   ./xmlchange REST_N="$rest_n"
fi
if [ -n "$rest_option" ]; then
   ./xmlchange REST_OPTION="$rest_option"
fi
if [ -n "$queue" ]; then
   ./xmlchange JOB_QUEUE="$queue"
fi

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
if [ -z "$refcase" ] && [ -n "$start_date" ]; then
   ./xmlchange RUN_STARTDATE="$start_date"
fi
if [ -n "$datm_y1" ]; then
   ./xmlchange DATM_CLMNCEP_YR_START="$datm_y1",DATM_CLMNCEP_YR_END="$datm_y2",DATM_CLMNCEP_YR_ALIGN=1
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

if [ "$share" -eq 1 ]; then
   chmod -R og+rX "$casedir"
fi

if [ "$submit" -eq 1 ]; then
   ./case.submit
fi
