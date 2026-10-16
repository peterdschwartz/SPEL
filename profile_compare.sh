#!/bin/bash
# Multicore profiling comparison of a reference (master) unit test and its
# OpenACC refactor, with master threaded the way ELM threads it.
#
# ELM runs its physics in an OpenMP loop over clumps (elm_drv): each thread runs
# whole routines serially on its own clump. SPEL_CLUMPS=N reproduces that in a
# unit test (N clumps of whole input sets, one OpenMP thread per clump). The
# configurations, all built with nvfortran (-fast):
#
#   master_off   master, ACC=OFF, 1 clump                      serial baseline
#   master_cN    master, ACC=OPENMP, N clumps x 1 thread       ELM's CPU threading
#   acc_off      refactor, ACC=OFF, 1 clump                    serial refactor
#   acc_cN       refactor, ACC=OPENMP, N clumps x 1 thread     same strategy as master
#   acc_mcN      refactor, ACC=MULTICORE, 1 clump x N threads  loop-level threading
#   ompN         (--omp-case) acc2omp translation, ACC=OPENMP, 1 clump x N threads
#
# acc_cN vs master_cN isolates the code changes; acc_mcN vs master_cN compares
# the two threading strategies on the same columns and cores.
#
# Every run uses SPEL_PROFILE=1 (no device->host copies, no output files) under
# Nsight Systems. The headline time is the NVTX range 'compute' (wall time of the
# clump loop, per time step); the routines' own ranges run inside the clump
# threads, so for N clumps they time one clump (their max shows load imbalance).
# Run it on the machine you want numbers for (e.g. inside an salloc on a
# Perlmutter compute node).
set -euo pipefail

usage() {
	cat <<EOF
Usage: $(basename "$0") [options] MASTER_CASE ACC_CASE

Cases are names under unit-tests/ (or paths to case directories).

Options:
  -n, --nsets N        copies of the input sites (default: 16). Clump runs need
                       N_clumps <= nsets (clumps are made of whole sets)
  -c, --cores "LIST"   thread counts (default: powers of two up to the physical
                       core count, and that count)
  -s, --steps N        stop after N time steps (SPEL_MAX_STEPS; default: all)
  -o, --outdir DIR     reports and summary (default: unit-tests/profiles/compare-<acc>-nsets<N>-<date>)
  --skip "LIST"        configurations to leave out: any of master_c acc_c acc_mc
  --omp-case CASE      also profile CASE built with ACC=OPENMP, 1 clump x N threads
                       (run \`spel acc2omp --case CASE\` first)
  --ranges "A B"       routine NVTX ranges to report besides 'compute' (default: those
                       pushed inside 'compute' in main.F90)
  --env FILE           source FILE first (modules, PKG_CONFIG_PATH/LD_LIBRARY_PATH for the
                       nvfortran-built netcdf-fortran)
  --link-inputs        if unit-tests/input-data/<case> is missing for the acc/omp case, link
                       the master case's spel-*.nc files into it
  -h, --help

Environment:
  LAUNCH      prefix for each run, e.g. LAUNCH="srun -n 1 -c 128 --cpu-bind=cores"
  NSYS        nsys executable (default: from PATH, else next to nvfortran)
  Thread placement variables (OMP_PROC_BIND, OMP_PLACES, ACC_BIND, ...) are passed
  through; set them for clump and multicore runs alike, e.g.
  OMP_PROC_BIND=spread OMP_PLACES=cores.

Strong scaling is one run (fixed --nsets); for weak scaling run it again with
nsets proportional to the core count.

Example (Perlmutter CPU node):
  salloc -N 1 -C cpu -q interactive -t 01:00:00
  ./profile_compare.sh --env ~/spel_nvhpc_env.sh -n 128 soiltemperature soiltemp_acc
EOF
}

SPEL_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NSETS=16
CORES=""
STEPS=""
OUTDIR=""
OMP_CASE=""
RANGES=""
ENV_FILE=""
LINK_INPUTS=0
SKIP=""

while [[ $# -gt 0 ]]; do
	case "$1" in
	-n | --nsets) NSETS=$2; shift 2 ;;
	-c | --cores) CORES=$2; shift 2 ;;
	-s | --steps) STEPS=$2; shift 2 ;;
	-o | --outdir) OUTDIR=$2; shift 2 ;;
	--skip) SKIP=$2; shift 2 ;;
	--omp-case) OMP_CASE=$2; shift 2 ;;
	--ranges) RANGES=$2; shift 2 ;;
	--env) ENV_FILE=$2; shift 2 ;;
	--link-inputs) LINK_INPUTS=1; shift ;;
	-h | --help) usage; exit 0 ;;
	-*) echo "unknown option $1" >&2; usage >&2; exit 2 ;;
	*) break ;;
	esac
done
[[ $# -eq 2 ]] || { usage >&2; exit 2; }
for s in $SKIP; do
	case $s in master_c | acc_c | acc_mc) ;; *) echo "--skip: unknown configuration '$s'" >&2; exit 2 ;; esac
done
skipped() { [[ " $SKIP " == *" $1 "* ]]; }

if [[ -n $ENV_FILE ]]; then
	# shellcheck disable=SC1090
	source "$ENV_FILE"
fi

case_dir() {
	if [[ -d $1 ]]; then (cd "$1" && pwd); else echo "$SPEL_ROOT/unit-tests/$1"; fi
}
MASTER_DIR=$(case_dir "$1")
ACC_DIR=$(case_dir "$2")
OMP_DIR=""
if [[ -n $OMP_CASE ]]; then OMP_DIR=$(case_dir "$OMP_CASE"); fi
for d in "$MASTER_DIR" "$ACC_DIR" $OMP_DIR; do
	[[ -f $d/main.F90 ]] || { echo "not a unit-test case: $d" >&2; exit 1; }
	if ! grep -q "SPEL_CLUMPS" "$d/main.F90"; then
		echo "$d predates SPEL_CLUMPS: regenerate it with \`spel create\`" >&2
		exit 1
	fi
done

# --- tools -------------------------------------------------------------------
command -v nvfortran >/dev/null || { echo "nvfortran not found (load the NVHPC module or use --env)" >&2; exit 1; }
if [[ -z ${NSYS:-} ]]; then
	NSYS=$(command -v nsys || true)
	nvbin=$(dirname "$(readlink -f "$(command -v nvfortran)")")
	if [[ -z $NSYS && -x $nvbin/nsys ]]; then NSYS=$nvbin/nsys; fi
fi
[[ -n ${NSYS:-} ]] || { echo "nsys not found (set NSYS)" >&2; exit 1; }
pkg-config --exists netcdf-fortran || { echo "pkg-config can't find netcdf-fortran (use --env)" >&2; exit 1; }

# --- inputs ------------------------------------------------------------------
data_dir() { echo "$SPEL_ROOT/unit-tests/input-data/$(basename "$1")"; }
MASTER_DATA=$(data_dir "$MASTER_DIR")
for d in "$ACC_DIR" $OMP_DIR; do
	dd=$(data_dir "$d")
	if ! compgen -G "$dd/spel-inputs*.nc" >/dev/null; then
		if [[ $LINK_INPUTS -eq 1 ]]; then
			mkdir -p "$dd"
			for f in "$MASTER_DATA"/spel-*.nc; do ln -sf "$f" "$dd/"; done
			echo "linked $MASTER_DATA/spel-*.nc into $dd"
		else
			echo "no input data in $dd (use --link-inputs to use the master case's)" >&2
			exit 1
		fi
	fi
done

# --- defaults ----------------------------------------------------------------
PHYS=$(lscpu | awk -F: '/^Core\(s\) per socket/{c=$2} /^Socket\(s\)/{s=$2} END{print c*s}')
if [[ -z $CORES ]]; then
	CORES=""
	n=1
	while ((n < PHYS)); do CORES+="$n "; n=$((n * 2)); done
	CORES+="$PHYS"
fi
CLUMP_CORES=""
for n in $CORES; do
	if ((n <= NSETS)); then CLUMP_CORES+="$n "; fi
done
if [[ $CLUMP_CORES != "$CORES " ]]; then
	echo "NOTE: clump runs need N <= nsets ($NSETS): master_c/acc_c only for N in ($CLUMP_CORES)"
fi
if [[ -z $RANGES ]]; then
	# NVTX ranges pushed between spel_prof_push('compute') and its pop
	RANGES=$(awk '
		/spel_prof_push\(.compute.\)/ {inside=1; depth=0; next}
		inside && /spel_prof_push\(/ {s=$0; sub(/.*spel_prof_push\(./, "", s); sub(/[^A-Za-z0-9_].*/, "", s); print s; depth++; next}
		inside && /spel_prof_pop\(/ {if (depth == 0) inside=0; else depth--}
	' "$ACC_DIR/main.F90" | sort -u | tr '\n' ' ')
fi
RANGES="compute $RANGES"
[[ -n $OUTDIR ]] || OUTDIR="$SPEL_ROOT/unit-tests/profiles/compare-$(basename "$ACC_DIR")-nsets$NSETS-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUTDIR"
OUTDIR=$(cd "$OUTDIR" && pwd)
RESULTS=$OUTDIR/results.csv
echo "label,case,acc,clumps,threads,range,instances,total_s,avg_ms,med_ms,max_ms" >"$RESULTS"

{
	echo "date:      $(date -Is)"
	echo "host:      $(hostname)"
	echo "cpu:       $(lscpu | awk -F: '/^Model name/{gsub(/^ +/, "", $2); print $2}') ($PHYS physical cores)"
	echo "nvfortran: $(nvfortran --version | awk 'NF{print; exit}')"
	echo "nsys:      $("$NSYS" --version)"
	echo "spel:      $(git -C "$SPEL_ROOT" rev-parse --short HEAD 2>/dev/null || echo ?)"
	echo "master:    $MASTER_DIR"
	echo "acc:       $ACC_DIR"
	if [[ -n $OMP_DIR ]]; then echo "omp:       $OMP_DIR"; fi
	echo "nsets=$NSETS steps=${STEPS:-all} cores=($CORES) ranges=($RANGES) skip=($SKIP)"
	echo "LAUNCH=${LAUNCH:-}"
	for v in OMP_PROC_BIND OMP_PLACES ACC_BIND; do
		if [[ -n ${!v:-} ]]; then echo "$v=${!v}"; fi
	done
} | tee "$OUTDIR/info.txt"

# --- build / run -------------------------------------------------------------
build() { # case_dir acc
	local dir=$1 acc=$2 log=$OUTDIR/build-$(basename "$1")-$2.log
	echo "== build $(basename "$dir") ACC=$acc"
	# answer check_config.sh's reconfigure prompt; `yes` exits on SIGPIPE, so
	# only check_config.sh's status counts
	if ! (set +o pipefail; cd "$dir" && yes | SPEL_FC=nvfortran SPEL_ACC=$acc SPEL_DBG=OFF ./check_config.sh) >"$log" 2>&1; then
		echo "   build FAILED (see $log)"
		return 1
	fi
}

profile() { # label case_dir acc clumps threads
	local label=$1 dir=$2 acc=$3 clumps=$4 threads=$5
	local log=$OUTDIR/$label.log
	echo "== run $label (clumps=$clumps threads=$threads)"
	local -a envs=(SPEL_PROFILE=1 SPEL_CLUMPS="$clumps" ACC_NUM_CORES="$threads" OMP_NUM_THREADS="$threads")
	if [[ -n $STEPS ]]; then envs+=(SPEL_MAX_STEPS="$STEPS"); fi
	# shellcheck disable=SC2086
	if ! (cd "$dir" && env "${envs[@]}" ${LAUNCH:-} "$NSYS" profile --trace=nvtx,osrt \
		--sample=none --cpuctxsw=none --force-overwrite=true -o "$OUTDIR/$label" \
		./build/elmtest "$NSETS") >"$log" 2>&1; then
		echo "   run FAILED (see $log)"
		return 1
	fi
	"$NSYS" stats -q --report nvtx_sum --format csv --output - "$OUTDIR/$label.nsys-rep" 2>>"$log" |
		python3 -c '
import csv, sys
label, case, acc, clumps, threads, ranges = sys.argv[1:7]
wanted = set(ranges.split())
rows = [r for r in csv.reader(sys.stdin) if r]
head = next((i for i, r in enumerate(rows) if "Range" in r), None)
if head is None:
    sys.exit(f"{label}: no nvtx_sum table")
col = {name: i for i, name in enumerate(rows[head])}
for r in rows[head + 1 :]:
    name = r[col["Range"]].split(":")[-1]
    if name in wanted:
        total, n = float(r[col["Total Time (ns)"]]), int(r[col["Instances"]])
        avg, med = float(r[col["Avg (ns)"]]), float(r[col["Med (ns)"]])
        mx = float(r[col["Max (ns)"]])
        print(f"{label},{case},{acc},{clumps},{threads},{name},{n},{total/1e9:.6f},"
              f"{avg/1e6:.6f},{med/1e6:.6f},{mx/1e6:.6f}")
' "$label" "$(basename "$dir")" "$acc" "$clumps" "$threads" "$RANGES" >>"$RESULTS"
}

# master: serial, then ELM-style clump threading
build "$MASTER_DIR" OFF && profile master_off "$MASTER_DIR" OFF 1 1 || true
if ! skipped master_c && build "$MASTER_DIR" OPENMP; then
	for n in $CLUMP_CORES; do profile "master_c$n" "$MASTER_DIR" OPENMP "$n" "$n" || true; done
fi
# refactor: serial, clump threading, loop-level (OpenACC multicore) threading
build "$ACC_DIR" OFF && profile acc_off "$ACC_DIR" OFF 1 1 || true
if ! skipped acc_c && build "$ACC_DIR" OPENMP; then
	for n in $CLUMP_CORES; do profile "acc_c$n" "$ACC_DIR" OPENMP "$n" "$n" || true; done
fi
if ! skipped acc_mc && build "$ACC_DIR" MULTICORE; then
	for n in $CORES; do profile "acc_mc$n" "$ACC_DIR" MULTICORE 1 "$n" || true; done
fi
if [[ -n $OMP_DIR ]] && build "$OMP_DIR" OPENMP; then
	for n in $CORES; do profile "omp$n" "$OMP_DIR" OPENMP 1 "$n" || true; done
fi

# --- summary -----------------------------------------------------------------
python3 - "$RESULTS" <<'EOF' | tee "$OUTDIR/summary.txt"
import csv, re, sys

rows = list(csv.DictReader(open(sys.argv[1])))
SERIES = {  # label prefix -> (description, serial label of the same code)
    "master_c": ("master, N clumps x 1 thread", "master_off"),
    "acc_c": ("refactor, N clumps x 1 thread", "acc_off"),
    "acc_mc": ("refactor, 1 clump x N threads (OpenACC)", "acc_off"),
    "omp": ("refactor, 1 clump x N threads (acc2omp)", "acc_off"),
}


def series(label):
    m = re.match(r"(master_c|acc_c|acc_mc|omp)(\d+)$", label)
    return (m.group(1), int(m.group(2))) if m else (label, 1)


compute = {r["label"]: r for r in rows if r["range"] == "compute"}
if not compute:
    sys.exit("no results")

print("\nWall time per time step (NVTX 'compute': all clumps)")
print("  vs master    : master serial (master_off) / this")
print("  vs master@N  : master with ELM's clump threading on the same N threads / this")
print("  vs own 1 thr : the same code on 1 thread / this\n")
hdr = f"{'config':<11} {'threads':>7} {'avg ms':>9} {'vs master':>10} {'vs master@N':>12} {'vs own 1 thr':>13}  description"
print(hdr)
print("-" * len(hdr))
base = compute.get("master_off")
for label, r in compute.items():
    kind, n = series(label)
    avg = float(r["avg_ms"])
    desc, serial = SERIES.get(kind, ("serial", label))
    ref_n = compute.get(f"master_c{n}") or (base if n == 1 else None)
    own1 = compute.get(f"{kind}1") or compute.get(serial)
    fmt = lambda ref, w: f"{float(ref['avg_ms']) / avg:{w-1}.2f}x" if ref else f"{'-':>{w}}"
    print(f"{label:<11} {r['threads']:>7} {avg:9.3f} {fmt(base, 10)} {fmt(ref_n, 12)} {fmt(own1, 13)}  {desc}")

routines = sorted({r["range"] for r in rows} - {"compute"})
for rng in routines:
    print(f"\nNVTX '{rng}' per call (inside the clump loop: one clump's share for N clumps)")
    print(f"{'config':<11} {'clumps':>6} {'calls':>7} {'avg ms':>9} {'max ms':>9}")
    for r in (r for r in rows if r["range"] == rng):
        print(f"{r['label']:<11} {r['clumps']:>6} {r['instances']:>7} {float(r['avg_ms']):9.3f} {float(r['max_ms']):9.3f}")
EOF
echo
echo "results: $RESULTS"
echo "reports: $OUTDIR/*.nsys-rep"
