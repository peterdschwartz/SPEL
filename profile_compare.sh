#!/bin/bash
# Multicore profiling comparison of a reference (master) unit test and its
# OpenACC refactor: times only the NVTX ranges of the routines under test
# (main.F90 pushes one per generated call inside the 'compute' range).
#
#   master case : nvfortran, ACC=OFF
#   acc case    : nvfortran, ACC=OFF, then ACC=MULTICORE for each core count
#   omp case    : (optional) nvfortran, ACC=OPENMP (translated by `spel acc2omp`)
#
# Every run uses SPEL_PROFILE=1 (no device->host copies, no output files) under
# Nsight Systems; the times come from `nsys stats --report nvtx_sum`.
# Run it on the machine you want numbers for (e.g. inside an salloc on a
# Perlmutter compute node).
set -euo pipefail

usage() {
	cat <<EOF
Usage: $(basename "$0") [options] MASTER_CASE ACC_CASE

Cases are names under unit-tests/ (or paths to case directories).

Options:
  -n, --nsets N        copies of the input sites (default: 16)
  -c, --cores "LIST"   ACC_NUM_CORES values for the multicore runs
                       (default: powers of two up to the physical core count, and that count)
  -s, --steps N        stop after N time steps (SPEL_MAX_STEPS; default: all)
  -o, --outdir DIR     reports and summary (default: unit-tests/profiles/compare-<acc>-nsets<N>-<date>)
  --omp-case CASE      also profile CASE built with ACC=OPENMP (run \`spel acc2omp --case CASE\` first)
                       with OMP_NUM_THREADS set to each core count
  --ranges "A B"       NVTX ranges to time (default: those pushed inside 'compute' in main.F90)
  --env FILE           source FILE first (modules, PKG_CONFIG_PATH/LD_LIBRARY_PATH for the
                       nvfortran-built netcdf-fortran)
  --link-inputs        if unit-tests/input-data/<case> is missing for the acc/omp case, link
                       the master case's spel-*.nc files into it
  -h, --help

Environment:
  LAUNCH      prefix for each run, e.g. LAUNCH="srun -n 1 -c 128 --cpu-bind=cores"
  NSYS        nsys executable (default: from PATH, else next to nvfortran)
  Thread placement variables (ACC_BIND, OMP_PROC_BIND, OMP_PLACES, ...) are passed through.

Example (Perlmutter CPU node):
  salloc -N 1 -C cpu -q interactive -t 01:00:00
  ./profile_compare.sh --env ~/spel_nvhpc_env.sh -n 16 soiltemperature soiltemp_acc
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

while [[ $# -gt 0 ]]; do
	case "$1" in
	-n | --nsets) NSETS=$2; shift 2 ;;
	-c | --cores) CORES=$2; shift 2 ;;
	-s | --steps) STEPS=$2; shift 2 ;;
	-o | --outdir) OUTDIR=$2; shift 2 ;;
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
if [[ -z $RANGES ]]; then
	# NVTX ranges pushed between spel_prof_push('compute') and its pop
	RANGES=$(awk '
		/spel_prof_push\(.compute.\)/ {inside=1; depth=0; next}
		inside && /spel_prof_push\(/ {s=$0; sub(/.*spel_prof_push\(./, "", s); sub(/[^A-Za-z0-9_].*/, "", s); print s; depth++; next}
		inside && /spel_prof_pop\(/ {if (depth == 0) inside=0; else depth--}
	' "$ACC_DIR/main.F90" | sort -u | tr '\n' ' ')
	[[ -n $RANGES ]] || { echo "no NVTX ranges found in $ACC_DIR/main.F90 (use --ranges)" >&2; exit 1; }
fi
[[ -n $OUTDIR ]] || OUTDIR="$SPEL_ROOT/unit-tests/profiles/compare-$(basename "$ACC_DIR")-nsets$NSETS-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUTDIR"
OUTDIR=$(cd "$OUTDIR" && pwd)
RESULTS=$OUTDIR/results.csv
echo "label,case,acc,threads,range,instances,total_s,avg_ms,med_ms" >"$RESULTS"

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
	echo "nsets=$NSETS steps=${STEPS:-all} cores=($CORES) ranges=($RANGES)"
	echo "LAUNCH=${LAUNCH:-}"
	for v in ACC_BIND OMP_PROC_BIND OMP_PLACES; do
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

profile() { # label case_dir acc threads
	local label=$1 dir=$2 acc=$3 threads=$4
	local log=$OUTDIR/$label.log
	echo "== run $label"
	local -a envs=(SPEL_PROFILE=1 ACC_NUM_CORES="$threads" OMP_NUM_THREADS="$threads")
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
label, case, acc, threads, ranges = sys.argv[1:6]
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
        print(f"{label},{case},{acc},{threads},{name},{n},{total/1e9:.6f},{avg/1e6:.6f},{med/1e6:.6f}")
' "$label" "$(basename "$dir")" "$acc" "$threads" "$RANGES" >>"$RESULTS"
}

build "$MASTER_DIR" OFF && profile master_off "$MASTER_DIR" OFF 1 || true
build "$ACC_DIR" OFF && profile acc_off "$ACC_DIR" OFF 1 || true
if build "$ACC_DIR" MULTICORE; then
	for n in $CORES; do profile "acc_mc$n" "$ACC_DIR" MULTICORE "$n" || true; done
fi
if [[ -n $OMP_DIR ]] && build "$OMP_DIR" OPENMP; then
	for n in $CORES; do profile "omp$n" "$OMP_DIR" OPENMP "$n" || true; done
fi

# --- summary -----------------------------------------------------------------
python3 - "$RESULTS" <<'EOF' | tee "$OUTDIR/summary.txt"
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
for rng in sorted({r["range"] for r in rows}):
    rs = [r for r in rows if r["range"] == rng]
    by = {r["label"]: r for r in rs}
    base = by.get("master_off")
    mc1 = by.get("acc_mc1") or by.get("acc_off")
    print(f"\nNVTX range '{rng}' (avg per call)")
    print(f"{'config':<12} {'threads':>7} {'calls':>6} {'total s':>9} {'avg ms':>9} {'vs master':>9} {'vs 1 thread':>11}")
    for r in rs:
        avg = float(r["avg_ms"])
        vs_master = f"{float(base['avg_ms']) / avg:8.2f}x" if base else "-"
        same_kind = r["label"].startswith(("acc_mc", "omp"))
        ref = by.get("omp1") if r["label"].startswith("omp") else mc1
        vs_one = f"{float(ref['avg_ms']) / avg:10.2f}x" if (same_kind and ref) else "-"
        print(f"{r['label']:<12} {r['threads']:>7} {r['instances']:>6} {float(r['total_s']):9.3f} "
              f"{avg:9.3f} {vs_master:>9} {vs_one:>11}")
EOF
echo
echo "results: $RESULTS"
echo "reports: $OUTDIR/*.nsys-rep"
