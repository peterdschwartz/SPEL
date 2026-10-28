#!/bin/bash
# config
BUILD_DIR="build"
debug=ON
compiler="gfortran"
# OpenACC target: OFF, MULTICORE (nvfortran; threads from ACC_NUM_CORES), GPU, or
# OPENMP (after `spel acc2omp`; threads from OMP_NUM_THREADS).
# SPEL_ACC / SPEL_FC / SPEL_DBG override acc / compiler / debug,
# e.g. SPEL_ACC=MULTICORE SPEL_FC=nvfortran SPEL_DBG=OFF
acc=OFF
acc=${SPEL_ACC:-$acc}
compiler=${SPEL_FC:-$compiler}
debug=${SPEL_DBG:-$debug}

DESIRED_COMPILER=$(which $compiler)

CACHE_FILE="$BUILD_DIR/CMakeCache.txt"

# check if cache exists
if [[ ! -f "$CACHE_FILE" ]]; then
	echo "No CMakeCache.txt found. Running fresh cmake configuration..."
	mkdir -p "$BUILD_DIR"
	cmake -S . -B "$BUILD_DIR" -DDBG=$debug -DACC=$acc -DCMAKE_Fortran_COMPILER=$DESIRED_COMPILER
fi

# read current values from cache
CURRENT_COMPILER=$(grep '^CMAKE_Fortran_COMPILER:STRING=' "$CACHE_FILE" | cut -d= -f2)
curr_dbg=$(grep '^DBG:BOOL=' "$CACHE_FILE" | cut -d= -f2)
curr_acc=$(grep '^ACC:STRING=' "$CACHE_FILE" | cut -d= -f2)
curr_acc=${curr_acc:-OFF} # caches from before the ACC setting
# pkg-config results are cached, so a changed PKG_CONFIG_PATH (e.g. switching
# between gfortran and nvfortran netcdf-fortran builds) needs a reconfigure.
curr_ncf=$(grep '^NetCDFF_INCLUDEDIR:INTERNAL=' "$CACHE_FILE" | cut -d= -f2)
desired_ncf=$(pkg-config --variable=includedir netcdf-fortran 2>/dev/null)
desired_ncf=${desired_ncf:-$curr_ncf}

# report mismatches
if [[ "$CURRENT_COMPILER" != "$DESIRED_COMPILER" ]] || [[ "$curr_dbg" != "$debug" ]] || [[ "$curr_acc" != "$acc" ]] || [[ "$curr_ncf" != "$desired_ncf" ]]; then
	echo "  Detected mismatch in CMake configuration:"
	echo "    Current compiler   : $CURRENT_COMPILER"
	echo "    Desired compiler   : $DESIRED_COMPILER"
	echo "    Current build type : $curr_dbg"
	echo "    Desired build type : $debug"
	echo "    Current OpenACC    : $curr_acc"
	echo "    Desired OpenACC    : $acc"
	echo "    Current netcdf-fortran : $curr_ncf"
	echo "    Desired netcdf-fortran : $desired_ncf"
	echo
	read -p "Delete build directory and reconfigure? (y/N): " confirm
	if [[ "$confirm" == "y" || "$confirm" == "Y" ]]; then
		rm -rf "$BUILD_DIR"
		mkdir -p "$BUILD_DIR"
		cmake -S . -B "$BUILD_DIR" -DDBG=$debug -DACC=$acc -DCMAKE_Fortran_COMPILER=$DESIRED_COMPILER
	else
		echo "Leaving existing build directory unchanged."
	fi
else
	echo "✅ CMake configuration matches desired settings."
fi


# parallel build: CMake orders the Fortran modules (SPEL_MAKE_JOBS overrides the job count)
make -C "$BUILD_DIR" -j "${SPEL_MAKE_JOBS:-$(nproc)}"
