#!/bin/bash
# config
BUILD_DIR="build"
debug=ON
compiler="gfortran"
# OpenACC target: OFF, MULTICORE (nvfortran; threads from ACC_NUM_CORES) or GPU.
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

# report mismatches
if [[ "$CURRENT_COMPILER" != "$DESIRED_COMPILER" ]] || [[ "$curr_dbg" != "$debug" ]] || [[ "$curr_acc" != "$acc" ]]; then
	echo "  Detected mismatch in CMake configuration:"
	echo "    Current compiler   : $CURRENT_COMPILER"
	echo "    Desired compiler   : $DESIRED_COMPILER"
	echo "    Current build type : $curr_dbg"
	echo "    Desired build type : $debug"
	echo "    Current OpenACC    : $curr_acc"
	echo "    Desired OpenACC    : $acc"
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


make -C "$BUILD_DIR"
