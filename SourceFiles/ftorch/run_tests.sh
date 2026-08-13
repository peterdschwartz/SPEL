#!/bin/bash
set -x
nargs=$#
fc=gfortran

if [[ $narg -gt 0 || ! -d "build" ]]; then
	if [[ $1 == "clean" ]]; then
		rm -rf build
	fi
		
	cmake -S . -B build \
		-DCMAKE_PREFIX_PATH="$HOME/.local/ftorch" \
		-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
		-DCMAKE_Fortran_COMPILER=$fc
fi
cmake --build build 

