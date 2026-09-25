#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <remote_host> <destination_path>"
    exit 1
fi

REMOTE_HOST="$1"
DEST_PATH="$2"

files="FUTConstantsMod.F90  nc_allocMod.F90  nc_io.F90  ReadWriteMod.F90"

if [ $REMOTE_HOST = "local" ]; then
    rsync -avR $files "${DEST_PATH}"
else
    rsync -avR  $files "${REMOTE_HOST}:${DEST_PATH}"
fi  
