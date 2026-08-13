#!/bin/bash 

E3SM_SRCROOT=$1

if [ -z $E3SM_SRCROOT ]; then
	echo "ERROR Didn't provide E3SM_SRCROOT"
        exit 1
fi

cd $E3SM_SRCROOT

COMMITHEAD="$(git log -n 1 | head -1)"

echo $COMMITHEAD
