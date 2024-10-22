#!/bin/bash

# Just the main OFIQ "cmake --build" part from build.sh.

build_dir=build/build_linux

while [ -n "$1" ]
do  
if [ "$1" = "--os" ]; then
    shift
    if [ "$1" = "macos" ]; then
        build_dir=build/build_mac
    else
        echo "$1" is a not a supported OS
        exit
    fi
else
    echo "$1" is not a supported argument
    exit
fi
shift
done

cd ../
cmake --build $build_dir --target install -j 8

echo "Building finished"
