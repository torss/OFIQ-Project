#!/bin/bash

# Just the cmake --build part from build.sh.

build_dir=build/build_linux
cd ../
cmake --build $build_dir --target install -j 8

echo "Building finished"
