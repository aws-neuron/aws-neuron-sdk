#!/bin/bash

set -e

LOCAL=$(pwd)/..
mkdir -p build
pushd build
(VERBOSE=1 cmake -DCMAKE_PREFIX_PATH=$LOCAL/libtorch .. && VERBOSE=1 cmake --build . --config Release) || exit 1
popd
cp -f build/example_app ../example-app
