#!/bin/bash

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd $ROOT_DIR/..

mkdir -p build
cd build
cmake ..
make