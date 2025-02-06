# pygloo

Pygloo provides Python bindings for [gloo](https://github.com/facebookincubator/gloo), forked from [Pygloo](https://github.com/ray-project/pygloo),
and changed the management tools from bazel to cmake.
It is implemented using [pybind11](https://github.com/pybind/pybind11).

## Requirements
```python
Python >= 3.6
```

## Installation

### Prerequisites

Install hiredis and ibverbs first. If you don't need ibverbs, just ignore it.
```bash
# install hiredis and ibverbs
sudo apt update
sudo apt install libibverbs-dev
sudo apt install -y libhiredis-dev
```

Then build gloo from source to use ibverbs and redis. 
If you don't need ibverbs, just remove `-DUSE_IBVERBS=1` from cmake command.
```bash
# install gloo
git clone https://github.com/konnase/gloo.git
cd gloo
mkdir -p build
cd build
cmake -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
    -DUSE_IBVERBS=1 -DBUILD_BENCHMARK=1 -DUSE_REDIS=1 \
    -DBUILD_SHARED_LIBS=1 \
    ../ 
make
make install
```

### Building from source
One can build pygloo from source if none of released wheels fit with the development environment.

Pygloo uses CMakeLists to automatically manange dependencies and compilation.
To compile from source, build and install pygloo following this command:
```python
python setup.py install
```

## Testing
See `tests` directory. Ignore test files written by ray project.

## Example
An example for send/recv, check [test_send_recv.py](./tests/test_send_recv.py).


## License
Gloo is licensed under the Apache License, Version 2.0.