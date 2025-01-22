# pygloo

Pygloo provides Python bindings for [gloo](https://github.com/facebookincubator/gloo).
It is implemented using [pybind11](https://github.com/pybind/pybind11).

## Requirements
```python
Python >= 3.6
```

## Installation

### Prerequisites

Install gloo and hiredis first.
```bash
# install gloo
git clone https://github.com/facebookincubator/gloo.git
cd gloo
mkdir -p build
cd build
cmake -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
    -DUSE_IBVERBS=1 -DBUILD_BENCHMARK=1 -DUSE_REDIS=1 \
    -DBUILD_SHARED_LIBS=1 \
    ../ 
make
make install

# install hiredis
sudo apt-get install -y libhiredis-dev
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
An example for send/recv.
```python
import numpy as np
import os
import time
import shutil
import torch
import pygloo

world_size = int(os.getenv("WORLD_SIZE", default=1))
rank = int(os.getenv("RANK", default=0))
ib_device = os.getenv("IB_DEVICE", default="mlx5_0")
file_path = os.getenv("FILE_PATH", default="/mnt/public/liqingping/opensource/gloo/tmp/file_store")

def test_allreduce(rank, world_size, fileStore_path):
    '''
    rank  # Rank of this process within list of participating processes
    world_size  # Number of participating processes
    '''
    if rank==0:
        if os.path.exists(fileStore_path):
            shutil.rmtree(fileStore_path)
        os.makedirs(fileStore_path)
    else: time.sleep(0.5)

    context = pygloo.rendezvous.Context(rank, world_size)

    attr = pygloo.transport.ibverbs.attr(ib_device, 1, 1)
    dev = pygloo.transport.ibverbs.CreateDevice(attr)
    # attr = pygloo.transport.tcp.attr("localhost")
    # dev = pygloo.transport.tcp.CreateDevice(attr)

    fileStore = pygloo.rendezvous.FileStore(fileStore_path)

    context.connectFullMesh(fileStore, dev)

    # sendbuf = np.array([[1,2,3],[1,2,3]], dtype=np.float32)
    # sendbuf += rank
    # recvbuf = np.zeros_like(sendbuf, dtype=np.float32)
    # sendptr = sendbuf.ctypes.data
    # recvptr = recvbuf.ctypes.data
    # data_size = sendbuf.size if isinstance(sendbuf, np.ndarray) else sendbuf.numpy().size

    sendbuf = torch.Tensor([[1,2,3],[1,2,3]]).float()
    sendbuf += rank
    recvbuf = torch.zeros_like(sendbuf)
    sendptr = sendbuf.data_ptr()
    recvptr = recvbuf.data_ptr()
    data_size = sendbuf.numel()

    print(f"rank {rank} sends {sendbuf}")

    if rank == 0:
        peer = 1
    else:
        peer = 0
    
    sr = pygloo.SendRecverFloat(context, sendptr, recvptr, data_size, peer)
    sr.send()
    sr.recv()
    sr.waitSend()


    print(f"rank {rank} receives {recvbuf}")
    ## example output
    # (pid=30445) rank 0 sends [[1. 2. 3.]
    # (pid=30445)              [1. 2. 3.]],
    # (pid=30445)     receives [[2. 4. 6.]
    # (pid=30445)              [2. 4. 6.]]
    # (pid=30446) rank 1 sends [[2. 4. 6.]
    # (pid=30446)              [2. 4. 6.]]
    # (pid=30446)     receives [[1. 2. 3.]
    # (pid=30446)              [1. 2. 3.]],

if __name__ == "__main__":
    print(f"rank {rank} of world_size {world_size}")
    try:
        test_allreduce(rank, world_size, file_path)
    except Exception as e:
        print(f"rank {rank} error {e}")

```


## License
Gloo is licensed under the Apache License, Version 2.0.