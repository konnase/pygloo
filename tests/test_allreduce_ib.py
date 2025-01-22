import numpy as np
import os
import time
import shutil
import torch
import pygloo

world_size = int(os.getenv("WORLD_SIZE", default=1))
rank = int(os.getenv("RANK", default=0))
ib_device = os.getenv("IB_DEVICE", default="mlx5_0")

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
    # Perform rendezvous for TCP pairs
    dev = pygloo.transport.ibverbs.CreateDevice(attr)

    fileStore = pygloo.rendezvous.FileStore(fileStore_path)

    context.connectFullMesh(fileStore, dev)

    sendbuf = np.array([[1,2,3],[1,2,3]], dtype=np.float32)
    sendptr = sendbuf.ctypes.data
    data_size = sendbuf.size if isinstance(sendbuf, np.ndarray) else sendbuf.numpy().size
    print(f"rank {rank} sends {sendbuf}")

    # sendbuf = torch.Tensor([[1,2,3],[1,2,3]]).float()
    # recvbuf = torch.zeros_like(sendbuf)
    # sendptr = sendbuf.data_ptr()
    # recvptr = recvbuf.data_ptr()

    datatype = pygloo.glooDataType_t.glooFloat32
    pygloo.allreduce_ring(context, sendptr, data_size, datatype)

    print(f"rank {rank} receives {sendbuf}")
    ## example output
    # (pid=30445) rank 0 sends [[1. 2. 3.]
    # (pid=30445)              [1. 2. 3.]],
    # (pid=30445)     receives [[2. 4. 6.]
    # (pid=30445)              [2. 4. 6.]]
    # (pid=30446) rank 1 sends [[1. 2. 3.]
    # (pid=30446)              [1. 2. 3.]],
    # (pid=30446)     receives [[2. 4. 6.]
    # (pid=30446)              [2. 4. 6.]]

if __name__ == "__main__":
    print(f"rank {rank} of world_size {world_size}")
    test_allreduce(rank, world_size, "/tmp/pygloo/allreduce_ib")
