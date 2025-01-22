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

    sendbuf = np.array([[1,2,3],[1,2,3]], dtype=np.float32)
    recvbuf = np.zeros_like(sendbuf, dtype=np.float32)
    sendptr = sendbuf.ctypes.data
    recvptr = recvbuf.ctypes.data
    print(f"rank {rank} sends {sendbuf}")
    data_size = sendbuf.size if isinstance(sendbuf, np.ndarray) else sendbuf.numpy().size

    if rank == 0:
        peer = 1
    else:
        peer = 0
    
    sr = pygloo.SendRecverFloat(context, sendptr, recvptr, data_size, peer)
    sr.send()
    sr.recv()
    sr.waitSend()

    # sendbuf = torch.Tensor([[1,2,3],[1,2,3]]).float()
    # recvbuf = torch.zeros_like(sendbuf)
    # sendptr = sendbuf.data_ptr()
    # recvptr = recvbuf.data_ptr()

    print(f"rank {rank} receives {recvbuf}")
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
    try:
        test_allreduce(rank, world_size, file_path)
    except Exception as e:
        print(f"rank {rank} error {e}")
