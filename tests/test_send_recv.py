import numpy as np
import os
import time
import shutil
import torch
import pygloo

world_size = int(os.getenv("WORLD_SIZE", default=1))
rank = int(os.getenv("RANK", default=0))
use_ib = int(os.getenv("USE_IB", default=1))
ib_device = os.getenv("IB_DEVICE", default="mlx5_0")
ip_addr = os.getenv("IP_ADDR", default="localhost")
file_path = os.getenv("FILE_PATH", default="/mnt/public/liqingping/opensource/gloo/tmp/file_store")

def test_send_recv(rank, world_size, fileStore_path):
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

    attr = pygloo.transport.ibverbs.attr(ib_device, 1, 1) if use_ib == 1 else pygloo.transport.tcp.attr(ip_addr)
    dev = pygloo.transport.ibverbs.CreateDevice(attr) if use_ib == 1 else pygloo.transport.tcp.CreateDevice(attr)
    if use_ib == 1:
        print(f"rank {rank} using ib device {ib_device}")

    fileStore = pygloo.rendezvous.FileStore(fileStore_path)

    context.connectFullMesh(fileStore, dev)

    # sendbuf = np.array([[1,2,3],[1,2,3]], dtype=np.float32)
    # sendbuf += rank
    # recvbuf = np.zeros_like(sendbuf, dtype=np.float32)
    # sendptr = sendbuf.ctypes.data
    # recvptr = recvbuf.ctypes.data
    # data_size = sendbuf.size if isinstance(sendbuf, np.ndarray) else sendbuf.numpy().size

    data_count = int(1024**3/4) # 1GB
    # data_count = data_count * 5
    # data_count = 3
    sendbuf = torch.rand(data_count, dtype=torch.float32)
    sendbuf += rank
    recvbuf = torch.zeros_like(sendbuf)
    sendptr = sendbuf.data_ptr()
    recvptr = recvbuf.data_ptr()
    data_size = sendbuf.numel()

    print(f"rank {rank} sends {data_size} elements: {sendbuf}")

    if rank == 0:
        peer = 1
    else:
        peer = 0
    
    sr = pygloo.SendRecverFloat(context, sendptr, recvptr, data_size, peer)
    # sr.setDebug(1)

    # both send/recv
    for _ in range(1):
        start = time.time()
        sr.send()
        # print(f"rank {rank} send time: {time.time() - start}")
        sr.recv()
        # print(f"rank {rank} recv time: {time.time() - start}")
        sr.waitSend()
        print(f"rank {rank} wait recv time: {time.time() - start}")

    # # rank 0 send, rank 1 recv
    # start = time.time()
    # if rank == 0:
    #     sr.send()
    #     sr.waitSend()
    # else:
    #     sr.recv()
    # print(f"rank {rank} wait recv time: {time.time() - start}")

    del sr # SendRecver is a wrapper of C++ object, so we need to delete it manually


    print(f"rank {rank} receives {recvbuf}")

    zero_count = torch.sum(recvbuf == 0).item()
    print(f"元素等于 0 的个数: {zero_count}")

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
        test_send_recv(rank, world_size, file_path)
    except Exception as e:
        print(f"rank {rank} error {e}")
