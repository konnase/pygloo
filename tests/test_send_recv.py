import numpy as np
import os
import time
import shutil
import torch
import pygloo

master_addr = os.getenv('MASTER_ADDR', default="localhost")
gloo_port = int(os.getenv('GLOO_PORT', default=29700))
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

    # store = pygloo.rendezvous.FileStore(fileStore_path)
    store = pygloo.rendezvous.TCPStore(master_addr, gloo_port, world_size, 1 if rank==0 else 0)

    context.connectFullMesh(store, dev)

    # sendbuf = np.array([[1,2,3],[1,2,3]], dtype=np.float32)
    # sendbuf += rank
    # recvbuf = np.zeros_like(sendbuf, dtype=np.float32)
    # sendptr = sendbuf.ctypes.data
    # recvptr = recvbuf.ctypes.data
    # data_size = sendbuf.size if isinstance(sendbuf, np.ndarray) else sendbuf.numpy().size

    data_count = int(1024**3/4) # 1GB
    # data_count = data_count * 5
    # data_count = 3
    sendbuf = torch.ones(data_count, dtype=torch.float32)
    sendbuf += rank
    recvbuf = torch.zeros_like(sendbuf)
    sendptr = sendbuf.data_ptr()
    recvptr = recvbuf.data_ptr()
    data_size = sendbuf.numel()

    print(f"rank {rank} sends {data_size} elements: {sendbuf}")

    if rank == 0:
        peer = 1
        sd = pygloo.SenderFloat(context, sendptr, data_size, peer)
    else:
        peer = 0
        rc = pygloo.RecverFloat(context, recvptr, data_size, peer)


    # warmup
    warmup_iter = 10
    for i in range(warmup_iter):
        if rank == 0:
            sd.send(0, 2 ** 20, 0)
            sd.waitSend()
        else:
            rc.recv()

    # 模拟数据变化
    sendbuf += 10

    # rank 0 send, rank 1 recv
    start = time.time()
    last_time = start
    iter = 10
    for i in range(iter):
        if rank == 0:
            sd.send(0, 2 ** 20, 0)
            sd.waitSend()
        else:
            rc.recv()
        now_time = time.time()
        elapsed_time = now_time - last_time
        last_time = now_time
        print(f"Iter: {i}, time: {elapsed_time:.3f} s")
    total_time = time.time() - start
    bw = data_count * 4 * iter / total_time / 1024 / 1024 / 1024
    print(f"rank {rank} wait recv time: {total_time / iter:.3f} s")
    print(f"average bandwidth: {bw:.3f} GB/s")

    # release sender and recver
    if rank == 0:
        del sd # Sender or Recver is a wrapper of C++ object, so we need to delete it manually
    else:
        del rc # Sender or Recver is a wrapper of C++ object, so we need to delete it manually
    del context
    del store

    print(f"rank {rank} receives {recvbuf}")

    zero_count = torch.sum(recvbuf == 0).item()
    print(f"元素等于 0 的个数: {zero_count}")

    ## example output
    # rank 1 of world_size 2
    # rank 1 using ib device mlx5_2
    # Type: St10shared_ptrIN4gloo9transport7ContextEE
    # rank 1 sends 268435456 elements: tensor([2., 2., 2.,  ..., 2., 2., 2.])
    # Iter: 0, time: 0.059 s
    # Iter: 1, time: 0.058 s
    # Iter: 2, time: 0.058 s
    # Iter: 3, time: 0.058 s
    # Iter: 4, time: 0.058 s
    # Iter: 5, time: 0.058 s
    # Iter: 6, time: 0.058 s
    # Iter: 7, time: 0.058 s
    # Iter: 8, time: 0.058 s
    # Iter: 9, time: 0.058 s
    # rank 1 wait recv time: 0.058 s
    # average bandwidth: 17.299 GB/s
    # rank 1 receives tensor([11., 11., 11.,  ..., 11., 11., 11.])

if __name__ == "__main__":
    print(f"rank {rank} of world_size {world_size}")
    try:
        test_send_recv(rank, world_size, file_path)
    except Exception as e:
        print(f"rank {rank} error {e}")
