import numpy as np
import os
import time
import shutil
import torch
import pygloo

import torch.distributed as dist

master_addr = os.getenv('MASTER_ADDR', default="localhost")
master_port = int(os.getenv('MASTER_PORT', default=29700))
gloo_port = int(os.getenv('GLOO_PORT', default=29701))
world_size = int(os.getenv("WORLD_SIZE", default=1))
rank = int(os.getenv("RANK", default=0))
use_ib = int(os.getenv("USE_IB", default=1))
ib_device = os.getenv("IB_DEVICE", default="mlx5_0")
file_path = os.getenv("FILE_PATH", default="/mnt/public/liqingping/tests/dlckpt_file_store")

def test_send_recv(rank, world_size, fileStore_path):
    '''
    rank  # Rank of this process within list of participating processes
    world_size  # Number of participating processes
    '''
    backend = 'gloo'
    init_method = f"tcp://{master_addr}:{master_port}"
    print(f'init_method:{init_method}, {master_addr}, {master_port}')
    dist.init_process_group(init_method=init_method, rank=rank, world_size=world_size, backend=backend)
    print(f'torch初始化完')

    if rank==0:
        if os.path.exists(fileStore_path):
            shutil.rmtree(fileStore_path)
        os.makedirs(fileStore_path)
    else: time.sleep(0.5)

    context = pygloo.rendezvous.Context(rank, world_size)

    hostname = os.getenv("HOSTNAME", default="localhost")
    attr = pygloo.transport.ibverbs.attr(ib_device, 1, 1) if use_ib == 1 else pygloo.transport.tcp.attr(hostname)
    dev = pygloo.transport.ibverbs.CreateDevice(attr) if use_ib == 1 else pygloo.transport.tcp.CreateDevice(attr)
    if use_ib == 1:
        print(f"rank {rank} using ib device {ib_device}")

    tmp_store = pygloo.rendezvous.FileStore(fileStore_path)
    # tmp_store = pygloo.rendezvous.TCPStore(master_addr, gloo_port, world_size, 1 if rank==0 else 0)
    store = pygloo.rendezvous.PrefixStore(str(world_size), tmp_store)

    context.connectFullMesh(store, dev)
    slot = context.nextSlot(1)

    # sendbuf = np.array([[1,2,3],[1,2,3]], dtype=np.float32)
    # sendbuf += rank
    # recvbuf = np.zeros_like(sendbuf, dtype=np.float32)
    # sendptr = sendbuf.ctypes.data
    # recvptr = recvbuf.ctypes.data
    # data_size = sendbuf.size if isinstance(sendbuf, np.ndarray) else sendbuf.numpy().size

    buffer_size = int(1024**3/4) # 1GB
    # buffer_size = int(3)
    num_buffers = 1
    data_count = buffer_size * num_buffers
    sendbuf = torch.ones(data_count, dtype=torch.float32)
    for i in range(num_buffers):
        sendbuf[i*buffer_size:(i+1)*buffer_size] += i
    sendbuf += rank
    recvbuf = torch.zeros_like(sendbuf)
    sendptr = sendbuf.data_ptr()
    recvptr = recvbuf.data_ptr()
    data_size = sendbuf.numel()


    for i in range(0, world_size, 1):
        peer_rank = (i+1) % world_size
        if i == rank:
            start_time = time.time()
            sd = pygloo.SenderFloat(context, sendptr, slot, data_size, peer_rank)
            # print(f"rank: {rank}, create SenderFloat time: {time.time() - start_time:.3f} s")
            for j in range(0, num_buffers, 1):
                print(f"rank: {rank} -> {peer_rank}, sends {data_size} elements")
                sd.send(j*buffer_size*4, buffer_size*4, j*buffer_size*4) # (local_offset, size, remote_offset)
                sd.waitSend()
            del sd # Sender or Recver is a wrapper of C++ object, so we need to delete it manually

        elif rank == peer_rank:
            start_time = time.time()
            rc = pygloo.RecverFloat(context, recvptr, slot, data_size, i)
            # print(f"rank: {rank}, create RecverFloat time: {time.time() - start_time:.3f} s")
            for j in range(0, num_buffers, 1):
                print(f"rank: {rank} <- {i}, recvs {data_size} elements")
                rc.recv()
            del rc # Sender or Recver is a wrapper of C++ object, so we need to delete it manually
        dist.barrier()
        print(f"round: {i} finished.")
       
    del context
    del store

    # print(f"rank: {rank}, receives {recvbuf}")

if __name__ == "__main__":
    print(f"rank {rank} of world_size {world_size}")
    try:
        test_send_recv(rank, world_size, file_path)
    except Exception as e:
        print(f"rank {rank} error {e}")
