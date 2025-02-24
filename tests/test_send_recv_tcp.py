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
ib_device = os.getenv("IB_DEVICE", default="mlx5_0")
file_path = os.getenv("FILE_PATH", default="/mnt/public/liqingping/tests/dlckpt_file_store")

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

    hostname = os.getenv("HOSTNAME", default="localhost")
    attr = pygloo.transport.tcp.attr(hostname)
    dev = pygloo.transport.tcp.CreateDevice(attr)

    tmp_store = pygloo.rendezvous.FileStore(fileStore_path)
    # tmp_store = pygloo.rendezvous.TCPStore(master_addr, gloo_port, world_size, 1 if rank==0 else 0)
    store = pygloo.rendezvous.PrefixStore(str(world_size), tmp_store)

    context.connectFullMesh(store, dev)

    data_count = int(1024**3/4) # 1GB

    if rank == 0:
        # sendbuf = np.array([[1,2,3],[1,2,3]], dtype=np.float32)
        # sendptr = sendbuf.ctypes.data

        sendbuf = torch.ones(data_count, dtype=torch.float32)
        sendptr = sendbuf.data_ptr()

        data_size = sendbuf.size if isinstance(sendbuf, np.ndarray) else sendbuf.numpy().size
        datatype = pygloo.glooDataType_t.glooFloat32
        peer = 1
        start_time = time.time()
        pygloo.send(context, sendptr, data_size, datatype, peer)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"rank {rank} sends {sendbuf} in {elapsed_time} seconds")

    elif rank == 1:
        # recvbuf = np.zeros((2,3), dtype=np.float32)
        # recvptr = recvbuf.ctypes.data

        recvbuf = torch.zeros(data_count, dtype=torch.float32)
        recvptr = recvbuf.data_ptr()

        data_size = recvbuf.size if isinstance(recvbuf, np.ndarray) else recvbuf.numpy().size
        datatype = pygloo.glooDataType_t.glooFloat32
        peer = 0

        start_time = time.time()
        pygloo.recv(context, recvptr, data_size, datatype, peer)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"rank {rank} receives {recvbuf} in {elapsed_time} seconds")
    else:
        raise Exception("Only support 2 process to test send function and recv function")

if __name__ == "__main__":
    print(f"rank {rank} of world_size {world_size}")
    try:
        test_send_recv(rank, world_size, file_path)
    except Exception as e:
        print(f"rank {rank} error {e}")
