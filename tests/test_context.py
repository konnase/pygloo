import numpy as np
import os
import time
import shutil
import torch
import pygloo

world_size = int(os.getenv("WORLD_SIZE", default=1))
rank = int(os.getenv("RANK", default=0))

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

    context = pygloo.rendezvous.Context(rank, world_size, 2)
    print(f"first rank {context.rank}, size {context.size} connects to peers")

    pygloo.test_context(context)
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
    test_allreduce(rank, world_size, "/tmp/pygloo/allreduce_tcp")
    
