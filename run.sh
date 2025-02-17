re_install=${1:-1}

if [ "$re_install" -eq 1 ]; then
    pip install dist/pygloo-0.0.1-cp310-cp310-linux_x86_64.whl --force-reinstall
fi

export LD_LIBRARY_PATH=/usr/local/lib
export USE_IB=1
# torchrun --nproc-per-node=2 tests/test_allreduce.py
torchrun --nproc-per-node=2 tests/test_send_recv.py
# valgrind --leak-check=full torchrun --nproc-per-node=2 tests/test_send_recv.py


# run independently on 2 nodes
# rank 0: MASTER_ADDR=jo-daspjlbtqjbdh3jl-worker-0 GLOO_PORT=29500 RANK=0 WORLD_SIZE=2 IB_DEVICE=mlx5_2 IP_ADDR=10.10.10.2 LD_LIBRARY_PATH=/usr/local/lib python test_send_recv.py
# rank 1: MASTER_ADDR=jo-daspjlbtqjbdh3jl-worker-0 GLOO_PORT=29500 RANK=1 WORLD_SIZE=2 IB_DEVICE=mlx5_2 IP_ADDR=10.10.10.2 LD_LIBRARY_PATH=/usr/local/lib python test_send_recv.py