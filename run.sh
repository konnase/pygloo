re_install=${1:-1}

if [ "$re_install" -eq 1 ]; then
    pip install dist/pygloo-0.0.1-cp310-cp310-linux_x86_64.whl --force-reinstall
fi

export LD_LIBRARY_PATH=/usr/local/lib
torchrun --nproc-per-node=2 tests/test_allreduce_tcp.py
# torchrun --nproc-per-node=2 tests/test_context.py