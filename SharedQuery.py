#!/usr/bin/env python
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# 共享内存大小
win_size = 1

# 在每个进程中创建共享内存
win = MPI.Win.Allocate_shared(win_size * MPI.INT.Get_size(), MPI.INT.Get_size() , comm=comm)
# 将共享内存映射到本地内存
buf, itemsize = win.Shared_query(0)

# 获取共享内存的地址
shared_mem_addr, _ = win.Shared_query(rank)

# 将共享内存映射到numpy数组
print(win_size//itemsize,)
shared_count = np.ndarray(buffer=buf, dtype='i', shape=(1,))

# 在rank为0的进程中初始化共享内存
if rank == 0:
    shared_count[0] = 0

# 同步所有进程中的共享内存
comm.Barrier()

from time import sleep 
for i in range(3):
    
    win.Lock(0)
    
    # sleep(0.01)
    before = shared_count[0]
    sleep(1)
    print('seelp')
    shared_count[0] += 1
    after = shared_count[0]
    print('before', before, 'after', after, before-after == -1)
    win.Unlock(0)
# 同步所有进程中的共享内存
# comm.Barrier()

# 同步所有进程中的共享内存
comm.Barrier()

# 输出每个进程中的共享内存值及地址
print(f"Rank {rank}: shared_count = {shared_count[0]}, shared_mem_addr = {shared_mem_addr}")