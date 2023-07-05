#!/usr/bin/env python3
'''
Hdf5 data loader for pytorch.

References:
https://www.jianshu.com/p/ee4b76b32779

'''
import h5py
import numpy as np
from mpi4py import MPI
# import mpi_array

MPI_COMM_WORLD_RANK = MPI.COMM_WORLD.Get_rank()  # The process ID (integer 0-3 for 4-process run)
MPI_COMM_WORLD_SIZE = MPI.COMM_WORLD.Get_size()

print('MPI_COMM_WORLD_RANK', MPI_COMM_WORLD_RANK)
print('MPI_COMM_WORLD_SIZE', MPI_COMM_WORLD_SIZE)

class H5Dataset():
    def __init__(self, path_h5, dataset_name='dataset', cords_name='cords', r_or_w='r', \
        max_sample_num=1*10**5, dim_per_sample=(512,512,3)):
        '''
        MPI based multiple writer. 
        dataset: to save the tile array. 
        cords: to save the roi, x and y.
        '''
        self.img_dataset = h5py.File(path_h5, r_or_w, driver='mpio', comm=MPI.COMM_WORLD)
        self.dataset_name = dataset_name
        self.cords_name = cords_name

        self.cache_for_batch = [] # a 2-d list

        self.init_total_num_shared_between_processes()

        if r_or_w == 'w':
            self.dataset = self.img_dataset.create_dataset(self.dataset_name, (max_sample_num, *dim_per_sample), dtype=np.uint8)
            self.cords = self.img_dataset.create_dataset(self.cords_name, (max_sample_num,3), dtype=np.uint32)
            self.img_dataset.attrs['total_num'] = 0
            self.total_num_array = np.ndarray(buffer=self.buf, dtype='i', shape=(self.size_for_total_num_array,))
            self.total_num_array[:1] = 0
            # self.total_num_array = mpi_array.array([0])
            # self.total_num = self.img_dataset.attrs['total_num']
            # print('self.total_num is {}'.format(self.img_dataset.attrs['total_num']))
        elif r_or_w in ['r','a']:
            self.dataset = self.img_dataset[self.dataset_name]
            self.cords = self.img_dataset[self.cords_name]
            self.total_num = self.img_dataset.attrs['total_num']
        else:
            raise Exception('Only r or w are accepted')

    def init_total_num_shared_between_processes(self):
        '''
        for shared total_num between mpi processes
        https://stackoom.com/cn_en/question/2CIra
        '''
        # create a shared array of size 1000 elements of type double
        self.size_for_total_num_array = 1
        itemsize = MPI.INT.Get_size() 
        if MPI_COMM_WORLD_RANK == 0: 
            nbytes = self.size_for_total_num_array * itemsize 
        else: 
            nbytes = 0

        # # on rank 0, create the shared block
        # # on rank 1 get a handle to it (known as a window in MPI speak)
        self.win = MPI.Win.Allocate_shared(nbytes, itemsize, comm=MPI.COMM_WORLD) 
        # self.win_for_lock = MPI.Win.Allocate_shared(nbytes, itemsize, comm=MPI.COMM_WORLD) 

        # # create a numpy array whose data points to the shared mem
        self.buf, itemsize = self.win.Shared_query(0) 
        assert itemsize == MPI.INT.Get_size() 
        
        # self.mylock = self.win_for_lock.Shared_query(0)[0]
        # self.mylock[0] = 0

        # self.lock_win = MPI.Win.Create(np.zeros(1, dtype=int), comm=MPI.COMM_WORLD) 
        # self.lock = self.lock_win[0]

        MPI.COMM_WORLD.barrier()

    def __len__(self):
        return self.total_num

    def __getitem__(self, item: int):
        img0 = self.dataset[item]
        cords0 = self.cords[item]
        return img0, cords0 # np.string_(label0)

    def __del__(self):
        self.close()
    
    def close(self):
        print('close rank {}'.format(MPI_COMM_WORLD_RANK))
        self.img_dataset.attrs['total_num'] = self.total_num_array[0] \
            if self.total_num_array[0] > self.img_dataset.attrs['total_num'] \
            else self.img_dataset.attrs['total_num']
            
        if hasattr(self, 'img_dataset'):
            try:
                self.img_dataset.close()
            except Exception as e:
                print('self.img_dataset.close() failed due to {}'.format(e))

    def total_num_array_add_one(self, add=1):
        '''
        return the number before adding one
        '''
        # self.lock.acquire()
        # MPI.Win.Lock(self.win, rank=MPI_COMM_WORLD_RANK, lock_type=MPI.LOCK_EXCLUSIVE)
        self.win.Lock(lock_type=MPI.LOCK_EXCLUSIVE, rank=0)
        
        # while self.mylock[0]:
        #     pass
        # self.mylock[0] = 1
        total_num_temp = self.total_num_array[0]
        self.total_num_array[0] += add
        # self.mylock[0] = 0
        # self.lock.release()
        # MPI.Win.Unlock(self.win, rank=MPI_COMM_WORLD_RANK)

        print('self.total_num_array[:1][0], total_num_temp', self.total_num_array[:1][0], total_num_temp)
        assert self.total_num_array[:1][0] - total_num_temp == 1, \
            'error in total_num_array counting, {} - {} should be 1'.format(self.total_num_array[:1][0], total_num_temp)
            
        self.win.Unlock(0)

        return total_num_temp

    def save_array_and_label(self, feature, ith_roi, x, y):
        total_num_temp = self.total_num_array_add_one()
        self.dataset[total_num_temp, ...] = feature
        self.cords[total_num_temp, ...] = [ith_roi, x, y]
        
if __name__ == '__main__':
    pass