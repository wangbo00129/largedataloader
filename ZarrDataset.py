#!/usr/bin/env python3
'''
Zarr data loader for pytorch.

References:
https://www.jianshu.com/p/ee4b76b32779

'''
import zarr
import numpy as np
from mpi4py import MPI
# import mpi_array

MPI_COMM_WORLD_RANK = MPI.COMM_WORLD.Get_rank()  # The process ID (integer 0-3 for 4-process run)
MPI_COMM_WORLD_SIZE = MPI.COMM_WORLD.Get_size()

print('MPI_COMM_WORLD_RANK', MPI_COMM_WORLD_RANK)
print('MPI_COMM_WORLD_SIZE', MPI_COMM_WORLD_SIZE)

from .H5Dataset import H5Dataset

class ZarrDataset(H5Dataset):
    def __init__(self, path_zarr, dataset_name='dataset', cords_name='cords', r_or_w='w', \
        max_sample_num=1*10**5, dim_per_sample=(512,512,3), chunks=(1,256,256,3)):
        '''
        MPI based multiple writer. 
        dataset: to save the tile array. 
        cords: to save the roi, x and y.
        '''
        path_zarr_sync = path_zarr+'.sync'
        self.synchronizer = zarr.ProcessSynchronizer(path_zarr_sync)
        self.img_dataset = zarr.open(path_zarr, r_or_w, synchronizer=self.synchronizer)
        self.dataset_name = dataset_name
        self.cords_name = cords_name

        self.cache_for_batch = [] # a 2-d list

        self.init_total_num_shared_between_processes()

        if r_or_w == 'w':
            self.dataset = self.img_dataset.require_dataset(self.dataset_name, shape=(max_sample_num, *dim_per_sample), dtype=np.uint8, chunks=chunks)
            self.cords = self.img_dataset.require_dataset(self.cords_name, shape=(max_sample_num,3), dtype=np.uint32)
            self.img_dataset.attrs['total_num'] = 0
            self.total_num_array = np.ndarray(buffer=self.buf, dtype='i', shape=(self.size_for_total_num_array,))
            self.total_num_array[:1] = 0
            # self.total_num_array = mpi_array.array([0])
            # self.total_num = self.img_hdf5.attrs['total_num']
            # print('self.total_num is {}'.format(self.img_hdf5.attrs['total_num']))
        elif r_or_w in ['r','a']:
            self.dataset = self.img_dataset[self.dataset_name]
            self.cords = self.img_dataset[self.cords_name]
            self.total_num = self.img_dataset.attrs['total_num']
        else:
            raise Exception('Only r or w are accepted')