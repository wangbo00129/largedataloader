'''
Reference: nvidia's Accessing_File_with_GDS.ipynb
'''

from torch.utils.data import Dataset
import os
import pathlib
import torch
import re
import cupy as cp
from cucim.clara.filesystem import CuFileDriver
import cucim.clara.filesystem as fs

class RawDataSet(Dataset):
    def __init__(self,
                #  img_path,
                 csv_path,
                 img_transform=None, to_tensor=True):
                 
        with open(csv_path, 'r') as f:
            lines = f.readlines()
            self.img_list = [
                # os.path.join(img_path, re.split(',|\\n', i)[0]) for i in lines
                re.split(',|\\n', i)[0] for i in lines
            ]
            self.label_list = [int(re.split(',|\\n', i)[1]) for i in lines]
            self.classes = sorted(list(set([re.split(',|\\n', i)[1] for i in lines])))
        self.img_transform = img_transform
        self.to_tensor = to_tensor

    def read_raw(self, path, array_shape=[512,512,3], device='cuda'):
        # Create an array with size 10 (in bytes)
        if self.to_tensor:
            cp_arr = torch.zeros(array_shape, dtype=torch.uint8).to(device)
        else:
            cp_arr = cp.zeros(array_shape, dtype=cp.uint8)
    

        fno = os.open(path, os.O_RDONLY | os.O_DIRECT)
        fd = CuFileDriver(fno)

        fd.pread(cp_arr, cp_arr.size, 0)      
        fd.close()
        os.close(fno)
        return cp_arr

    def __getitem__(self, index):
        img_path = self.img_list[index]
        label = self.label_list[index]
        img = self.read_raw(img_path)
        people = pathlib.Path(img_path).parent.stem
        name = pathlib.Path(img_path).stem
        if self.img_transform is not None:
            img = self.img_transform(img)
        return img, label, people, name

    def __len__(self):
        return len(self.label_list)

if __name__ == '__main__':
    print(RawDataSet.read_raw('/home/wangb/pipelines/changablepipelineforsvspreprocessor/xxx_roi0_1000_2000.raw'))