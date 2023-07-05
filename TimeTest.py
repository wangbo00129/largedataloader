import sys
sys.path.append('/home/wangb/pipelines/largedataloader/')
from pipelines.largedataloader.H5Dataset import H5Dataset
import numpy as np
import time

def testLoader(loader=H5Dataset(path_h5='/home/wangb/pipelines/largedataloader/cf_color.h5', r_or_w='r'),\
    test_iteration=1000):
    print(loader)
    len_loader = len(loader)
    print('length', len_loader)
    rand_indices = np.random.randint(0, len_loader, test_iteration)    
    start = time.time()
    for index in rand_indices:
        loader.__getitem__(index)
    end = time.time()
    total_time = end - start
    print(total_time)
    return total_time

if __name__ == '__main__':
    total_time = testLoader()