#!/usr/bin/env python3
'''

'''
import pandas as pd
import numpy as np
import torch
import cucim

class SlideDataLoader(torch.utils.data.Dataset):
    def __init__(self, path_large_image, label_suffix='.label', r_or_w='r', \
        max_sample_num=10**5, dim_per_sample=(512,512), num_patches_per_row=100, num_patches_per_col=100):
        self.path_large_image = path_large_image
        self.label_file = self.path_large_image + label_suffix
        self.dim_per_sample = dim_per_sample
        self.num_patches_per_row = num_patches_per_row
        self.num_patches_per_col = num_patches_per_col
        
        if r_or_w == 'w':
            pass
        elif r_or_w in ['r']:
            self.large_image_reader = cucim.CuImage(self.path_large_image)
            self.df_label = pd.read_csv(self.label_file, sep='\t', index_col=0)
        else:
            raise Exception('Only r or w are accepted')

    def __len__(self):
        return self.df_label.shape[0]

    def __getitem__(self, item: int):
        x_start_for_item = self.df_label.loc[item, 'x_start']
        y_start_for_item = self.df_label.loc[item, 'y_start']
        self.dim_per_sample
        img0 = self.large_image_reader.read_region((x_start_for_item, y_start_for_item), self.dim_per_sample, 0)
        img0 = np.array(img0)
        label0 = self.df_label.loc[item,'label']
        return img0, label0

    def __del__(self):
        self.close()
    
    def close(self):
        pass
    
    def save_array_and_label(self, feature, label):
        # To be implemented
        # row_index_for_item = int(item / self.num_patches_per_col)
        # col_index_for_item = int(item % self.num_patches_per_col)

        total_num = self.df_label.shape[0]
        self.df_label[total_num,'label'] = label

        total_num += 1

    def save_array_and_label_in_batch(self, feature, label, last_save=False):
        '''
        Not save immediately.
        last_save: batch will be saved and the current feature, label will not be recorded anymore.
        '''
        # full
        if len(self.cache_for_batch) >= self.batch_size or last_save:
            current_size = len(self.cache_for_batch)
            self.dataset[len(self):len(self)+current_size, ...] = np.stack([x[0] for x in self.cache_for_batch])
            self.label[len(self):len(self)+current_size] = np.stack([[x[1]] for x in self.cache_for_batch])
            self.img_hdf5.attrs['total_num'] += current_size
            print('current_size: {}; after update: {}'.format(current_size, len(self)))
            self.cache_for_batch = []
            
        if last_save:
            return

        self.cache_for_batch.append([feature, label])
        

if __name__ == '__main__':
    import numpy as np
    loader = SlideDataLoader('/home/wangb/projects/20220801_svs/svs/TCGA-EA-A4BA-01Z-00-DX1/TCGA-EA-A4BA-01Z-00-DX1.7EB090B2-E79E-417F-A871-247353679D7B.svs')
    for i in range(100):
        rdm = np.random.randint(0,100,1)
        # print(loader.__getitem__(rdm))