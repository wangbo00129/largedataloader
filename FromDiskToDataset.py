import os
from glob import glob
import cv2
import re
from pipelines.largedataloader.H5Dataset import H5Dataset

def globAndToDataset(path_parent_folder_for_patient_folder, dataloader, image_suffix='.png', \
    strip_parent_folder=True, save_method_name='save_array_and_label_in_batch'):
    paths = glob(os.path.join(path_parent_folder_for_patient_folder,'*','*{}'.format(image_suffix)))
    print('globed', len(paths))
    for p in paths:
        img = cv2.imread(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = p
        if strip_parent_folder:
            label = re.sub('^{}'.format(path_parent_folder_for_patient_folder), '', label)
        getattr(dataloader, save_method_name)(img, label)

if __name__ == '__main__':
    pass
    # write
    # path_parent_folder_for_patient_folder = '/data/data/FromHospital/cf_color/'
    # dataloader = H5DataLoader(path_h5='cf_color.h5', r_or_w='w')
    # globAndToDataLoader(path_parent_folder_for_patient_folder, dataloader)
    
    # read
    H5Dataset(path_h5='cf_color.h5', r_or_w='w')