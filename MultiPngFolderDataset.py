'''
Modified from
https://vict0rs.ch/2021/06/15/pytorch-h5/
'''
import torch
from torch.utils.data import Dataset
from glob import glob
import numpy as np

def pil_loader(path):
    from PIL import Image
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    '''
    copied from custom_dset.py
    '''
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

def cv2_loader(path):
    import cv2
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

class MultiPngFolderDataset(Dataset):
    def __init__(self, png_folder_paths, limit=-1, suffix='.png', png_read_func=cv2_loader):
        self.limit = limit
        self.png_folder_paths = png_folder_paths
        self.png_read_func = png_read_func

        self._archives = [glob(png_folder_path+'/*png') for png_folder_path in self.png_folder_paths]
        self.indices = {}
        idx = 0
        for a, archive in enumerate(self.archives):
            for i in range(len(archive)):
                self.indices[idx] = (a, i)
                idx += 1

        self._archives = None

    @property
    def archives(self):
        if self._archives is None: # lazy loading here!
            self._archives = [glob(png_folder_path+'/*png') for png_folder_path in self.png_folder_paths]
        return self._archives

    def __getitem__(self, index):
        a, i = self.indices[index]
        archive = self.archives[a]
        dataset = archive # archive[f"trajectory_{i}"]
        data = torch.from_numpy(np.array(self.png_read_func(dataset[i])))
        # labels = dict(dataset.attrs)
        # return {"data": data, "labels": labels}
        return data, 1

    def __len__(self):
        if self.limit > 0:
            return min([len(self.indices), self.limit])
        return len(self.indices)

if __name__ == '__main__':
    pass