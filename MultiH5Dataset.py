# https://vict0rs.ch/2021/06/15/pytorch-h5/
import torch
from torch.utils.data import Dataset
import h5py

class MultiH5Dataset(Dataset):
    def __init__(self, h5_paths, limit=-1):
        self.limit = limit
        self.h5_paths = h5_paths
        self._archives = [h5py.File(h5_path, "r") for h5_path in self.h5_paths]
        self.indices = {}
        idx = 0
        for a, archive in enumerate(self.archives):
            for i in range(archive.attrs['total_num']):
                self.indices[idx] = (a, i)
                idx += 1

        self._archives = None

    @property
    def archives(self):
        if self._archives is None: # lazy loading here!
            self._archives = [h5py.File(h5_path, "r") for h5_path in self.h5_paths]
        return self._archives

    def __getitem__(self, index):
        a, i = self.indices[index]
        archive = self.archives[a]
        dataset = archive['dataset'] # archive[f"trajectory_{i}"]
        data = torch.from_numpy(dataset[i])
        # labels = dict(dataset.attrs)
        # return {"data": data, "labels": labels}
        return data, 1

    def __len__(self):
        if self.limit > 0:
            return min([len(self.indices), self.limit])
        return len(self.indices)

if __name__ == '__main__':
    my_dataset = MultiH5Dataset(h5_paths=['/home/wangb/projects/20221201_patent/Test5/Color_Normalization/TCGA-4T-AA8H-01Z-00-DX1.A46C759C-74A2-4724-B6B5-DECA0D16E029.xml.h5',
        '/home/wangb/projects/20221201_patent/Test5/Color_Normalization/TCGA-5M-AAT4-01Z-00-DX1.725C46CA-9354-43AC-AA81-3E5A66354D6B.xml.h5',
        '/home/wangb/projects/20221201_patent/Test5/Color_Normalization/TCGA-5M-AAT5-01Z-00-DX1.548E7CEB-48FB-4037-A616-39AB025E7A73.xml.h5',
        '/home/wangb/projects/20221201_patent/Test5/Color_Normalization/TCGA-5M-AATE-01Z-00-DX1.483FFD2F-61A1-477E-8F94-157383803FC7.xml.h5',
        '/home/wangb/projects/20221201_patent/Test5/Color_Normalization/TCGA-AA-A01Q-01Z-00-DX1.4432694B-F24B-4942-91FD-27DEF1D84921.xml.h5',
        '/home/wangb/projects/20221201_patent/Test5/Color_Normalization/TCGA-AY-A71X-01Z-00-DX1.68F9BC0F-1D60-4AEF-9083-509387038F03.xml.h5',
        '/home/wangb/projects/20221201_patent/Test5/Color_Normalization/TCGA-DY-A0XA-01Z-00-DX1.6C98F8C6-0D17-4D7A-A404-B8E03E977D2B.xml.h5',
        '/home/wangb/projects/20221201_patent/Test5/Color_Normalization/TCGA-NH-A6GB-01Z-00-DX1.AD90C375-54ED-4EE4-A537-59A2E3FE4BCD.xml.h5',
        '/home/wangb/projects/20221201_patent/Test5/Color_Normalization/TCGA-QG-A5YV-01Z-00-DX1.9B7FD3EA-D1AB-44B3-B728-820939EF56EA.xml.h5',
        '/home/wangb/projects/20221201_patent/Test5/Color_Normalization/TCGA-SS-A7HO-01Z-00-DX1.D20B9109-F984-40DE-A4F1-2DFC61002862.xml.h5',])
    print(my_dataset)
    print(len(my_dataset))

    for i in range(1,1000):
        print(my_dataset[i][0].shape)
        if i % 100 == 0:
            print(i)