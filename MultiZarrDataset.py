'''
Modified from
https://vict0rs.ch/2021/06/15/pytorch-h5/
'''
import torch
from torch.utils.data import Dataset
import zarr

class MultiZarrDataset(Dataset):
    def __init__(self, zarr_paths, limit=-1):
        self.limit = limit
        self.zarr_paths = zarr_paths
        self._archives = [zarr.open(zarr_path, "r") for zarr_path in self.zarr_paths]
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
            self._archives = [zarr.open(zarr_path, "r") for zarr_path in self.zarr_paths]
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
    my_dataset = MultiZarrDataset(zarr_paths=[])
    print(my_dataset)
    print(len(my_dataset))

    for i in range(1,1000):
        print(my_dataset[i][0].shape)
        if i % 100 == 0:
            print(i)