from imp import reload
import pipelines.largedataloader.ZarrDataset as ZarrDataset
import pipelines.largedataloader.FromDiskToDataset as FromDiskToDataset
reload(ZarrDataset)
reload(FromDiskToDataset)
from pipelines.largedataloader.ZarrDataset import ZarrDataset
from pipelines.largedataloader.FromDiskToDataset import globAndToDataset

path_parent_folder_for_patient_folder = '/data/data/FromHospital/cf_color/'
dataloader = ZarrDataset(path_zarr='cf_color.zarr', r_or_w='w')
globAndToDataset(path_parent_folder_for_patient_folder, dataloader, save_method_name='save_array_and_label_in_batch')

