import numpy as np

from datasets.dataset_utils import FINAL_H, FINAL_W, resize_array
from datasets.ADSP_Dataset import ADSP_Dataset

DEM_BANDS = ['DEM']
DEM_SHIFT_VALUES = [0.0]
DEM_DIV_VALUES = [2000.0]


class Dem(ADSP_Dataset):
    def __init__(self, dataset_folder: str, legend_folder: str):
      super().__init__(dataset_folder , legend_folder, DEM_BANDS)
      # save the resized shape of the raster of the dataset
      self.shape_resized_raster = self.__get_len_with_mask_raster()
    
    def transform(self, raster_data: np.array, final_w: int = FINAL_W, final_h: int = FINAL_H) -> np.array:
      new_raster_data = []
      for i, band in enumerate(raster_data):
        new_raster_band = resize_array(band, (final_h, final_w))
        new_raster_data.append(new_raster_band)
      return np.array(new_raster_data)
    
    def get_mean_per_bands(self):
      all_mean_per_bands = self.__get_all_mean_per_bands()
      return [all_mean_per_bands[0]]

    def from_file_path_to_date(string):
      return "static dataset"

    def __get_len_with_mask_raster(self):
      return (len(self.bands), FINAL_H, FINAL_W)