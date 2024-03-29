import numpy as np

from datasets.dataset_utils import FINAL_H, FINAL_W, resize_array
from datasets.ADSP_Dataset import ADSP_Dataset

S3_BANDS = ['F1', 'F2', 'S7', 'S8', 'S9']

class Sentinel3(ADSP_Dataset):
    def __init__(self, dataset_folder: str, legend_folder: str):
      super().__init__(dataset_folder , legend_folder, S3_BANDS)
      # save the resized shape of the raster of the dataset
      self.shape_resized_raster = self.__get_len_with_mask_raster()
      
    def transform(self, raster_data: np.array, final_w: int = FINAL_W, final_h: int = FINAL_H) -> np.array:
      new_raster_data = []
      for i, band in enumerate(raster_data):
        new_raster_band = resize_array(band, (final_h, final_w))
        new_raster_data.append(new_raster_band)
      return np.array(new_raster_data)

    def remove_dates_from_files(self, dates_to_remove, len_retained_dates):
      dictionary_date_signle_file = {}
      self.files_temporal_aligned = []
      # check if dates aren't in remove_dates and append
      for f in self.files:
        check = True
        for s in dates_to_remove:
          if str(s) in str(f):
            check = False
        if check:
          string_date = str(f).split('/')[4].split('T')[0]
          if string_date not in dictionary_date_signle_file:
            dictionary_date_signle_file[string_date] = str(f)
      self.files_temporal_aligned = sorted(list(dictionary_date_signle_file.values()))
    
    def get_mean_per_bands(self):
      all_mean_per_bands = self.get_all_mean_per_bands()
      return all_mean_per_bands[:-1]
    
    def from_file_path_to_date(self, string):
      return string.split('/')[4].split('T')[0]

    def __get_len_with_mask_raster(self):
      return (len(self.bands) + 1, FINAL_H, FINAL_W)