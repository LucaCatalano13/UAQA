import numpy as np
import rasterio

from datasets.dataset_utils import FINAL_H, FINAL_W, resize_array
from datasets.ADSP_Dataset import ADSP_Dataset

S5_BANDS = ['CH4',
 'CO',
 'HCHO',
 'NO2',
 'O3',
 'SO2',
 'CLOUD_TOP_PRESSURE',
 'CLOUD_BASE_PRESSURE',
 'AER_AI_340_380',
 'AER_AI_354_388']

S5_SHIFT_VALUES = [float(0.0)] * len(S5_BANDS)

S5_DIV_VALUES = [float(1e4)] * len(S5_BANDS)

class Sentinel5(ADSP_Dataset):
    def __init__(self, dataset_folder: str, legend_folder: str):
      super().__init__(dataset_folder , legend_folder, S5_BANDS)

    def __getitem__(self, index: int) -> np.array:
      file = self.files[index]
      raster = rasterio.open(file)
      raster_data = self.transform(raster.read())
      return raster_data

    def transform(self, raster_data: np.array, final_w: int = FINAL_W, final_h: int = FINAL_H) -> np.array:
      new_raster_data = []
      for i, band in enumerate(raster_data):
        new_raster_band = resize_array(band, (final_h, final_w))
        new_raster_data.append(new_raster_band)
      return np.array(new_raster_data)
    
    def add_dates_from_files(self, tot_dates_all):
      self.files_temporal_aligned = []
      self.index_temporal_aligned = []
      f_idx = 0
      for i, d in enumerate(tot_dates_all):
          date = self.files[f_idx].split('/')[4].split('T')[0]
          if f_idx < len(self.files)-1:
            date_plus_one = self.files[f_idx + 1].split('/')[4].split('T')[0]
          if f_idx < len(self.files)-2:
            date_plus_two = self.files[f_idx + 2].split('/')[4].split('T')[0]
          if date == date_plus_one:
            if date_plus_one == date_plus_two:
              if d < date:
                self.index_temporal_aligned.append(i)
                self.files_temporal_aligned.append(d)
              if d == date:
                  self.files_temporal_aligned.append(self.files[f_idx])
                  f_idx += 3
            else:
              if d < date:
                  self.index_temporal_aligned.append(i)
                  self.files_temporal_aligned.append(d)
              if d == date:
                  self.files_temporal_aligned.append(self.files[f_idx])
                  f_idx += 2
          else:
            if d < date:
                self.index_temporal_aligned.append(i)
                self.files_temporal_aligned.append(d)
            if d == date:
                self.files_temporal_aligned.append(self.files[f_idx])
                f_idx += 1
                
    def get_mean_per_bands(self):
      all_mean_per_bands = self.__get_all_mean_per_bands()
      return all_mean_per_bands[::2]