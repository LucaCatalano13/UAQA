import numpy as np

from ..utils import FINAL_H, FINAL_W, resize_array
from .AdspDataset import ADSP_Dataset

DEM_BANDS = ['DEM']
DEM_SHIFT_VALUES = [0.0]
DEM_DIV_VALUES = [2000.0]


class Dem(ADSP_Dataset):
    def __init__(self, dataset_folder: str, legend_folder: str):
      super().__init__(dataset_folder , legend_folder)

    def transform(self, raster_data: np.array, final_w: int = FINAL_W, final_h: int = FINAL_H) -> np.array:
      new_raster_data = []
      for i, band in enumerate(raster_data):
        new_raster_band = resize_array(band, (final_h, final_w))
        new_raster_data.append(new_raster_band)
      return np.array(new_raster_data)
