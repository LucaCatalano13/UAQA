import numpy as np
from rasterio.plot import show
import cv2
from utils import FINAL_H, FINAL_W
from AdspDataset import ADSP_Dataset

ERA5_BANDS = ['u100',
 'v100',
 'u10',
 'v10',
 'd2m',
 't2m',
 'cbh',
 'e',
 'hcc',
 'msl',
 'mcc',
 'skt',
 'asn',
 'sd',
 'es',
 'sf',
 'ssr',
 'str',
 'sp',
 'tp']

ERA5_SHIFT_VALUES = [-272.15] * len(ERA5_BANDS)

ERA5_DIV_VALUES = [35.0] * len(ERA5_BANDS)

class Era5(ADSP_Dataset):
    #TODO: ci sono valori nan nella band, come li risolvo?
    def __init__(self, dataset_folder: str, legend_folder: str):
      super().__init__(dataset_folder , legend_folder)

    def transform(self, raster_data: np.array, final_w: int = FINAL_W, final_h: int = FINAL_H) -> np.array:
      new_raster_data = []
      for i, band in enumerate(raster_data):
        if band.min() == band.max() or ( np.isnan(band.min() ) and np.isnan(band.max())):
          new_raster_band = cv2.resize(band, (final_w, final_h), interpolation=cv2.INTER_LINEAR)
          new_raster_data.append(new_raster_band)
        #else:
          #TODO: se diversi i valori, come interpolo? --> non è questo il nostro caso
          #continue
      return np.array(new_raster_data)

    def show_raster(self, index):
      raster = self.get_raster(index)
      array = raster.read()
      for i, band in enumerate(array):
        print(f"Chanel {i}")
        print("Name channel: ", list(self.get_legend()[0].keys())[i])
        print("Min: ", band.min())
        print("Max: ", band.max())
        show((raster, i+1))