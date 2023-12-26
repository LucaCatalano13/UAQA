import numpy as np
import json
import rasterio
from rasterio.plot import show

from datasets.dataset_utils import FINAL_H, FINAL_W, resize_array, convert_matrix
from datasets.ADSP_Dataset import ADSP_Dataset

LC_BANDS = ['Continuous Urban Fabric (S.L. &amp;gt; 80%)',
 'Discontinuous Dense Urban Fabric (S.L. : 50% - 80%)',
 'Discontinuous Medium Density Urban Fabric (S.L. : 30% - 50%)',
 'Discontinuous Low Density Urban Fabric (S.L. : 10% - 30%)',
 'Discontinuous Very Low Density Urban Fabric (S.L. &amp;lt; 10%)',
 'Isolated Structures',
 'Industrial, commercial, public, military and private units',
 'Fast transit roads and associated land',
 'Other roads and associated land',
 'Railways and associated land',
 'Port areas',
 'Airports',
 'Mineral extraction and dump sites',
 'Construction sites',
 'Land without current use',
 'Green urban areas',
 'Sports and leisure facilities',
 'Arable land (annual crops)',
 'Permanent crops (vineyards, fruit trees, olive groves)',
 'Pastures',
 'Complex and mixed cultivation patterns',
 'Orchards at the fringe of urban classes',
 'Forests',
 'Herbaceous vegetation associations (natural grassland, moors...)',
 'Open spaces with little or no vegetations (beaches, dunes, bare rocks, glaciers)',
 'Wetland',
 'Water bodies']

LC_MAP_DICT_CLASSES = {1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:2, 9:2, 10:3, 11:4, 12:5, 13:6,
                                   14:1, 15:7, 16:8, 17:1, 18:8, 19:8, 20:8, 21:8, 22:8, 23:8, 24:8, 25:9, 26:10, 27:10}

LC_BANDS = ['Urban', 'Road', 'Railways', 'Port', 'Airports', 'Extraction', 'NoUse', 'Green', 'OpenSpaces', 'Water']

LC_SHIFT_VALUES = [0.0]

LC_DIV_VALUES = [2000.0]


class LandCover(ADSP_Dataset):
    def __init__(self, dataset_folder: str, legend_folder: str, old_new_classes_dict = LC_MAP_DICT_CLASSES):
      super().__init__(dataset_folder, legend_folder, LC_BANDS)
      self.original_taxonomy = json.load(open(self.labels_legends[0]))
      self.old_new_classes_dict = old_new_classes_dict
      # save the resized shape of the raster of the dataset
      self.shape_resized_raster = self.__get_len_with_mask_raster()
      
    def __set_new_classes(self, matrix: np.ndarray, verbose: bool=False) -> np.ndarray:
      if verbose:
        print(f'Old classes matrix: {matrix}')

      for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
          matrix[i,j] = self.old_new_classes_dict[matrix[i,j]]

      if verbose:
        print(f'New classes matrix: {matrix}')
      return matrix

    def transform(self, raster_data: np.array, final_w: int = FINAL_W, final_h: int = FINAL_H) -> np.array:
      new_raster_data = []
      #raster_data is (1, h, w), i want(h,w) as input
      raster_data = convert_matrix(self.__set_new_classes(raster_data[0]))
      #resize each channel to the final shape and stack them in a 3D array
      for i, band in enumerate(raster_data):
        new_raster_band = resize_array(band, (final_h, final_w))
        new_raster_data.append(new_raster_band)
      return np.array(new_raster_data)

    def show_raster(self, index):
      raster = self.get_raster(index)
      array = raster.read()
      for i, band in enumerate(array):
        print(f"Chanel {i}")
        print("Name channel: Land Cover")
        print("Min: ", band.min())
        print("Max: ", band.max())
        show((raster, i+1))
        
    def get_mean_per_bands(self):
      return self.__get_all_mean_per_bands()
    
    def from_file_path_to_date(self, string):
      return "static dataset"
    
    def __get_len_with_mask_raster(self):
      return (len(self.bands), FINAL_H, FINAL_W)