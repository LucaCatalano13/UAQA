import rasterio
from rasterio.plot import show
import json
from torch.utils.data import Dataset
import numpy as np
from glob import glob
from .dataset_utils import FINAL_H, FINAL_W

class ADSP_Dataset (Dataset):
    """
    Main Class for commmon procedures
    """
    def __init__(self, dataset_folder: str, legend_folder: str, bands: list):
        super().__init__()
        # read paths of files
        self.files = sorted(glob(f"{dataset_folder}/*.tiff", recursive = True))
        self.bands = bands
        # paths of files temporal aligned: some path and some dates (longer or equal to files)
        self.files_temporal_aligned = None
        # list of index on files_temporal_aligned that have no real data
        self.index_temporal_aligned = None
        # read paths of legend linked with file
        self.labels_legends =  sorted(glob(f"{legend_folder}/*.json", recursive = True))
        # save the resoluton per pixel of the raster data
        self.resolution = rasterio.open(self.files[0]).res
        # save the original shape of the raster of the dataset
        self.original_raster_shape = rasterio.open(self.files[0]).shape
    
    def __len__(self) -> int:
        # the len of the dataset equals to the number of the files it contains
        return len(self.files)

    def transform(self, raster_data):
        return raster_data
    
    def get_bands(self):
        return self.bands
    
    # def get_item_temporal_aligned(self, index):
    #     assert self.files_temporal_aligned is not None
    #     # retrieve and open the .tiff file
    #     file = self.files_temporal_aligned[index]
    #     #TODO with statemant --> perform
    #     raster = rasterio.open(file)
    #     # transform the raster into a numpy array
    #     raster_data = raster.read()
    #     return self.transform(raster_data)

    def get_item_temporal_aligned(self, index):
        assert self.files_temporal_aligned is not None
        # retrieve and open the .tiff file
        file = self.files_temporal_aligned[index]
        #if original dataset has no data for that index
        if index in self.index_temporal_aligned:
            return np.full(self.shape_resized_raster, np.nan)
        #otherwise open the real data
        #TODO with statemant --> perform
        raster = rasterio.open(file)
        # transform the raster into a numpy array
        raster_data = raster.read()
        return self.transform(raster_data)

    def __getitem__(self, index: int) -> np.array:
        # retrieve and open the .tiff file
        file = self.files[index]
        #TODO with statemant --> perform
        raster = rasterio.open(file)
        # transform the raster into a numpy array
        raster_data = raster.read()
        return self.transform(raster_data)

    def remove_dates_from_files(self, dates_to_remove, len_retained_dates):
        self.files_temporal_aligned = []
        # check if dates aren't in remove_dates and append
        for f in self.files:
            check = True
            for s in dates_to_remove:
                if str(s) in str(f):
                    check = False
            if check:
                self.files_temporal_aligned.append(f)

    def get_raster(self, index):
        file = self.files[index]
        raster = rasterio.open(file)
        return raster

    def get_legend(self):
        legend = []
        for i in range(len(self.labels_legends)):
            path = self.labels_legends[i]
            legend.append(json.load(open(path)))
        return legend

    def get_bound(self):
        meta = self.get_raster(0).meta
        w = meta["width"]
        h = meta["height"]
        # It retruns (xmin, ymax, xmax, ymin)
        return ( meta["transform"] * (0,0) + meta["transform"] * (w,h) )

    def show_raster(self, index):
        raster = self.get_raster(index)
        array = raster.read()
        for i, band in enumerate(array):
            print(f"Chanel {i}")
            print("Name channel: ", self.get_legend()[0][i])
            print("Min: ", band.min())
            print("Max: ", band.max())
            show((raster, i+1))

    def get_resolutions_in_m(self) -> tuple():
        pixel_width_degrees, pixel_height_degrees = self.resolution
        # Conversion factors
        lat_conversion_factor = 111 # Approximate value for latitude in km per degree
        # Calculate approximate pixel dimensions in meters
        pixel_width_meters = pixel_width_degrees * lat_conversion_factor * 1000
        pixel_height_meters = pixel_height_degrees * lat_conversion_factor * 1000
        return pixel_width_meters, pixel_height_meters

    def __get_all_mean_per_bands(self):
        dataset_bands = self.shape_resized_raster[0]
        sum_values_per_bands = np.zeros(dataset_bands)
        n_values = np.zeros(dataset_bands)
        #for all days available
        for day_index in range(len(self.files)):
            raster = self[day_index]
            #for all bands in dataset
            for band in range(dataset_bands):
                sum_values_per_bands[band] += np.nansum(raster[band])
                n_values[band] += np.sum(~np.isnan(raster[band]))
        return sum_values_per_bands / n_values
    
    def from_file_path_to_date(self, string):
        raise NotImplementedError
    
    def add_dates_from_files(self, tot_dates_all):
      self.files_temporal_aligned = []
      self.index_temporal_aligned = []
      end = False
      f_idx = 0
      for i, d in enumerate(tot_dates_all):
          if f_idx < len(self.files):
            date = self.from_file_path_to_date(self.files[f_idx])
          else:
            end = True
          if d < date or end:
            self.index_temporal_aligned.append(i)
            self.files_temporal_aligned.append(d)
          if d == date:
            self.files_temporal_aligned.append(self.files[f_idx])
            while f_idx < len(self.files) and date == self.from_file_path_to_date(self.files[f_idx]):
              f_idx += 1