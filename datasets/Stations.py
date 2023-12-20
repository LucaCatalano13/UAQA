import numpy as np
import rasterio
import cv2
from rasterio.plot import show
from datasets.dataset_utils import FINAL_H, FINAL_W, resize_array
from datasets.ADSP_Dataset import ADSP_Dataset

STATIONS_BANDS = ["SO2","C6H6","NO2","O3","PM10","PM25","CO"]
LOSS_DEFAULT_FACTOR = 0.3

class Stations(ADSP_Dataset):
    def __init__(self, dataset_folder: str, legend_folder: str, gold_data_path:str, gold_legend_path:str, loss_default_factor = LOSS_DEFAULT_FACTOR):
        super().__init__(dataset_folder , legend_folder, STATIONS_BANDS)
        self.loss_default_factor = loss_default_factor
        self.gold_data_path = gold_data_path
        self.gold_legend_path = gold_legend_path
        self.gold_stations = GoldStation(self.gold_data_path, self.gold_legend_path)
        #files_temporal_aligned [TS_all]

    def transform(self, raster_data: np.array, final_w: int = FINAL_W, final_h: int = FINAL_H) -> np.array:
        new_raster_data = []
        for i, band in enumerate(raster_data):
            new_raster_band = cv2.resize(band, (final_w, final_h), interpolation=cv2.INTER_CUBIC)
            new_raster_data.append(new_raster_band)
        return np.array(new_raster_data)
    
    def get_loss_factor(self, date, latlon):
        closest_dist_per_band = self.gold_stations.get_closest_dist_per_band(date, latlon)
        loss_factors = np.ndarray(len(STATIONS_BANDS))
        for i in range(len(closest_dist_per_band)):
            if np.isclose( closest_dist_per_band[i] , 0 ):
                loss_factors[i] = 1
            else:
                loss_factors[i] = self.loss_default_factor
        
        return loss_factors
    
    def get_item_temporal_aligned(self, time_index , row_ix, col_ix, date, latlon):
        assert self.files_temporal_aligned is not None
        # retrieve and open the .tiff file
        file = self.files_temporal_aligned[time_index]
        #if original dataset has no data for that index
        if time_index in self.index_temporal_aligned:
            #TODO: fill with which value?
            return np.full(len(STATIONS_BANDS), np.nan)
        #otherwise open the real data
        raster = rasterio.open(file)
        # transform the raster into a numpy array
        raster_data = raster.read()
        
        #(len(STATIONS_BANDS) , FINAL_H, FINAL_W)
        raster_array = self.transform(raster_data)
        #len(STATIONS_BANDS)
        label = raster_array[:, row_ix , col_ix].flatten()
        loss_factor = self.get_loss_factor(date, latlon)
        return label , loss_factor
        
    def show_raster(self, index):
        raster = self.get_raster(index)
        array = raster.read()
        for i, band in enumerate(array):
            print(f"Chanel {i}")
            print("Min: ", band.min())
            print("Max: ", band.max())
            show((raster, i+1))
    
    def get_mean_per_bands(self):
        all_mean_per_bands = self.__get_all_mean_per_bands()
        return [all_mean_per_bands[0]]
    
    
class GoldStation():
    def __init__(self , data_path: str , legend_path:str):
        self.data_path = data_path
        self.legend_path = legend_path
        self.data = self.__create_data(self.data_path , self.legend_path)
        
    def __create_data(self, data_path: str , legend_path:str):
        #open csv create dict of dictionary --> k [day][latlon] v [mesurements]
        pass 
    
    def get_closest_dist_per_band(self, date, latlon):
        """
        Given latlon of a pixel find for each pollutant the distance of the closest GoldenStation
        NPArray aligned with STATION_BANDS
      
        """
        #TODO: write it Luca, I belive in you
        pass
    

