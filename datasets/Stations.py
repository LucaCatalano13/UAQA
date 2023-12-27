import numpy as np
import pandas as pd
import math
import rasterio
import cv2
from rasterio.plot import show
from datasets.dataset_utils import FINAL_H, FINAL_W, resize_array
from datasets.ADSP_Dataset import ADSP_Dataset

STATIONS_BANDS = ["SO2","C6H6","NO2","O3","PM10","PM25","CO"]
LOSS_DEFAULT_FACTOR = 0.3
#Earth Radius
R = 6373.0

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
        if date not in self.gold_stations.data.keys():
            return np.full(len(STATIONS_BANDS), self.loss_default_factor)
        closest_dist_per_band, closest_label = self.gold_stations.get_closest_dist_per_band(date, latlon)
        loss_factors = np.ndarray(len(STATIONS_BANDS))
        for i in range(len(closest_dist_per_band)):
            if np.isclose(closest_dist_per_band[i], 0, atol=0.38769159659895186):
                loss_factors[i] = 1
            else:
                loss_factors[i] = self.loss_default_factor
        
        return loss_factors, closest_label
    
    def get_item_temporal_aligned(self, time_index, row_ix, col_ix, date, latlon):
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
        loss_factor, closest_label = self.get_loss_factor(date, latlon)

        for i in range(len(STATIONS_BANDS)):
            if loss_factor[i] == 1:
                label[i] = closest_label[i]

        return label , loss_factor
        
    def show_raster(self, index):
        raster = self.get_raster(index)
        array = raster.read()
        for i, band in enumerate(array):
            print(f"Chanel {i}")
            print("Min: ", band.min())
            print("Max: ", band.max())
            show((raster, i+1))
    
    def from_file_path_to_date(self, string):
        # è 4 su COLAB!!!!!!!
        # print(string.split('/')[5])
        return string.split('/')[6].split('.')[0]

    def get_mean_per_bands(self):
        all_mean_per_bands = self.__get_all_mean_per_bands()
        return [all_mean_per_bands[0]]
    
    
class GoldStation():
    def __init__(self, data_path: str, legend_path:str):
        self.data_path = data_path
        self.legend_path = legend_path
        
        self.data = self.__create_data(self.data_path, self.legend_path)
        
    def __create_data(self, data_path: str, legend_path:str):
        data = {}
        data_path_df = pd.read_csv(data_path)
        legend_path_df = pd.read_csv(legend_path, sep=";")
        legend_path_df['Location'] = legend_path_df['Location'].str.split(', ')
        legend_dict = dict(zip(legend_path_df['id_amat'], legend_path_df['Location']))
        
        for _, row in data_path_df.iterrows():
            date = row['date']
            if date not in data:
                data[date] = {}
            latlon = legend_dict.get(row['station_id'])
            if latlon:
                latlon = latlon[1][:-1] + ' ' + latlon[0][1:]
                data[date][latlon] = dict(row[4:])
        return data
    
    def get_closest_dist_per_band(self, date, latlon):
        """
        Given latlon of a pixel find for each pollutant the distance of the closest GoldenStation
        NPArray aligned with STATION_BANDS
      
        """
        data_single_date = self.data[date]
        distances = {}
        label = {}
        for band in STATIONS_BANDS:
            distances[band] = np.inf
        for i, band in enumerate(STATIONS_BANDS):
            for j, latlon_data in enumerate(list(data_single_date.keys())):
                latlon_data_list = latlon_data.split()
                if not np.isnan(data_single_date[latlon_data][band]):
                    lat1 = math.radians(latlon[1])
                    lat2 = math.radians(float(latlon_data_list[1]))
                    lon1 = math.radians(latlon[0])
                    lon2 = math.radians(float(latlon_data_list[0]))
                    diff_lon = lon2 - lon1
                    diff_lat = lat2 -lat1 
                    a = (math.sin(diff_lat/2))**2 + math.cos(lon1) * math.cos(float(lat2)) * (math.sin(diff_lon/2))**2
                    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
                    dist = R * c
                    if band in distances.keys():
                        if dist < distances[band]:
                            distances[band] = dist
                            label[band] = data_single_date[latlon_data][band]
                    else:
                        distances[band] = dist
                        label[band] = data_single_date[latlon_data][band]
        return list(distances.values()), list(label.values())


