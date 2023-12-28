import numpy as np
import torch
from datasets.CollectionDataset import CollectionDataset
from datasets.LandCover import LandCover
from datasets.Stations import Stations
from datasets.dataset_utils import FINAL_H , FINAL_W
from PixelTimeseries import PixelTimeSeries

class PixelTimeSeriesLabeled(PixelTimeSeries):
    def __init__(self, stations : Stations, num_timesteps: int, jump = None, collection_dataset: CollectionDataset = None, bound: LandCover = None, input_data_path: str = None):
        super().__init__(num_timesteps, jump , collection_dataset , bound, input_data_path)
        all_dates = np.unique(self.data[3])
        self.stations = stations
        #TODO: check once we have data
        self.stations.add_dates_from_files(all_dates)
    
    def __getitem__(self, index):
        t_index = index % self.num_slices_per_pixel
        start_t = t_index * self.jump
        #- 1 but excluded by [ start_t : end_t ]
        end_t = start_t + self.num_timesteps
        
        flat_ix = index // self.num_slices_per_pixel
        row_ix = flat_ix // FINAL_W
        col_ix = flat_ix % FINAL_W
            
        arrays, hard_mask, latlons, day_of_year, day_of_week = super().__getitem__(index)
        latlon = self.data[2][0, row_ix, col_ix, :]
        
        try : 
            date = self.data[3][end_t + 1, row_ix, col_ix]
            label , loss_factor = self.stations.get_item_temporal_aligned(end_t + 1, row_ix , col_ix, date, latlon)
        except:
            print("Not found")
            # date = self.data[3][end_t-1, row_ix, col_ix]
            # label , loss_factor = self.stations.get_item_temporal_aligned(end_t-1 , row_ix, col_ix, date, latlon)
            
        return arrays, hard_mask, latlons, day_of_year, day_of_week , label , loss_factor
    
    # input modello:
    # (BS, TUTTIGIORNI, 1, CHANNEL) --> BS * tuttigiorni/timesteps  (timesteps , 1 , channel)