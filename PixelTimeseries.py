import numpy as np
import torch
from functools import reduce
from collections import OrderedDict
from typing import OrderedDict as OrderedDictType
from typing import List
from torch.utils.data import Dataset

from datasets.dataset_utils import FINAL_H, FINAL_W
from utils import process_images, get_day_of_year_and_day_of_week
from datasets.CollectionDataset import CollectionDataset
from datasets.LandCover import LandCover

class PixelTimeSeries(Dataset):
    def __init__(self, num_timesteps: int, collection_dataset: CollectionDataset = None, bound: LandCover = None, input_data_path: str = None):
        super().__init__()
        self.num_timesteps = num_timesteps
        
        if input_data_path is None:
            self.__create_data(collection_dataset, bound)
        else:
            self.__load_data(input_data_path)
        till = 1000
        self.data = (self.data[0][:till], self.data[1][:till], self.data[2][:till], self.data[3][:till], self.data[4][:till])

    def __create_data(self, collection_dataset: CollectionDataset, bound: LandCover): 
        self.data =  process_images(collection_dataset, bound)
    
    def save_data(self, input_data_path_save): 
        torch.save(self.data, input_data_path_save)
    
    def __load_data(self, data_path):
        self.data = torch.load(data_path)

    def len_pixel_timeseries(self):
        # returns number of the days of the pixel timeseries
        return self.data[0].shape[0]

    def __len__(self):
        # returns number of pixels (FINAL_H * FINAL_W) * N_GIORNI/TIMESTEPS
        return ((self.data[0].shape[1] * self.data[0].shape[2]) * (self.len_pixel_timeseries() // self.num_timesteps)) - self.num_timesteps
    
    def __getitem__(self, index):
        t_index = index % (self.len_pixel_timeseries() // self.num_timesteps)
        start_t = t_index * self.num_timesteps
        #- 1 but excluded by [ start_t : end_t ]
        end_t = (t_index +1) * self.num_timesteps
        flat_ix = index // self.len_pixel_timeseries()
        row_ix = flat_ix // FINAL_W
        col_ix = flat_ix % FINAL_H
        arrays = self.data[0][start_t:end_t, row_ix, col_ix, :]
        hard_mask = self.data[1][start_t:end_t, row_ix, col_ix, :]
        latlons = self.data[2][start_t:end_t, row_ix, col_ix, :]
        day_of_year, day_of_week = get_day_of_year_and_day_of_week(self.data[3][start_t:end_t, row_ix, col_ix])
        # returns arrays as a pixel of shape (TUTTIGIORNI, 1, 1, CHANNEL), hard_mask as a pixel of shape (TUTTIGIORNI, 1, 1, CHANNEL), 
        # latlons as a pixel of shape (1, 1, 2), day_of_year as a pixel of shape (1, 1, 1), day_of_week as a pixel of shape (1, 1, 1)
        return arrays, hard_mask, latlons, day_of_year, day_of_week
    
    def retrieve_internal_indices(self, index):
        t_index = index % (self.len_pixel_timeseries() // self.num_timesteps)
        start_t = t_index * self.num_timesteps
        end_t = (t_index + 1) * self.num_timesteps
        flat_ix = index // self.len_pixel_timeseries()
        row_ix = flat_ix // FINAL_W
        col_ix = flat_ix % FINAL_H
        return start_t, end_t - 1, row_ix, col_ix
        
    # input modello:
    # (BS, TUTTIGIORNI, 1, CHANNEL) --> BS * tuttigiorni/timesteps  (timesteps , 1 , channel)