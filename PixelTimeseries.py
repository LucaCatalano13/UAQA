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
    def __init__(self, collection_dataset: CollectionDataset = None, bound: LandCover = None, input_data_path: str = None):
        super().__init__()
        if input_data_path is None:
            self.__create_data(collection_dataset, bound)
        else:
            self.__load_data(input_data_path)

    def __init__(self, data_path):
        super().__init__()
        self.data = self.__load_data(data_path)

    def __create_data(self, collection_dataset: CollectionDataset, bound: LandCover): 
        self.data =  torch.Tensor(process_images(collection_dataset, bound))
    
    def save_data(self, input_data_path_save): 
        torch.save(self.data, input_data_path_save)
    
    def __load_data(self, data_path):
        self.data = torch.load(data_path)

    def __len__(self):
        # returns number of pixels (FINAL_H * FINAL_W)
        return self.data.shape[1] * self.data.shape[2]
    
    def __getitem__(self, index):
        arrays = self.data[0][:, index // FINAL_H, index % FINAL_W, :]
        hard_mask = self.data[1][:, index // FINAL_H, index % FINAL_W, :]
        latlons = self.data[2][:, index // FINAL_H, index % FINAL_W, :]
        day_of_year, day_of_week = get_day_of_year_and_day_of_week(self.data[3][index // FINAL_H, index % FINAL_W])
        # returns arrays as a pixel of shape (TUTTIGIORNI, 1, 1, CHANNEL), hard_mask as a pixel of shape (TUTTIGIORNI, 1, 1, CHANNEL), 
        # latlons as a pixel of shape (1, 1, 2), day_of_year as a pixel of shape (1, 1, 1), day_of_week as a pixel of shape (1, 1, 1)
        return arrays, hard_mask, latlons, day_of_year, day_of_week
    
    # input modello:
    # (BS, TUTTIGIORNI, 1, CHANNEL) --> BS * tuttigiorni/timesteps  (timesteps , 1 , channel)