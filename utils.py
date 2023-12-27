import numpy as np
import torch
from pyproj import Transformer, CRS
from typing import List, Optional
from tqdm import tqdm
import datetime
import math
from shapely.geometry import Point
from shapely.ops import transform
from pyproj import Transformer, CRS

from datasets.Dem import DEM_BANDS
from datasets.LandCover import LC_BANDS
from datasets.Sentinel3 import S3_BANDS
from datasets.Sentinel5 import S5_BANDS
from datasets.Era5 import ERA5_BANDS
from datasets.CollectionDataset import BANDS, REMOVE_BANDS, ADD_BY, DIVIDE_BY
from datasets.dataset_utils import FINAL_H, FINAL_W

def normalize(cls, x):
  if isinstance(x, np.ndarray):
    x = ((x + ADD_BY) / DIVIDE_BY).astype(np.float32)
  else:
    x = (x + torch.tensor(ADD_BY)) / torch.tensor(DIVIDE_BY)
  return x

def construct_single_presto_input(
    era5: Optional[torch.Tensor] = None,
    era5_bands: Optional[List[str]] = None,
    dem: Optional[torch.Tensor] = None,
    dem_bands: Optional[List[str]] = None,
    lc: Optional[torch.Tensor] = None,
    lc_bands: Optional[List[str]] = None,
    s3: Optional[torch.Tensor] = None,
    s3_bands: Optional[List[str]] = None,
    s5: Optional[torch.Tensor] = None,
    s5_bands: Optional[List[str]] = None,
    normalize: bool = False):

    '''
    Constructin single presto input
    '''
    # The dimension of the list is 1 and its element is alway 1 becouse it is the dimension obtained by the unsqueeze called before
    num_timesteps_list = [x.shape[0] for x in [era5, dem, lc, s3, s5] if x is not None]
    assert len(num_timesteps_list) > 0
    assert all(num_timesteps_list[0] == timestep for timestep in num_timesteps_list)
    # pick the only value = 1
    num_timesteps = num_timesteps_list[0]
    # initialize mask of shape (1, all bands available in our datasets)
    hard_mask = torch.ones(num_timesteps, len(BANDS))
    #  initialize x of shape (1, all bands available in our datasets)
    x = torch.zeros(num_timesteps, len(BANDS))
    # for each input, if it exists, add it to x and set the corresponding mask to 0
    for band_group in [
        (era5, era5_bands, ERA5_BANDS),
        (dem, dem_bands, DEM_BANDS),
        (lc, lc_bands, LC_BANDS),
        (s3, s3_bands, S3_BANDS),
        (s5, s5_bands, S5_BANDS),
    ]:
        data, input_bands, output_bands = band_group
        if data is not None:
            assert input_bands is not None
        else:
            continue
        # remove the bands that we don't want to use
        kept_output_bands = [x for x in output_bands if x not in REMOVE_BANDS]
        # construct a mapping from the input bands to the expected bands
        kept_input_band_idxs = [i for i, val in enumerate(input_bands) if val in kept_output_bands]
        kept_input_band_names = [val for val in input_bands if val in kept_output_bands]

        input_to_output_mapping = [BANDS.index(val) for val in kept_input_band_names]
        # add the data to x and set the corresponding mask to 0
        x[:, input_to_output_mapping] = data[:, kept_input_band_idxs]
        hard_mask[:, input_to_output_mapping] = 0
        
    #TODO: here we generate the real hard mask of the final dataset data (x) is in resize format already
    # set the mask to 1 if the data is nan
    hard_mask[x.isnan()] = 1

    if normalize:
        x = normalize(x)
    return x, hard_mask

def get_city_grids(bounds, radius=500):
# TODO: not used, we are already in 4326 and we have to go back to 4326
    #converted is used only in gridbox, that we haven't
    l = np.ndarray((FINAL_H, FINAL_W, 2))
    x_min, y_max, x_max, y_min = bounds
    # EPSG 32632
    point_min = Point(x_min, y_min)
    point_max = Point(x_max, y_max)
    converted = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(32632), always_xy=True).transform
    point_min_converted = transform(converted, point_min)
    point_max_converted = transform(converted, point_max)
    x_min, x_max, y_min, y_max = point_min_converted.x, point_max_converted.x, point_min_converted.y, point_max_converted.y
    x = x_min+radius/2
    ix_x = 0
    while x < x_max:
        y = y_min+radius/2
        ix_y = 0
        while y < y_max:
            new_point = Point(x, y)
            inv_converted = Transformer.from_crs(CRS.from_epsg(32632), CRS.from_epsg(4326), always_xy=True).transform
            point_converted = transform(inv_converted, new_point)
            x_conv, y_conv = point_converted.x, point_converted.y
            l[ix_y, ix_x, :] = np.array([x_conv, y_conv])
            y+=radius
            ix_y+=1
        x+=radius
        ix_x+=1
    return np.array(l)

def process_images(collection_dataset, bounds, amount_of_data = None):
    '''
    From a single dataset of all datsets analyzed generate the input to Presto architecture
    '''
    if amount_of_data is None:
        amount_of_data = len(collection_dataset)

    arrays = np.ndarray(shape=(amount_of_data, FINAL_H, FINAL_W, len(BANDS)), dtype=np.float32)
    hard_masks = np.ndarray(shape=(amount_of_data, FINAL_H, FINAL_W, len(BANDS)), dtype=np.float32)
    latlons = np.ndarray(shape=(amount_of_data, FINAL_H, FINAL_W, 2), dtype=np.float32)
    dates = np.ndarray(shape=(amount_of_data, FINAL_H, FINAL_W), dtype=object)

    #for each available day
    for i in tqdm(range(amount_of_data)):
        era, lc, s3, s5, dem, date = collection_dataset[i]
        #for each pixel
        for row_idx in range(FINAL_H):
            for col_idx in range(FINAL_W):
                # then, get the per pixel data and mask for all the timestamps
                era_for_pixel = torch.from_numpy(era[:, row_idx, col_idx]).float()
                s3_for_pixel = torch.from_numpy(s3[:-1, row_idx, col_idx]).float()
                s5_for_pixel = torch.from_numpy(s5[::2, row_idx, col_idx]).float()
                dem_for_pixel = torch.from_numpy(dem[:-1, row_idx, col_idx]).float()
                lc_for_pixel = torch.from_numpy(lc[:, row_idx, col_idx]).float()
                
                # add 1 dimension to stack the arrays (we want it divided by channels)
                era_with_time_dimension = era_for_pixel.unsqueeze(0)
                s3_with_time_dimension = s3_for_pixel.unsqueeze(0)
                s5_with_time_dimension = s5_for_pixel.unsqueeze(0)
                dem_with_time_dimension = dem_for_pixel.unsqueeze(0)
                lc_with_time_dimension = lc_for_pixel.unsqueeze(0)

                #stack all BANDS of different sources and generate its hard mask
                x, hard_mask = construct_single_presto_input(
                    s3=s3_with_time_dimension, s3_bands=S3_BANDS,
                    era5=era_with_time_dimension, era5_bands=ERA5_BANDS,
                    s5=s5_with_time_dimension, s5_bands=S5_BANDS,
                    dem=dem_with_time_dimension, dem_bands=DEM_BANDS,
                    lc=lc_with_time_dimension, lc_bands=LC_BANDS
                )
                
                #save all channels
                arrays[i, row_idx, col_idx, :] = x
                #save relative hard masks
                hard_masks[i, row_idx, col_idx, :] = hard_mask
                #save date
                dates[i, row_idx, col_idx] = date
        
        # Generate the coordinates of each pixel 
        latlons[i, :, :] = get_city_grids(bounds)

    # fill the nan value with the mean of the band evaluated in all the dataset    
    nan_indices = np.argwhere(np.isnan(arrays))
    for idx in tqdm(nan_indices):
      arrays[tuple(idx)] = collection_dataset.mean_bands[idx[-1]]
    
    return (torch.Tensor(arrays), torch.Tensor(hard_masks), torch.Tensor(latlons), torch.Tensor(dates))

def get_day_of_year_and_day_of_week(date_list):
    day_of_year = []
    day_of_week = []

    for date_str in date_list:
        date = datetime.datetime.strptime(str(date_str), "%Y-%m-%d")

        # Day of Year (1st Jenuary is 1, so -1 to use them as indices)
        day_of_year.append(date.timetuple().tm_yday - 1)

        # Day of Week (Monday is 0 and Sunday is 6)
        day_of_week.append(date.weekday())

    # Convert lists to tensors
    day_of_year_tensor = torch.tensor(day_of_year)
    day_of_week_tensor = torch.tensor(day_of_week)

    return day_of_year_tensor, day_of_week_tensor