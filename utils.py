import numpy as np
import torch
from pyproj import Transformer, CRS
from typing import List, Optional
from tqdm import tqdm
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
    normalize: bool = False,
):
    
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
    mask = torch.ones(num_timesteps, len(BANDS))
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
        mask[:, input_to_output_mapping] = 0

    if normalize:
        x = normalize(x)
    return x, mask

def get_city_grids(bounds, radius=0.0045):
    # TODO: not used, we are already in 4326 and we have to go back to 4326
    #converted is used only in gridbox, that we haven't
    l = []
    converted = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(32632), always_xy=True).transform
    xmin, ymax, xmax, ymin = bounds
    x = xmin+radius/2
    ix_x = 0
    while x < xmax:
        y = ymin+radius/2
        ix_y = 0
        while y < ymax:
            l.append(torch.Tensor([x, y]))
            y+=radius
            ix_y+=1
        x+=radius
        ix_x+=1
    return l

def process_images(collection_dataset, bounds):
    '''
    From a single dataset of all datsets analyzed generate the input to Presto architecture
    '''
    arrays, masks, latlons = [], [], []
    for i in tqdm(range(len(collection_dataset))):
        era, lc, s3, s5, dem = collection_dataset[i]

        # TODO: EPSG:4326 --> EPSG:32632 --> Ma poi su Slack dite di ritornare a EPSG:4326
        # convert box
        # function in SLACK to transform
        for row_idx in range(FINAL_H):
            for col_idx in range(FINAL_W):
                # then, get the eo_data, mask and dynamic world
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

                x, mask = construct_single_presto_input(
                    s3=s3_with_time_dimension, s3_bands=S3_BANDS,
                    era5=era_with_time_dimension, era5_bands=ERA5_BANDS,
                    s5=s5_with_time_dimension, s5_bands=S5_BANDS,
                    dem=dem_with_time_dimension, dem_bands=DEM_BANDS,
                    lc=lc_with_time_dimension, lc_bands=LC_BANDS
                )
                arrays.append(x)
                masks.append(mask)
    latlons = get_city_grids(bounds)

    return (torch.stack(arrays, axis=0),
            torch.stack(masks, axis=0),
            # torch.stack(dynamic_worlds, axis=0),
            torch.stack(latlons, axis=0),
            # torch.tensor(labels),
            # image_names,
        )
