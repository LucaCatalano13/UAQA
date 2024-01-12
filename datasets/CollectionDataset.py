import numpy as np
from functools import reduce
from collections import OrderedDict
from typing import OrderedDict as OrderedDictType
from typing import List

from .Dem import DEM_BANDS
from .LandCover import LC_BANDS
from .Sentinel3 import S3_BANDS
from .Sentinel5 import S5_BANDS
from .Era5 import ERA5_BANDS

DYNAMIC_BANDS = S3_BANDS + S5_BANDS + ERA5_BANDS
STATIC_BANDS = DEM_BANDS + LC_BANDS
RAW_BANDS = DYNAMIC_BANDS + STATIC_BANDS
REMOVE_BANDS = []
BANDS = [ band for band in RAW_BANDS if band not in REMOVE_BANDS]

NORMED_BANDS = [x for x in BANDS]
NUM_BANDS = len(NORMED_BANDS)

BANDS_GROUPS_IDX: OrderedDictType[str, List[int]] = OrderedDict(
    {
        "S3": [NORMED_BANDS.index(b) for b in S3_BANDS if b not in REMOVE_BANDS],
        "S5": [NORMED_BANDS.index(b) for b in S5_BANDS if b not in REMOVE_BANDS],
        "ERA5": [NORMED_BANDS.index(b) for b in ERA5_BANDS if b not in REMOVE_BANDS],
        "DEM": [NORMED_BANDS.index(b) for b in DEM_BANDS if b not in REMOVE_BANDS],
        "LC": [NORMED_BANDS.index(b) for b in LC_BANDS if b not in REMOVE_BANDS]
    }
)

BAND_EXPANSION = [len(x) for x in BANDS_GROUPS_IDX.values()]

class CollectionDataset():
  def __init__(self, era = None, land_cover = None, sentinel3 = None, sentinel5 = None, dem = None):
    # read paths of batch files and metadata
    self.era = era
    self.dem = dem
    self.sentinel3 = sentinel3
    self.sentinel5 = sentinel5
    self.land_cover = land_cover

    self.mean_bands = self.__generate_mean()

    self.len_retained_dates = None
    self.aligned_dates = None
    self.__temporal_alignment()
    
  def __generate_mean(self):
    datasets = [ self.era , self.dem, self.sentinel3, self.sentinel5, self.land_cover]
    mean_bands = np.ndarray( len(BANDS) )
    
    for dataset in datasets:
      means = dataset.get_mean_per_bands()
      for i , band in enumerate( dataset.get_bands() ):
        band_idx = BANDS.index(band)
        mean_bands[ band_idx ] = means[i]
        
    return mean_bands

  def __temporal_alignment(self):
    # retrieve dates from path
    s3_dates = np.unique([f.split('/')[4].split('T')[0] for f in self.sentinel3.files])
    s5_dates = np.unique([f.split('/')[4].split('T')[0] for f in self.sentinel5.files])
    era5_dates = np.unique([f.split('/')[4].split('.')[0] for f in self.era.files])
    # find dates that appear in all datasets considered
    # dates_all_datasets = reduce(np.intersect1d, (s3_dates, s5_dates, era5_dates))
    # find dates that appear at least in one dataset considered (it is a superset of dates_all_datasets)
    dates_least_one_datasets = np.unique(reduce(np.union1d, (s3_dates, s5_dates, era5_dates)))
    # we remove dates that are not in all datasets
    # tot_dates_to_remove = np.unique(np.setdiff1d(dates_least_one_datasets, dates_all_datasets))
    # self.len_retained_dates = len(dates_all_datasets)
    # remove operations in dynamic datasets
    # self.era.remove_dates_from_files(tot_dates_to_remove, self.len_retained_dates)
    # self.sentinel3.remove_dates_from_files(tot_dates_to_remove, self.len_retained_dates)
    # self.sentinel5.remove_dates_from_files(tot_dates_to_remove, self.len_retained_dates)
    
    self.len_retained_dates = len(dates_least_one_datasets)
    self.aligned_dates = dates_least_one_datasets

    self.era.add_dates_from_files(dates_least_one_datasets)
    self.sentinel3.add_dates_from_files(dates_least_one_datasets)
    self.sentinel5.add_dates_from_files(dates_least_one_datasets)
    
  def __len__(self):
    if self.len_retained_dates is not None:
      return self.len_retained_dates
    return None

  def __getitem__(self, index):
    # dynamic
    era = self.era.get_item_temporal_aligned(index)
    s3 = self.sentinel3.get_item_temporal_aligned(index)
    s5 = self.sentinel5.get_item_temporal_aligned(index)
    # static
    lc = self.land_cover[0]
    dem = self.dem[0]
    date = self.aligned_dates[index]
    
    #if index not in self.sentinel3.index_temporal_aligned:
    #  date = self.sentinel3.files_temporal_aligned[index].split('/')[4].split('T')[0]
    #else:
    #  date = self.sentinel3.files_temporal_aligned[index]
    
    return era, lc, s3, s5, dem, date