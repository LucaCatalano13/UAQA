import numpy as np
from functools import reduce

from .Dem import DEM_BANDS, DEM_SHIFT_VALUES, DEM_DIV_VALUES
from .LandCover import LC_BANDS, LC_SHIFT_VALUES, LC_DIV_VALUES
from .Sentinel3 import S3_BANDS, S3_SHIFT_VALUES, S3_DIV_VALUES
from .Sentinel5 import S5_BANDS, S5_SHIFT_VALUES, S5_DIV_VALUES
from .Era5 import ERA5_BANDS, ERA5_SHIFT_VALUES, ERA5_DIV_VALUES

DYNAMIC_BANDS = S3_BANDS + S5_BANDS + ERA5_BANDS
STATIC_BANDS = DEM_BANDS + LC_BANDS
RAW_BANDS = DYNAMIC_BANDS + STATIC_BANDS
REMOVE_BANDS = []
BANDS = [ band for band in RAW_BANDS if band not in REMOVE_BANDS]


DYNAMIC_BANDS_SHIFT = S3_SHIFT_VALUES + S5_SHIFT_VALUES + ERA5_SHIFT_VALUES
DYNAMIC_BANDS_DIV = S3_DIV_VALUES + S5_DIV_VALUES + ERA5_DIV_VALUES

STATIC_BANDS_SHIFT = LC_SHIFT_VALUES + DEM_SHIFT_VALUES
STATIC_BANDS_DIV = LC_DIV_VALUES + DEM_DIV_VALUES

ADD_BY = (
    [DYNAMIC_BANDS_SHIFT[i] for i, x in enumerate(DYNAMIC_BANDS) if x not in REMOVE_BANDS]
    + STATIC_BANDS_SHIFT
    + [0.0]
)
DIVIDE_BY = (
    [DYNAMIC_BANDS_DIV[i] for i, x in enumerate(DYNAMIC_BANDS) if x not in REMOVE_BANDS]
    + STATIC_BANDS_DIV
    + [1.0]
)


NORMED_BANDS = [x for x in BANDS]
NUM_BANDS = len(NORMED_BANDS)

class CollectionDataset():
    def __init__(self, era = None, land_cover = None, sentinel3 = None, sentinel5 = None, dem = None):
      # read paths of batch files and metadata
      self.era = era
      self.dem = dem
      self.sentinel3 = sentinel3
      self.sentinel5 = sentinel5
      self.land_cover = land_cover
      self.len_retained_dates = None
      self.__temporal_alignment()

    def __len__(self):
      if self.len_retained_dates is not None:
        return self.len_retained_dates
      return None

    def __temporal_alignment(self):
      # retrieve dates from path
      s3_dates = np.unique([f.split('/')[4].split('T')[0] for f in self.sentinel3.files])
      s5_dates = np.unique([f.split('/')[4].split('T')[0] for f in self.sentinel5.files])
      era5_dates = np.unique([f.split('/')[4].split('.')[0] for f in self.era.files])
      # find dates that appear in all datasets considered
      dates_all_datasets = reduce(np.intersect1d, (s3_dates, s5_dates, era5_dates))
      # find dates that appear at least in one dataset considered (it is a superset of dates_all_datasets)
      dates_least_one_datasets = np.unique(reduce(np.union1d, (s3_dates, s5_dates, era5_dates)))
      # we remove dates that are not in all datasets
      tot_dates_to_remove = np.unique(np.setdiff1d(dates_least_one_datasets, dates_all_datasets))
      self.len_retained_dates = len(dates_all_datasets)
      # remove operations in dynamic datasets
      self.era.remove_dates_from_files(tot_dates_to_remove, self.len_retained_dates)
      self.sentinel3.remove_dates_from_files(tot_dates_to_remove, self.len_retained_dates)
      self.sentinel5.remove_dates_from_files(tot_dates_to_remove, self.len_retained_dates)

    def __getitem__(self, index):
      # dynamic
      era = self.era.get_item_temporal_aligned(index)
      s3 = self.sentinel3.get_item_temporal_aligned(index)
      s5 = self.sentinel5.get_item_temporal_aligned(index)
      # static
      lc = self.land_cover[0]
      dem = self.dem[0]
      return era, lc, s3, s5, dem