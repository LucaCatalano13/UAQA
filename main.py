import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.Era5 import Era5
from datasets.Dem import Dem
from datasets.Sentinel3 import Sentinel3
from datasets.Sentinel5 import Sentinel5
from datasets.LandCover import LandCover
from datasets.CollectionDataset import CollectionDataset
import parser

from presto.presto import Encoder, Decoder, Presto
from PixelTimeseries import PixelTimeSeries


#TODO: not to use, we have to change all
if __name__ == "__main__":
    args = parser.parse_arguments()
    
    if args.input_data_path is not None:
        train_dataset = PixelTimeSeries(num_timesteps=5, input_data_path = args.input_data_path)
    else:
        train_era = Era5(dataset_folder = args.era5_path, legend_folder = args.era5_legend_path)
        train_dem = Dem(dataset_folder = args.dem_path, legend_folder = args.dem_legend_path)
        train_sentinel3 = Sentinel3(dataset_folder = args.sentinel3_path, legend_folder = args.sentinel3_legend_path)
        train_sentinel5 = Sentinel5(dataset_folder = args.sentinel5_path, legend_folder = args.sentinel5_legend_path)
        train_land_cover = LandCover(dataset_folder = args.land_cover_path, legend_folder = args.land_cover_legend_path)
        train_collection_dataset = CollectionDataset(era = train_era, dem = train_dem, sentinel3 = train_sentinel3, 
                                                     sentinel5 = train_sentinel5, land_cover = train_land_cover)
        train_bound = train_land_cover.get_bound()
        train_dataset = PixelTimeSeries(num_timesteps=5, collection_dataset=train_collection_dataset, bound=train_bound)
        
    if args.input_test_path is not None:
        test_dataset = PixelTimeSeries(num_timesteps=5, input_data_path = args.input_test_path)
    else:
        test_era = Era5(dataset_folder = args.era5_test_path, legend_folder = args.era5_legend_path)
        test_dem = Dem(dataset_folder = args.dem_test_path, legend_folder = args.dem_legend_path)
        test_sentinel3 = Sentinel3(dataset_folder = args.sentinel3_test_path, legend_folder = args.sentinel3_legend_path)
        test_sentinel5 = Sentinel5(dataset_folder = args.sentinel5_test_path, legend_folder = args.sentinel5_legend_path)
        test_land_cover = LandCover(dataset_folder = args.land_cover_test_path, legend_folder = args.land_cover_legend_path)
        test_collection_dataset = CollectionDataset(era = test_era, dem = test_dem, sentinel3 = test_sentinel3, 
                                                     sentinel5 = test_sentinel5, test_cover = test_land_cover)
        test_bound = test_land_cover.get_bound()
        test_dataset = PixelTimeSeries(num_timesteps=5, collection_dataset=test_collection_dataset, bound=test_bound)
    
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [int(len(train_dataset)*0.8), int(len(train_dataset)*0.2)])

    train_dataloader = DataLoader(
            train_dataset,
            batch_size=64,
            shuffle=False,
        )

    val_dataloader = DataLoader(
            val_dataset,
            batch_size=64,
            shuffle=False,
        )

    encoder = Encoder()
    decoder = Decoder(encoder.channel_embed)
    presto = Presto(encoder, decoder)

    for (x, hard_mask, latlons, day_of_year, day_of_week) in tqdm(train_dataloader):
        presto(x=x, mask = hard_mask, latlons=latlons, 
                        day_of_year = day_of_year, day_of_week = day_of_week)