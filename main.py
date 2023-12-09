import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from datasets.Era5 import Era5
from datasets.Dem import Dem
from datasets.Sentinel3 import Sentinel3
from datasets.Sentinel5 import Sentinel5
from datasets.LandCover import LandCover
from datasets.CollectionDataset import CollectionDataset
from utils import process_images, get_day_of_year_and_day_of_week
import parser

from presto.presto import Encoder, Decoder, Presto
from PixelTimeseries import PixelTimeSeries

if __name__ == "__main__":
    args = parser.parse_arguments()
    if args.load_data_input:
        dataset = PixelTimeSeries(num_timesteps=5, input_data_path = args.input_data_path)
    else:
        era = Era5(dataset_folder = args.era5_path, legend_folder = args.era5_legend_path)
        dem = Dem(dataset_folder = args.dem_path, legend_folder = args.dem_legend_path)
        sentinel3 = Sentinel3(dataset_folder = args.sentinel3_path, legend_folder = args.sentinel3_legend_path)
        sentinel5 = Sentinel5(dataset_folder = args.sentinel5_path, legend_folder = args.sentinel5_legend_path)
        land_cover = LandCover(dataset_folder = args.land_cover_path, legend_folder = args.land_cover_legend_path)
        collection_dataset = CollectionDataset(era = era, dem = dem, sentinel3 = sentinel3, sentinel5 = sentinel5, land_cover = land_cover)
        bound = land_cover.get_bound()
        dataset = PixelTimeSeries(num_timesteps=5, collection_dataset=collection_dataset, bound=bound)

    train_dataloader = DataLoader(
            dataset,
            batch_size=64,
            shuffle=False,
        )

    encoder = Encoder()
    decoder = Decoder(encoder.channel_embed)
    presto = Presto(encoder, decoder)

    for (x, hard_mask, latlons, day_of_year, day_of_week) in tqdm(train_dataloader):
        presto(x=x, mask = hard_mask, latlons=latlons, 
                        day_of_year = day_of_year, day_of_week = day_of_week)