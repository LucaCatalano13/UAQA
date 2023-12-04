import torch

from datasets.Era5 import Era5
from datasets.Dem import Dem
from datasets.Sentinel3 import Sentinel3
from datasets.Sentinel5 import Sentinel5
from datasets.LandCover import LandCover
from datasets.CollectionDataset import CollectionDataset
from utils import process_images
import parser

if __name__ == "__main__":
    args = parser.parse_arguments()
    era = Era5(dataset_folder = args.era5_path, legend_folder = args.era5_legend_path)
    dem = Dem(dataset_folder = args.dem_path, legend_folder = args.dem_legend_path)
    sentinel3 = Sentinel3(dataset_folder = args.sentinel3_path, legend_folder = args.sentinel3_legend_path)
    sentinel5 = Sentinel5(dataset_folder = args.sentinel5_path, legend_folder = args.sentinel5_legend_path)
    land_cover = LandCover(dataset_folder = args.land_cover_path, legend_folder = args.land_cover_legend_path)
    collection_dataset = CollectionDataset(era = era, dem = dem, sentinel3 = sentinel3, sentinel5 = sentinel5, land_cover = land_cover)
    if args.load_data_input:
        data = torch.load(args.input_data_path)
        print(len(data))
    else:
        data = process_images(collection_dataset, land_cover.get_bound())
        torch.save(data, args.input_data_path)