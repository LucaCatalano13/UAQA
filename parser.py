import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Training parameters

    
    # Architecture parameters
    parser.add_argument("--load_data_input", type=bool, default=True,
                        help="load checkpoint of input data")
    
    parser.add_argument("--input_data_path", type=str, default="/data_file.pt",
                        help="path for loading input data")
    
    # Visualizations parameters

    # Paths parameters
    parser.add_argument("--era5_path", type=str, default="milan_crop/era5",
                        help="path to Era5 dataset")
    parser.add_argument("--land_cover_path", type=str, default="milan_crop/land_cover",
                        help="path to Land Cover dataset")
    parser.add_argument("--sentinel3_path", type=str, default="milan_crop/sentinel3",
                        help="path to Sentinel 3 dataset")
    parser.add_argument("--sentinel5_path", type=str, default="milan_crop/sentinel5p",
                        help="path to Sentinel 5 dataset")
    parser.add_argument("--dem_path", type=str, default="milan_crop/dem",
                        help="path to Dem dataset")

    parser.add_argument("--era5_legend_path", type=str, default="milan_crop_legend/era5",
                            help="path to Era5 dataset")
    parser.add_argument("--land_cover_legend_path", type=str, default="milan_crop_legend/land_cover",
                        help="path to Land Cover dataset")
    parser.add_argument("--sentinel3_legend_path", type=str, default="milan_crop_legend/sentinel3",
                        help="path to Sentinel 3 dataset")
    parser.add_argument("--sentinel5_legend_path", type=str, default="milan_crop_legend/sentinel5p",
                        help="path to Sentinel 5 dataset")
    parser.add_argument("--dem_legend_path", type=str, default="milan_crop_legend/dem",
                        help="path to Dem dataset")


    # Visualization parameters

    args = parser.parse_args()
    return args
