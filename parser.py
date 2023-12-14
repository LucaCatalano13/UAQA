import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=128,
                        help="batch size")
    
    parser.add_argument("--num_epochs", type=int, default=20,
                        help="number of epochs")
    
    # Architecture parameters
    parser.add_argument("--only_test", default=False,
                        help="avoid the train phase and test on --test_path")
    
    parser.add_argument("--input_data_path", type=str, default="/content/drive/MyDrive/data_small_file.pt",
                        help="path for loading input data")
    
    parser.add_argument("--input_test_data_path", type=str, default="data_test_file.pt",
                        help="path for loading input data")
    
    parser.add_argument("--model_presto_path", type=str, default=None,
                        help="path for loading presto")
    
    # Visualizations parameters

    # Paths parameters
    # train and validation paths
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
    # test paths
    parser.add_argument("--era5_test_path", type=str, default="milan_crop_23/era5_2023",
                        help="path to Era5 test dataset")
    parser.add_argument("--land_cover_test_path", type=str, default="milan_crop_23/land_cover",
                        help="path to Land Cover test dataset")
    parser.add_argument("--sentinel3_test_path", type=str, default="milan_crop_23/sentinel3_lst_23",
                        help="path to Sentinel 3 test dataset")
    parser.add_argument("--sentinel5_test_path", type=str, default="milan_crop_23/sentinel5p_23",
                        help="path to Sentinel 5 test dataset")
    parser.add_argument("--dem_test_path", type=str, default="milan_crop_23/dem",
                        help="path to Dem test dataset")
    # legend paths
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
