import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    #log parameters
    
    parser.add_argument("--wandb_project", type=str, default="UAQA",
                        help="project on wanddb where to log into")
    
    parser.add_argument("--wandb_name", type=str, default= None,
                        help="Name of current experiment")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=128,
                        help="batch size")
    
    parser.add_argument("--num_workers", type=int, default=2,
                        help="batch size")
    
    parser.add_argument("--num_timesteps", type=int, default=7,
                        help="number of days for each training slice of pixel timeseries")

    parser.add_argument("--jump", type=int, default=2,
                        help="jump number of days for each training slice of pixel timeseries")
    
    parser.add_argument("--mask_ratio_random", type=float, default=0.2,
                        help="mask ratio random strategy for soft mask")
    
    parser.add_argument("--mask_ratio_timesteps", type=float, default=0.15,
                        help="mask timesteps strategy for soft mask")
    
    parser.add_argument("--mask_ratio_bands", type=float, default=0.2,
                        help="mask bands strategy for soft mask")
    
    parser.add_argument("--num_epochs", type=int, default=20,
                        help="number of epochs")
    
    parser.add_argument("--only_test", default=False,
                        help="avoid the train phase and test on --test_path")
    
    # Finetuning Training Parameters
    parser.add_argument("--loss_default_factor", type = float, default=0.3,
                        help="loss factor to weight the weakly label during finetuning training")
    
    parser.add_argument("--MLP_hidden_features", type = int, default=64,
                        help="size of hidden layer of MLP regression head")
    
    parser.add_argument("--MLP_out_features", type = int, default=7,
                        help="number of target values to predict with MLP regression head")
    
    # Architecture parameters
    # Encoder
    parser.add_argument("--encoder_embedding_size", type = int,  default= 128,
                        help="dimension of enocder latent space")
    
    parser.add_argument("--encoder_channel_embed_ratio", type = float,  default= 0.25,
                        help="Positional encoding used for channel embedding")
    
    parser.add_argument("--encoder_temp_embed_ratio", type = float,default= 0.25,
                        help="Positional encoding used for temporal embedding")
    
    parser.add_argument("--encoder_depth", type = int,default= 2,
                        help="number of blocks in encoder")
    
    parser.add_argument("--encoder_mlp_ratio",type = int, default= 2,
                        help="ratio of mlp")
    
    parser.add_argument("--encoder_num_heads", type = int,default= 8,
                        help="Number of heads in encoder")
    
    parser.add_argument("--encoder_max_sequence_length", type = int,default= 24,
                        help="Max sequence lenght in encoder")
    
    # Decoder    
    parser.add_argument("--decoder_embed_dim", type = int,default= 128,
                        help="dimension of decoder latent space")
    
    parser.add_argument("--decoder_depth", type = int,default= 2,
                        help="number of blocks in decoder")
    
    parser.add_argument("--decoder_mlp_ratio", type = int,default= 2,
                        help="ratio of mlp")
    
    parser.add_argument("--decoder_num_heads", type = int,default= 8,
                        help="Number of heads in decoder")
    
    parser.add_argument("--decoder_max_sequence_length",type = int, default= 24,
                        help="Max sequence lenght in decoder")
    
    # Paths parameters
    parser.add_argument("--input_train_path", type=str, default=None,
                        help="path for loading input data")
    
    parser.add_argument("--input_test_path", type=str, default=None,
                        help="path for loading input data")
    
    parser.add_argument("--model_presto_path", type=str, default=None,
                        help="path for loading presto")
    
    parser.add_argument("--model_presto_forecasting_path", type=str, default=None,
                        help="path for loading presto regression model")
    
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
    parser.add_argument("--stations_path", type=str, default=None,
                        help="path to stations .tiff dataset")
    parser.add_argument("--golden_stations_path", type=str, default=None,
                        help="path to stations .csv dataset")
     
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
    parser.add_argument("--stations_test_path", type=str, default=None,
                        help="path to stations .tiff dataset")
    parser.add_argument("--golden_stations_test_path", type=str, default=None,
                        help="path to stations .csv dataset")
    
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
    parser.add_argument("--stations_legend_path", type=str, default=None,
                        help="path to stations .tiff dataset legend")
    parser.add_argument("--golden_stations_legend_path", type=str, default=None,
                        help="path to stations .csv dataset legend")


    # Visualization parameters

    args = parser.parse_args()
    return args
