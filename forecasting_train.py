from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from pytorch_lightning.loggers import WandbLogger

from datasets.Era5 import Era5
from datasets.Dem import Dem
from datasets.Sentinel3 import Sentinel3
from datasets.Sentinel5 import Sentinel5
from datasets.LandCover import LandCover
from datasets.Stations import Stations
from datasets.CollectionDataset import CollectionDataset
import parser

from PixelTimeseriesLabeled import PixelTimeSeriesLabeled
from PrestoForecasting import PrestoForecasting
from PrestoMaskedLanguageModel import PrestoMaskedLanguageModel

if __name__ == "__main__":
    args = parser.parse_arguments()
    if args.input_train_path is not None:
        print("Loading Train Stations")
        train_stations = Stations(dataset_folder = args.stations_path, legend_folder = args.stations_legend_path, gold_data_path = args.golden_stations_path, gold_legend_path = args.golden_stations_legend_path)
        print("Loaded Train Stations")
        train_dataset = PixelTimeSeriesLabeled(stations = train_stations, num_timesteps=  args.num_timesteps, jump=args.jump, input_data_path = args.input_train_path)
    
    else:
        train_era = Era5(dataset_folder = args.era5_path, legend_folder = args.era5_legend_path)
        train_dem = Dem(dataset_folder = args.dem_path, legend_folder = args.dem_legend_path)
        train_sentinel3 = Sentinel3(dataset_folder = args.sentinel3_path, legend_folder = args.sentinel3_legend_path)
        train_sentinel5 = Sentinel5(dataset_folder = args.sentinel5_path, legend_folder = args.sentinel5_legend_path)
        train_land_cover = LandCover(dataset_folder = args.land_cover_path, legend_folder = args.land_cover_legend_path)
        train_collection_dataset = CollectionDataset(era = train_era, dem = train_dem, sentinel3 = train_sentinel3, 
                                                     sentinel5 = train_sentinel5, land_cover = train_land_cover)
        train_bound = train_land_cover.get_bound()
        print("Loading Train Stations")
        train_stations = Stations(dataset_folder = args.stations_path, legend_folder = args.stations_legend_path, gold_data_path = args.golden_stations_path, gold_legend_path = args.golden_stations_legend_path)
        print("Loaded Train Stations")
        train_dataset = PixelTimeSeriesLabeled(stations = train_stations, num_timesteps= args.num_timesteps,  jump=args.jump, collection_dataset=train_collection_dataset, bound=train_bound)
    
    # Normalize data
    train_dataset.normalize_data()

    if args.input_test_path is not None:
        print("Loading Test Stations")
        test_stations = Stations(dataset_folder = args.stations_test_path, legend_folder = args.stations_legend_path, gold_data_path = args.golden_stations_test_path, gold_legend_path = args.golden_stations_legend_path)
        print("Loaded Test Stations")
        test_dataset = PixelTimeSeriesLabeled(stations = test_stations, num_timesteps=args.num_timesteps, jump=args.jump, input_data_path = args.input_test_path)
    else:
        test_era = Era5(dataset_folder = args.era5_test_path, legend_folder = args.era5_legend_path)
        test_dem = Dem(dataset_folder = args.dem_test_path, legend_folder = args.dem_legend_path)
        test_sentinel3 = Sentinel3(dataset_folder = args.sentinel3_test_path, legend_folder = args.sentinel3_legend_path)
        test_sentinel5 = Sentinel5(dataset_folder = args.sentinel5_test_path, legend_folder = args.sentinel5_legend_path)
        test_land_cover = LandCover(dataset_folder = args.land_cover_test_path, legend_folder = args.land_cover_legend_path)
        test_collection_dataset = CollectionDataset(era = test_era, dem = test_dem, sentinel3 = test_sentinel3, 
                                                     sentinel5 = test_sentinel5, test_cover = test_land_cover)
        test_bound = test_land_cover.get_bound()
        print("Loading Test Stations")
        test_stations = Stations(dataset_folder = args.stations_test_path, legend_folder = args.stations_legend_path, gold_data_path = args.golden_stations_test_path, gold_legend_path = args.golden_stations_legend_path)
        print("Loaded Test Stations")
        test_dataset = PixelTimeSeriesLabeled(stations = test_stations, num_timesteps= args.num_timesteps, jump=args.jump, collection_dataset=test_collection_dataset, bound=test_bound)
    
    # Normalize data
    test_dataset.normalize_data()

    print("End dataset loading")
    if not args.only_test:
        # train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [int(len(train_dataset)*0.8), int(len(train_dataset)*0.2)])

        train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.2, random_state=42, shuffle=False)
        
        print("Train Dataloader loading")
        train_dataloader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                num_workers=2,
                shuffle=False,
            )
        print("Val Dataloader loading")
        val_dataloader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=False,
            )
    
    print("Test Dataloader loading")
    test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )

    #Load Encoedr Pretrained from PrestoMLM
    assert args.model_presto_path is not None

    print("Loading Encoder")
    kwargs_encoder = {"embedding_size": args.encoder_embedding_size, "channel_embed_ratio": args.encoder_channel_embed_ratio, 
                  "temp_embed_ratio": args.encoder_temp_embed_ratio, "depth": args.encoder_depth, 
                  "mlp_ratio": args.encoder_mlp_ratio, "num_heads": args.encoder_num_heads, "max_sequence_length": args.encoder_max_sequence_length}

    kwargs_decoder = {"encoder_embed_dim": args.encoder_embedding_size, "decoder_embed_dim": args.decoder_embed_dim,
                  "decoder_depth": args.decoder_depth, "decoder_num_heads": args.decoder_num_heads, 
                  "mlp_ratio": args.decoder_mlp_ratio, "max_sequence_length": args.decoder_max_sequence_length}
    
    kwargs_model = {"encoder_config": kwargs_encoder, "decoder_config": kwargs_decoder, "mask_ratio_random": args.mask_ratio_random, "mask_ratio_bands": args.mask_ratio_bands, 
                    "mask_ratio_timesteps": args.mask_ratio_timesteps, "normalized": False}
    
    prestoMLM = PrestoMaskedLanguageModel.load_from_checkpoint(args.model_presto_path, **kwargs_model)
    encoder = prestoMLM.encoder
    kwargs_model_forecasting = {"encoder": encoder, "normalized": True, "MLP_hidden_features" : args.MLP_hidden_features, "MLP_out_features" : args.MLP_out_features }
    
    #Load or initialize a Forecastinf model based on Presto Encoder
    if args.model_presto_forecasting_path is not None:
        # Checkpoint init
        presto_forecasting = PrestoForecasting.load_from_checkpoint(args.model_presto_forecasting_path, **kwargs_model_forecasting)
    else:
        # Random Xavier initialization
        presto_forecasting = PrestoForecasting(**kwargs_model_forecasting)
        
    wandb_logger = WandbLogger(project=args.wandb_project,
                            name=args.wandb_name,
                            log_model='all')

    arg_dict = vars(args)
    for k, v in arg_dict.items():
        wandb_logger.experiment.config[k] = v

    checkpoint_cb = ModelCheckpoint(
        monitor='val_loss',
        filename='_epoch({epoch:02d})',
        auto_insert_metric_name=False,
        save_weights_only=True,
        save_top_k=3,
        mode='min'
    )

    # Instantiate a trainer
    trainer = pl.Trainer(
        logger=wandb_logger,            
        accelerator='gpu',
        devices=[0],
        default_root_dir='./LOGS',  # Tensorflow can be used to viz
        num_sanity_val_steps=0,  # runs a validation step before stating training
        precision=16,  # we use half precision to reduce  memory usage
        max_epochs=args.num_epochs,
        check_val_every_n_epoch=1,  # run validation every epoch
        callbacks=[checkpoint_cb],  # we only run the checkpointing callback (you can add more)
        reload_dataloaders_every_n_epochs=1,  # we reload the dataset to shuffle the order
        log_every_n_steps=1, 
    )
    # Train or test only with a pretrained model
    if not args.only_test:
        trainer.validate(model=presto_forecasting, dataloaders=val_dataloader)
        trainer.fit(model=presto_forecasting, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model=presto_forecasting, dataloaders=test_dataloader)