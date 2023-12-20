import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import wandb
from pytorch_lightning.loggers import WandbLogger

from datasets.Era5 import Era5
from datasets.Dem import Dem
from datasets.Sentinel3 import Sentinel3
from datasets.Sentinel5 import Sentinel5
from datasets.LandCover import LandCover
from datasets.CollectionDataset import CollectionDataset
import parser

from presto.presto import Encoder, Decoder, Presto
from PixelTimeseries import PixelTimeSeries
from PrestoMaskedLanguageModel import PrestoMaskedLanguageModel

if __name__ == "__main__":
    args = parser.parse_arguments()
    if args.input_train_path is not None:
        train_dataset = PixelTimeSeries(num_timesteps= args.num_timesteps , input_data_path = args.input_train_path)
    else:
        train_era = Era5(dataset_folder = args.era5_path, legend_folder = args.era5_legend_path)
        train_dem = Dem(dataset_folder = args.dem_path, legend_folder = args.dem_legend_path)
        train_sentinel3 = Sentinel3(dataset_folder = args.sentinel3_path, legend_folder = args.sentinel3_legend_path)
        train_sentinel5 = Sentinel5(dataset_folder = args.sentinel5_path, legend_folder = args.sentinel5_legend_path)
        train_land_cover = LandCover(dataset_folder = args.land_cover_path, legend_folder = args.land_cover_legend_path)
        train_collection_dataset = CollectionDataset(era = train_era, dem = train_dem, sentinel3 = train_sentinel3, 
                                                     sentinel5 = train_sentinel5, land_cover = train_land_cover)
        train_bound = train_land_cover.get_bound()
        train_dataset = PixelTimeSeries(num_timesteps=args.num_timesteps, collection_dataset=train_collection_dataset, bound=train_bound)
        
    if args.input_test_path is not None:
        test_dataset = PixelTimeSeries(num_timesteps=args.num_timesteps, input_data_path = args.input_test_path)
    else:
        test_era = Era5(dataset_folder = args.era5_test_path, legend_folder = args.era5_legend_path)
        test_dem = Dem(dataset_folder = args.dem_test_path, legend_folder = args.dem_legend_path)
        test_sentinel3 = Sentinel3(dataset_folder = args.sentinel3_test_path, legend_folder = args.sentinel3_legend_path)
        test_sentinel5 = Sentinel5(dataset_folder = args.sentinel5_test_path, legend_folder = args.sentinel5_legend_path)
        test_land_cover = LandCover(dataset_folder = args.land_cover_test_path, legend_folder = args.land_cover_legend_path)
        test_collection_dataset = CollectionDataset(era = test_era, dem = test_dem, sentinel3 = test_sentinel3, 
                                                     sentinel5 = test_sentinel5, test_cover = test_land_cover)
        test_bound = test_land_cover.get_bound()
        test_dataset = PixelTimeSeries(num_timesteps=args.num_timesteps, collection_dataset=test_collection_dataset, bound=test_bound)
    
    # train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [int(len(train_dataset)*0.8), int(len(train_dataset)*0.2)])

    train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.2, random_state=42)
    
    train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=2,
            shuffle=False,
        )

    val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )
    
    test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )

    if args.model_presto_path is not None:
        #Checkpoint init
        presto_ml = PrestoMaskedLanguageModel.load_from_checkpoint(args.model_presto_path)
    else:
        kwargs_encoder = {"embedding_size": args.encoder_embedding_size, "channel_embed_ratio": args.encoder_channel_embed_ratio, 
                  "temp_embed_ratio": args.encoder_temp_embed_ratio, "depth": args.encoder_depth, 
                  "mlp_ratio": args.encoder_mlp_ratio, "num_heads": args.encoder_num_heads, "max_sequence_length": args.encoder_max_sequence_length}

        kwargs_decoder = {"encoder_embed_dim": args.encoder_embedding_size, "decoder_embed_dim": args.decoder_embed_dim,
                  "decoder_depth": args.decoder_depth, "decoder_num_heads": args.decoder_num_heads, 
                  "mlp_ratio": args.decoder_mlp_ratio, "max_sequence_length": args.decoder_max_sequence_length}
        
        #Random Xavier initialization
        encoder = Encoder(**kwargs_encoder)
        decoder = Decoder(encoder.channel_embed, **kwargs_decoder)
        presto = Presto(encoder, decoder)
        presto_ml = PrestoMaskedLanguageModel(model = presto, mask_ratio_random=args.mask_ratio_random, 
                                              mask_ratio_bands=args.mask_ratio_bands, mask_ratio_timesteps=args.mask_ratio_timesteps, normalized=True)
    

    wandb_logger = WandbLogger(project=args.wandb_project,
                            name=args.wandb_name,
                            log_model='all')

    wandb_logger.experiment.config = args

    checkpoint_cb = ModelCheckpoint(
        monitor='loss',
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
        trainer.validate(model=presto_ml, dataloaders=val_dataloader)
        trainer.fit(model=presto_ml, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model=presto_ml, dataloaders=test_dataloader)