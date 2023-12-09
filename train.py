import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
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
from PrestoMaskedLanguageModel import PrestoMaskedLanguageModel

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

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])
    
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
    
    # TODO
    test_dataloader = DataLoader()

    encoder = Encoder()
    decoder = Decoder(encoder.channel_embed)
    presto = Presto(encoder, decoder)

    if args.load_model_presto_path is not None:
        presto_ml = PrestoMaskedLanguageModel.load_from_checkpoint(args.model_presto_path)
    else:
        presto_ml = PrestoMaskedLanguageModel(model = presto)
    

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
        accelerator='gpu',
        devices=[0],
        default_root_dir='./LOGS',  # Tensorflow can be used to viz
        num_sanity_val_steps=0,  # runs a validation step before stating training
        precision=16,  # we use half precision to reduce  memory usage
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=1,  # run validation every epoch
        callbacks=[checkpoint_cb],  # we only run the checkpointing callback (you can add more)
        reload_dataloaders_every_n_epochs=1,  # we reload the dataset to shuffle the order
        log_every_n_steps=20,
    )

    # Train or test only with a pretrained model
    if not args.only_test:
        trainer.validate(model=presto_ml, dataloaders=val_dataloader)
        trainer.fit(model=presto_ml, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model=presto_ml, dataloaders=test_dataloader)