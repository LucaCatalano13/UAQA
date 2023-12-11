import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from transformers import AutoModelForSequenceClassification
from einops import repeat

from presto.presto import Presto
from random import choice, randint, random, sample
from datasets.CollectionDataset import BANDS_GROUPS_IDX, BAND_EXPANSION

MASK_STRATEGIES = (
    "group_bands",
    "random_timesteps",
    "chunk_timesteps",
    "random_combinations",
)

def random_masking(mask, num_tokens_to_mask: int):
    if num_tokens_to_mask > 0:
        # then, we flatten the mask and dw arrays
        all_tokens_mask = mask.flatten()
        unmasked_tokens = all_tokens_mask == False
        idx = np.flatnonzero(unmasked_tokens)
        np.random.shuffle(idx)
        idx = idx[:num_tokens_to_mask]
        all_tokens_mask[idx] = True
        mask = all_tokens_mask.reshape( mask.shape )
    return mask

def make_mask(x, strategy: str, mask_ratio: float):
    #x shape is [BS, TS , CH]
    num_timesteps = x.shape[1]
    num_band_groups = len(BANDS_GROUPS_IDX)
    #one mask for all the elem in batch repeated
    mask_shape = (num_timesteps , num_band_groups)
    mask = torch.full( mask_shape , False)
    num_tokens_to_mask = int( (num_timesteps * num_band_groups) * mask_ratio)
    
    if strategy == "random_combinations":
        mask = random_masking(mask, num_tokens_to_mask)

    elif strategy == "group_bands":
        num_band_groups_to_mask = int(num_tokens_to_mask / num_timesteps)
        num_tokens_to_mask -= num_timesteps * num_band_groups_to_mask
        assert num_tokens_to_mask >= 0
        
        band_groups = list(range(len(BANDS_GROUPS_IDX)))
        band_groups_to_mask = sample(band_groups, num_band_groups_to_mask)
        
        for band_group in band_groups_to_mask:
            mask[:, band_group] = True
                
        mask = random_masking(mask, num_tokens_to_mask)
        
    elif strategy == "random_timesteps":
        timesteps_to_mask = int(num_tokens_to_mask / num_band_groups)
        num_tokens_to_mask -= num_band_groups * timesteps_to_mask 
        timesteps = sample( list(range(len(num_timesteps))) , k = timesteps_to_mask )
        mask[timesteps] = True
        mask = random_masking( mask, num_tokens_to_mask )

    elif strategy == "chunk_timesteps":
        timesteps_to_mask = int(num_tokens_to_mask /num_band_groups )
        num_tokens_to_mask -=  num_band_groups * timesteps_to_mask
        start_idx = randint(0, num_timesteps - timesteps_to_mask)
        mask[start_idx : start_idx + timesteps_to_mask] = True 
        mask = random_masking(mask, num_tokens_to_mask)
        
    else:
        raise ValueError(f"Unknown strategy {strategy} not in {MASK_STRATEGIES}")

    mask = np.repeat(mask, BAND_EXPANSION, axis=1)   
    return repeat(
            mask , "t g -> b t g", b=x.shape[0]
        )

class BCELossWithSmoothing(nn.BCELoss):
    def __init__(
        self, smoothing: float = 0.1, weight=None, size_average=None, reduce=None, reduction="mean"
    ):
        super().__init__(
            weight=weight, size_average=size_average, reduce=reduce, reduction=reduction
        )
        assert smoothing < 1
        assert smoothing >= 0
        self.smoothing = smoothing

    def forward(self, input, target):
        return super().forward(
            input, torch.clamp(target, min=self.smoothing, max=(1 - self.smoothing))
        )


class PrestoMaskedLanguageModel(pl.LightningModule):

    def __init__(self, model):
        super().__init__()
        self.lr = 0.001
        self.model = model
        self.loss_fn = self.configure_loss_function()
        self.optimizer = self.configure_optimizers()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = self.lr)

    def configure_loss_function(self):
        return BCELossWithSmoothing()

    def loss_function(self, outputs, labels):
        return self.loss_fn(outputs, labels)
    
    def forward(self, x, latlons, soft_hard_mask = None, day_of_year = 0, day_of_week = 0):
        x = self.model(x = x, mask = soft_hard_mask, latlons = latlons, 
                        day_of_year = day_of_year, day_of_week = day_of_week)
        return x

    def training_step(self, batch, batch_idx):
        x, hard_mask, latlons, day_of_year, day_of_week = batch
        # define soft mask
        #TODO: train strategy problems: solve indx sampling, they must not be in hard
        soft_mask = make_mask(x , mask_ratio=.2)
        soft_mask_separated = torch.clone(soft_mask)
        soft_mask_separated[ hard_mask.bool() ] = False
        # mask x
        soft_hard_mask = torch.logical_or(soft_mask.bool(), hard_mask.bool())
        # label = masked_x
        labels = torch.Tensor(x[soft_mask])
        # forward
        reconstructed_x = self(x, latlons, soft_hard_mask, day_of_year, day_of_week)
        # compute loss between reconstructed_masked_x (of the masked positions) and masked_x (label)
        reconstructed_masked_x = reconstructed_x[soft_mask]
        loss = self.loss_function(reconstructed_masked_x, labels)
        return {"loss": loss}

    def inference_step(self, batch):
        x, hard_mask, latlons, day_of_year, day_of_week = batch
        # forward
        reconstructed_x = self(x, latlons, hard_mask, day_of_year, day_of_week)
        return reconstructed_x

    def validation_step(self, batch, batch_idx):
        return self.inference_step(batch)

    def test_step(self, batch, batch_idx):
        return self.inference_step(batch)

    def inference_epoch_end(self, outputs, inference_batch):
        x, hard_mask, latlons, day_of_year, day_of_week = inference_batch
        reconstructed_x = outputs
        # evaluate validation and test with the loss of the all values in dataset
        loss = self.loss_function(reconstructed_x[~hard_mask], x[~hard_mask])
        return {"loss": loss}