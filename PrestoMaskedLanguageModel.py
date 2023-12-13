import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from einops import repeat
import math

from presto.presto import Presto
from random import choice, randint, random, sample
from datasets.CollectionDataset import BANDS, BANDS_GROUPS_IDX, BAND_EXPANSION

MASK_STRATEGIES = (
    "group_bands",
    "random_timesteps",
    "chunk_timesteps",
    "random_combinations",
)

def random_masking(mask, hard_mask,  num_tokens_to_mask_array: int):
    assert mask.shape[0] == num_tokens_to_mask_array.shape[0]
    for i in range(num_tokens_to_mask_array.shape[0]):
        if num_tokens_to_mask_array[i] > 0:
            # then, we flatten the mask and dw arrays
            all_tokens_mask = mask[i].flatten()
            unmasked_tokens = np.logical_and( all_tokens_mask == False , hard_mask[i].flatten() == False )
            idx = np.flatnonzero(unmasked_tokens)
            np.random.shuffle(idx)
            idx = idx[:num_tokens_to_mask_array[i]]
            all_tokens_mask[idx] = True
            mask[i] = all_tokens_mask.reshape( mask[i].shape )
    return mask

def make_mask(x, hard_mask, strategy: str, mask_ratio: float):
    # x shape is [BS, TS , CH]
    batch_size = x.shape[0]
    num_timesteps = x.shape[1]
    num_band_groups = len(BANDS)
    mask_shape = (batch_size, num_timesteps , num_band_groups)
    mask = torch.full(mask_shape , False)
    # each element of batch has same ratio of masked tokens given the hard mask
    num_tokens_to_mask = np.zeros(batch_size, dtype=int)
    for i in range(batch_size):
        num_tokens_to_mask[i] = int(((num_timesteps * num_band_groups) - hard_mask[i].sum()) * mask_ratio)
    
    if strategy == "random_combinations":
        mask = random_masking(mask, hard_mask, num_tokens_to_mask)

    elif strategy == "group_bands":
        #num_band_groups_to_mask = int(num_tokens_to_mask / num_timesteps)
        #num_tokens_to_mask -= num_timesteps * num_band_groups_to_mask
        #assert num_tokens_to_mask >= 0
        band_groups = list(BANDS_GROUPS_IDX.keys())
        #band_groups_to_mask = sample(band_groups, num_band_groups_to_mask)
        band_groups_to_mask = sample(band_groups, 1)

        for band_group in band_groups_to_mask:
            for idx in BANDS_GROUPS_IDX[band_group]:
                mask[:, :, idx ] = True
            mask[ hard_mask.bool() == True ] = False

        for i in range(batch_size):
            num_tokens_to_mask[i] -= mask[i].sum() 

        mask = random_masking(mask, hard_mask, num_tokens_to_mask)

    elif strategy == "random_timesteps":
        # x shape is [BS, TS , CH]
        # timesteps_to_mask = int(num_tokens_to_mask / num_band_groups)
        # timesteps_to_mask = int(num_timesteps * mask_ratio)
        print("Num timesteps: ", num_timesteps)
        num_timesteps_to_mask = math.ceil(num_timesteps * mask_ratio)
        timesteps_to_mask = np.zeros((batch_size, num_timesteps_to_mask), dtype=int)
        print("Num timesteps to mask: ", num_timesteps_to_mask)
        for i in range(batch_size):
            print(sample(range(0, num_timesteps), num_timesteps_to_mask))
            timesteps_to_mask[i] = sample(range(0, num_timesteps), num_timesteps_to_mask)
            mask[i][timesteps_to_mask[i]] = True
            mask[i][hard_mask[i].bool()] = False
            n_timesteps_tokens_not_masked = hard_mask[i][timesteps_to_mask[i]].sum()
            print("n_timesteps_not_masked", n_timesteps_tokens_not_masked)
            num_tokens_to_mask[i] = n_timesteps_tokens_not_masked
        mask = random_masking( mask, hard_mask, num_tokens_to_mask )

    elif strategy == "chunk_timesteps":
        print("Num timesteps: ", num_timesteps)
        num_timesteps_to_mask = math.ceil(num_timesteps * mask_ratio)
        timesteps_to_mask = np.zeros((batch_size, num_timesteps_to_mask), dtype=int)
        print("Num timesteps to mask: ", num_timesteps_to_mask)
        for i in range(batch_size):
            start_timestep = sample(range(0, num_timesteps), 1)[0]
            if (start_timestep + num_timesteps_to_mask) > num_timesteps:
                start_timestep -= ((start_timestep + num_timesteps_to_mask) - num_timesteps)
            timesteps_to_mask[i] = range(start_timestep, start_timestep + num_timesteps_to_mask)
            print("timesteps_to_mask", timesteps_to_mask[i])
            mask[i][timesteps_to_mask[i]] = True
            mask[i][hard_mask[i].bool()] = False
            n_timesteps_not_masked = hard_mask[i][timesteps_to_mask[i]].sum()
            print("n_timesteps_not_masked", n_timesteps_not_masked)
            num_tokens_to_mask[i] = n_timesteps_not_masked
        mask = random_masking(mask, hard_mask, num_tokens_to_mask)
        
    else:
        raise ValueError(f"Unknown strategy {strategy} not in {MASK_STRATEGIES}")

    #mask = np.repeat(mask, BAND_EXPANSION, axis=1)   
    return mask

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