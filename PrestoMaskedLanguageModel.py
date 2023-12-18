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

BANDS_GROUPS_NOT_TO_TRAIN_ON_MLM = ["LC" , "DEM"]
BANDS_NOT_TO_TRAIN_ON_MLM = ['Urban',
 'Road',
 'Railways',
 'Port',
 'Airports',
 'Extraction',
 'NoUse',
 'Green',
 'OpenSpaces',
 'Water',
 'DEM']


def random_masking(mask, harder_mask,  num_tokens_to_mask_array: int):
    assert mask.shape[0] == num_tokens_to_mask_array.shape[0]
    for i in range(num_tokens_to_mask_array.shape[0]):
        if num_tokens_to_mask_array[i] > 0:
            # then, we flatten the mask and dw arrays
            all_tokens_mask = mask[i].flatten()
            unmasked_tokens = np.logical_and( all_tokens_mask.cpu() == False , harder_mask[i].cpu().flatten() == False )
            idx = np.flatnonzero(unmasked_tokens)
            np.random.shuffle(idx)
            idx = idx[:num_tokens_to_mask_array[i]]
            all_tokens_mask[idx] = True
            mask[i] = all_tokens_mask.reshape( mask[i].shape )
    return mask

def make_mask(x, hard_mask, strategy: str, mask_ratio_random: float, mask_ratio_timesteps: float, 
              mask_ratio_bands: float, bands_not_to_mask:list = BANDS_NOT_TO_TRAIN_ON_MLM ):
    # x shape is [BS, TS , CH]
    batch_size = x.shape[0]
    num_timesteps = x.shape[1]
    num_bands = len(BANDS) 
    #mask of same shape of x all False
    mask_shape = (batch_size, num_timesteps , num_bands)
    mask = torch.full(mask_shape , False)
    #avoid some bands during MLM training (usually static ones)
    index_not_to_train_on = [BANDS.index(value) for value in bands_not_to_mask]
    harder_mask = hard_mask.clone()
    harder_mask[:, :, index_not_to_train_on] = True
    
    num_tokens_to_mask = np.zeros(batch_size, dtype=int)
    for i in range(batch_size): 
        num_tokens_to_mask[i] = int(((num_timesteps * num_bands) - harder_mask[i].sum()) * mask_ratio_random)
    
    if strategy == "random_combinations":
        mask = random_masking(mask, harder_mask, num_tokens_to_mask)

    elif strategy == "group_bands":
        #num_band_groups_to_mask = int(num_tokens_to_mask / num_timesteps)
        #num_tokens_to_mask -= num_timesteps * num_band_groups_to_mask
        #assert num_tokens_to_mask >= 0
        band_groups = list(BANDS_GROUPS_IDX.keys())
        #band_groups_to_mask = sample(band_groups, num_band_groups_to_mask)
        num_bands_group_to_mask = math.ceil(len(band_groups) * mask_ratio_bands)
        band_groups_to_mask = sample(band_groups, num_bands_group_to_mask)

        for band_group in band_groups_to_mask:
            for idx in BANDS_GROUPS_IDX[band_group]:
                mask[:, :, idx ] = True
            mask[ harder_mask.bool() == True ] = False

        for i in range(batch_size):
            num_tokens_to_mask[i] -= mask[i].sum() 

        mask = random_masking(mask, harder_mask, num_tokens_to_mask)

    elif strategy == "random_timesteps":
        # x shape is [BS, TS , CH]
        # timesteps_to_mask = int(num_tokens_to_mask / num_band_groups)
        # timesteps_to_mask = int(num_timesteps * mask_ratio)

        num_timesteps_to_mask = math.ceil(num_timesteps * mask_ratio_timesteps)
        timesteps_to_mask = np.zeros((batch_size, num_timesteps_to_mask), dtype=int)
        for i in range(batch_size):
            timesteps_to_mask[i] = sample(range(0, num_timesteps), num_timesteps_to_mask)
            mask[i][timesteps_to_mask[i]] = True
            mask[i][harder_mask[i].bool()] = False
            n_timesteps_tokens_not_masked = harder_mask[i][timesteps_to_mask[i]].sum()
            num_tokens_to_mask[i] = n_timesteps_tokens_not_masked
        mask = random_masking( mask, harder_mask, num_tokens_to_mask )

    elif strategy == "chunk_timesteps":
        num_timesteps_to_mask = math.ceil(num_timesteps * mask_ratio_timesteps)
        timesteps_to_mask = np.zeros((batch_size, num_timesteps_to_mask), dtype=int)
        for i in range(batch_size):
            start_timestep = sample(range(0, num_timesteps), 1)[0]
            if (start_timestep + num_timesteps_to_mask) > num_timesteps:
                start_timestep -= ((start_timestep + num_timesteps_to_mask) - num_timesteps)
            timesteps_to_mask[i] = range(start_timestep, start_timestep + num_timesteps_to_mask)
            mask[i][timesteps_to_mask[i]] = True
            mask[i][harder_mask[i].bool()] = False
            n_timesteps_not_masked = harder_mask[i][timesteps_to_mask[i]].sum()
            num_tokens_to_mask[i] = n_timesteps_not_masked
        mask = random_masking(mask, harder_mask, num_tokens_to_mask)
        
    else:
        raise ValueError(f"Unknown strategy {strategy} not in {MASK_STRATEGIES}")

    #mask = np.repeat(mask, BAND_EXPANSION, axis=1)   
    return mask

class PrestoMaskedLanguageModel(pl.LightningModule):
    def __init__(self, model, mask_ratio_random, mask_ratio_timesteps, mask_ratio_bands, bands_not_to_mask = BANDS_NOT_TO_TRAIN_ON_MLM, normalized = False):
        super().__init__()
        self.lr = 0.001
        self.model = model
        self.loss_fn = self.configure_loss_function()
        self.optimizer = self.configure_optimizers()
        self.mask_ratio_random = mask_ratio_random
        self.mask_ratio_bands = mask_ratio_bands
        self.mask_ratio_timesteps = mask_ratio_timesteps
        self.bands_not_to_mask = bands_not_to_mask
        self.normalized = normalized

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = self.lr)

    def configure_loss_function(self):
        return nn.MSELoss()

    def loss_function(self, outputs, labels):
        return self.loss_fn(outputs, labels)
    
    def forward(self, x, latlons, soft_hard_mask = None, day_of_year = 0, day_of_week = 0):        
        x = self.model(x = x, mask = soft_hard_mask, latlons = latlons, 
                        day_of_year = day_of_year, day_of_week = day_of_week)
        return x

    def training_step(self, batch, batch_idx):
        x, hard_mask, latlons, day_of_year, day_of_week = batch
        if self.normalized:
            # Normalized values        
            mean_values = x.mean(dim=(0, 1), keepdim=True)
            std_values = x.std(dim=(0, 1), unbiased=False, keepdim=True)
            x = (x - mean_values) / (std_values + 1e-05)
        # define soft mask
        soft_mask = make_mask(x=x, hard_mask = hard_mask, strategy = MASK_STRATEGIES[randint(0, len(MASK_STRATEGIES) - 1)], 
                              mask_ratio_random = self.mask_ratio_random, mask_ratio_bands = self.mask_ratio_bands, 
                              mask_ratio_timesteps = self.mask_ratio_timesteps, bands_not_to_mask = self.bands_not_to_mask)
        soft_mask_separated = torch.clone(soft_mask)
        soft_mask_separated[ hard_mask.bool() ] = False
        # mask x
        # soft_hard_mask = torch.logical_or(soft_mask.cpu().bool(), hard_mask.cpu().bool())
        # label = masked_x
        labels = torch.Tensor(x[soft_mask])
        # forward
        reconstructed_x = self(x, latlons, soft_mask_separated, day_of_year, day_of_week)
        # compute loss between reconstructed_masked_x (of the masked positions) and masked_x (label)
        reconstructed_masked_x = reconstructed_x[soft_mask]
        loss = self.loss_function(reconstructed_masked_x, labels)
        print(loss)
        self.log('loss', loss.item(), logger=True, prog_bar=True, on_step=False, on_epoch=True)
        return {"loss": loss}

    def inference_step(self, batch):
        x, hard_mask, latlons, day_of_year, day_of_week = batch
        if self.normalized:
            # Normalized values        
            mean_values = x.mean(dim=(0, 1), keepdim=True)
            std_values = x.std(dim=(0, 1), unbiased=False, keepdim=True)
            x = (x - mean_values) / (std_values + 1e-05)
        # forward
        reconstructed_x = self(x, latlons, hard_mask, day_of_year, day_of_week)
        return reconstructed_x

    def validation_step(self, batch, batch_idx):
        return self.inference_step(batch)

    def test_step(self, batch, batch_idx):
        return self.inference_step(batch)
    
    def inference_epoch_end(self, outputs, inference_batch):
        x, hard_mask, latlons, day_of_year, day_of_week = inference_batch
        if self.normalized:
            # Normalized values        
            mean_values = x.mean(dim=(0, 1), keepdim=True)
            std_values = x.std(dim=(0, 1), unbiased=False, keepdim=True)
            x = (x - mean_values) / (std_values + 1e-05)
        reconstructed_x = outputs
        # evaluate validation and test with the loss of the all values in dataset
        loss = self.loss_function(reconstructed_x[~hard_mask], x[~hard_mask])
        print("Validation Loss: ", loss)
        self.log('loss', loss.item(), logger=True)
        return {"loss": loss}