import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from einops import repeat
import math

from presto.presto import Encoder, Decoder, Presto
from random import choice, randint, random, sample
from datasets.CollectionDataset import BANDS, BANDS_GROUPS_IDX, BAND_EXPANSION

class PrestoForecasting(pl.LightningModule):
    def __init__(self, encoder_config, normalized = False, ):
        super().__init__()
        self.lr = 0.001
        self.encoder = Encoder(**encoder_config)
        #decide regressor
        self.regressor = None #Parametrized RELU as activ function for MLP
        self.loss_fn = self.configure_loss_function()
        self.optimizer = self.configure_optimizers()
        self.normalized = normalized

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = self.lr)

    def configure_loss_function(self):
        return nn.MSELoss()

    def loss_function(self, outputs, y_true, loss_factor):
        loss_factor = torch.Tensor(loss_factor)
        loss = 0
        loss = torch.Tensor( loss )
        for i, output, label in enumerate(zip(outputs, y_true)):
            loss += loss_factor[i] * self.loss_fn(output, label)
        loss = loss/loss_factor.sum()
        return loss
    
    def forward(self, x, latlons, hard_mask = None, day_of_year = 0, day_of_week = 0):        
        x = self.encoder(x = x, mask = hard_mask, latlons = latlons, 
                        day_of_year = day_of_year, day_of_week = day_of_week)
        y_pred = self.regressor(x)
        return y_pred

    def training_step(self, batch, batch_idx):
        x, hard_mask, latlons, day_of_year, day_of_week, y_true, loss_factor = batch
        if self.normalized:
            # Normalized values        
            mean_values = x.mean(dim=(0, 1), keepdim=True)
            std_values = x.std(dim=(0, 1), unbiased=False, keepdim=True)
            x = (x - mean_values) / (std_values + 1e-05)
        # forward
        y_pred = self(x, latlons, hard_mask, day_of_year, day_of_week)
        loss = self.loss_function(y_pred, y_true, loss_factor)
        self.log('train_loss', loss.item(), logger=True, prog_bar=True, on_step=False, on_epoch=True)
        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx):
        x, hard_mask, latlons, day_of_year, day_of_week, y_true, loss_factor= batch
        if self.normalized:
            # Normalized values        
            mean_values = x.mean(dim=(0, 1), keepdim=True)
            std_values = x.std(dim=(0, 1), unbiased=False, keepdim=True)
            x = (x - mean_values) / (std_values + 1e-05)
        # forward
        y_pred = self(x, latlons, hard_mask, day_of_year, day_of_week)
        loss = self.loss_function(y_pred, y_true, loss_factor)
        self.log('val_loss', loss.item(), logger=True, prog_bar=True, on_step=False, on_epoch=True)
        return y_pred

    def test_step(self, batch, batch_idx):
        x, hard_mask, latlons, day_of_year, day_of_week, y_true, loss_factor = batch
        if self.normalized:
            # Normalized values        
            mean_values = x.mean(dim=(0, 1), keepdim=True)
            std_values = x.std(dim=(0, 1), unbiased=False, keepdim=True)
            x = (x - mean_values) / (std_values + 1e-05)
        # forward
        y_pred = self(x, latlons, hard_mask, day_of_year, day_of_week)
        loss = self.loss_function(y_pred, y_true, loss_factor)
        self.log('test_loss', loss.item(), logger=True, prog_bar=True, on_step=False, on_epoch=True)
        return y_pred