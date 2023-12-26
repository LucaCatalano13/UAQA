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


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        bias=True,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer(num_parameters=hidden_features)
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class PrestoForecasting(pl.LightningModule):
    def __init__(self, encoder, normalized = False, MLP_hidden_features = 64, MLP_out_features = 7 ):
        super().__init__()
        #encoder
        self.encoder = encoder
        #regressor head
        self.MLP_hidden_features = MLP_hidden_features
        self.MLP_out_features = MLP_out_features
        self.regressor = Mlp(self.encoder.embedding_size, 
                             hidden_features=self.MLP_hidden_features, 
                             out_features = self.MLP_out_features , 
                             act_layer= nn.PReLU)
        #training params
        self.lr = 0.001
        self.loss_fn = self.configure_loss_function()
        self.optimizer = self.configure_optimizers()
        self.normalized = normalized

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = self.lr)

    def configure_loss_function(self):
        return nn.MSELoss(reduction='none')

    def loss_function(self, outputs, y_true, loss_factor):
        return torch.sum(loss_factor * (outputs - y_true) ** 2)
    
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
        self.log('train_loss', loss, logger=True, prog_bar=True, on_step=False, on_epoch=True)
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
        self.log('val_loss', loss, logger=True, prog_bar=True, on_step=False, on_epoch=True)
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
        self.log('test_loss', loss, logger=True, prog_bar=True, on_step=False, on_epoch=True)
        return y_pred