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
from datasets.Stations import STATIONS_BANDS

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

        self.fc1 = nn.Linear(in_features, hidden_features[0], bias=bias)
        self.act1 = act_layer(num_parameters=hidden_features[0])
        # self.drop1 = nn.Dropout(drop)
        
        self.fc2 = nn.Linear(hidden_features[0], hidden_features[1], bias=bias)
        self.act2 = act_layer(num_parameters=hidden_features[1])
        
        self.fc3 = nn.Linear(hidden_features[1], out_features, bias = bias)
        # self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        # x = self.drop1(x)
        x = self.fc2(x)
        x = self.act2(x)
        # x = self.drop2(x)
        x = self.fc3(x)
        return x


class PrestoForecasting(pl.LightningModule):
    def __init__(self, encoder, normalized = False, MLP_hidden_features = 64, MLP_out_features = 7 ):
        super().__init__()
        #encoder
        self.encoder = encoder
        #regressor head
        self.MLP_hidden_features = MLP_hidden_features
        self.MLP_out_features = MLP_out_features
        # TODO: MLP per inquinante
        self.regressors = [ Mlp(self.encoder.embedding_size, 
                                hidden_features= [self.MLP_hidden_features, self.MLP_hidden_features//2] , 
                                out_features = 1 , 
                                act_layer= nn.PReLU) 
                                    for _ in range(self.MLP_out_features) ]
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for regressor in self.regressors:
            regressor.to(device)
        
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
        #TODO: try with / outputs.shape[0] to weight differently batches w.r.t. factors
        return torch.sum(loss_factor * (outputs - y_true) ** 2) / torch.sum(loss_factor)
    
    def forward(self, x, latlons, hard_mask = None, day_of_year = 0, day_of_week = 0):        
        x = self.encoder(x = x, mask = hard_mask, latlons = latlons, 
                        day_of_year = day_of_year, day_of_week = day_of_week)
        y_pred = torch.Tensor([regressor(x) for regressor in self.regressors])
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

        self.log_metrics(y_pred, y_true, loss_factor, "TRAIN")
        
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
        
        self.log_metrics(y_pred, y_true, loss_factor, "VAL")
        self.log('val_loss', loss.item(), logger=True, prog_bar=True, on_step=False, on_epoch=True)
        return y_pred

    #TODO: solo gold stations?
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
        
        self.log_metrics(y_pred, y_true, loss_factor, "TEST")
        self.log('test_loss', loss.item(), logger=True, prog_bar=True, on_step=False, on_epoch=True)
        return y_pred
    
    
    def log_metrics(self , y_pred, y_true, loss_factor, str_step):
        for i, pollutant in enumerate(STATIONS_BANDS):
            mae = loss_factor[:][i] * torch.abs( y_pred[:][i] - y_true[:][i])
            batch_avg_mae = torch.sum( mae ) / torch.sum(loss_factor[:][i])
            batch_avg_percentage_error = (torch.sum( (mae/y_true[:][i])  ) / torch.sum(loss_factor[:][i])) * 100
            self.log(f'{str_step}: MAE of {pollutant}', batch_avg_mae, logger=True, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f'{str_step}: % error of {pollutant}', batch_avg_percentage_error, logger=True, prog_bar=True, on_step=False, on_epoch=True)