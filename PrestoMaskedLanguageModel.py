import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from transformers import AutoModelForSequenceClassification

from presto.presto import Presto

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
        # TODO
        soft_mask = None
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