import math
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import MeanSquaredError
from torchvision import models
import torch.nn.functional as F

import config


class LogisticsRegressionModel(pl.LightningModule):
    def __init__(self):
        super(LogisticsRegressionModel, self).__init__()
        self.resnet = models.resnet18()
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, 1)
        self.sigmoid = nn.Sigmoid()
        # self.fitnessFunction = nn.BCELoss() 


    def forward(self, x):
        logits = self.resnet(x)
        output = self.sigmoid(logits)  # Apply the sigmoid activation
        return output

    def training_step(self, batch, batch_idx):
        loss, pred, y = self.common_step(batch, batch_idx)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, pred, y = self.common_step(batch, batch_idx)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss, pred, y = self.common_step(batch, batch_idx)
        
        self.log("test_loss", loss)
        return loss

    def common_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        predictions = (pred >= 0.5).float()
        predictions = pred.squeeze()
        y= y.to(predictions.dtype)
        loss = nn.BCELoss()(predictions, y)
        self.log("val_loss", loss)
        return loss,pred, y

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        return optimizer
