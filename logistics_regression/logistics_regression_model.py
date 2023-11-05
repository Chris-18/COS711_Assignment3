import math
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import MeanSquaredError
from torchvision import models
import torch.nn.functional as F


class LogisticsRegressionModel(pl.LightningModule):
    def __init__(self, config):
        super(LogisticsRegressionModel, self).__init__()
        self.resnet = models.resnet18()
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, 1)
        self.sigmoid = nn.Sigmoid()
        self.learning_rate = config["learning_rate"]
        self.eval_loss = []
        # self.fitnessFunction = nn.BCELoss()
        self.optimizer = config["optimizer"]

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
        self.eval_loss.append(loss)
        self.log("val_loss", loss)
        return loss

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.eval_loss).mean()
        # avg_mse = torch.stack(self.eval_mse).mean()
        self.log("val_loss", avg_loss, sync_dist=True)
        # self.log("log_mse", avg_mse, sync_dist=True)
        self.eval_loss.clear()
        # self.eval_mse.clear()

    def test_step(self, batch, batch_idx):
        loss, pred, y = self.common_step(batch, batch_idx)

        self.log("test_loss", loss)
        return loss

    def common_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        predictions = (pred >= 0.5).float()
        predictions = pred.squeeze()
        y = y.to(predictions.dtype)
        loss = nn.BCELoss()(predictions, y)
        self.log("val_loss", loss)
        return loss, pred, y

    def configure_optimizers(self):
        if self.optimizer == "adam":
            optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer == "sgd":
            optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)
        elif self.optimizer == "adagrad":
            optimizer = optim.Adagrad(self.parameters(), lr=self.learning_rate)
        else:
            raise Exception("Invalid optimizer")
        return optimizer
