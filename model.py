import math
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import MeanSquaredError
from torchvision import models


class CropDamageModel(pl.LightningModule):
    def __init__(self, config):
        super(CropDamageModel, self).__init__()
        self.resnet = models.resnet18()
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, 1)
        self.fitnessFunction = nn.L1Loss()
        self.learning_rate = config["learning_rate"]
        self.eval_loss = []
        self.eval_mse = []
        self.optimizer = config["optimizer"]

    def forward(self, x):
        # Get the class logits from the ResNet model
        return self.resnet(x)

    def training_step(self, batch, batch_idx):
        loss, pred, y = self.common_step(batch, batch_idx)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, pred, y = self.common_step(batch, batch_idx)
        # mse = MeanSquaredError(pred, y)
        # self.eval_mse.append(mse)
        self.eval_loss.append(loss)
        self.log("val_loss", loss)
        return {"val_loss": loss}

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
        pred = pred.squeeze()
        loss = self.fitnessFunction(pred, y)
        return loss, pred, y

    def predict_step(self, x):
        return self(x)

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
