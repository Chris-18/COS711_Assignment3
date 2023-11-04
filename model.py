import math
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import MeanSquaredError
from torchvision import models

import config


class CropDamageModel(pl.LightningModule):
    def __init__(self):
        super(CropDamageModel, self).__init__()
        self.resnet = models.resnet18()
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, config.NUM_CLASSES)
        self.fitnessFunction = nn.L1Loss()

    def forward(self, x):
        # Get the class logits from the ResNet model
        logits = self.resnet(x)

        # # Apply softmax to convert logits to class probabilities
        # probabilities = torch.softmax(logits, dim=1)
        #
        # # Get the class with the highest probability as the predicted class
        # _, predicted_class = torch.max(probabilities, 1)
        #
        # results = predicted_class * 10

        return logits

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
        pred = pred.squeeze()
        loss = self.fitnessFunction(pred, y)
        return loss, pred, y

    def predict_step(self, x):
        pred = self(x, None)
        # Apply softmax to convert pred to class probabilities
        probabilities = torch.softmax(pred, dim=1)

        # Get the class with the highest probability as the predicted class
        _, predicted_class = torch.max(probabilities, 1)

        results = predicted_class * 10
        return results

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        return optimizer
