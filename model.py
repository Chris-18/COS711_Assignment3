import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchmetrics import MeanSquaredError


class CropDamageModel(pl.LightningModule):
    def __init__(self, num_classes):
        super(CropDamageModel, self).__init__()
        self.resnet = models.resnet18()
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)
        self.fitnessFunction = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.resnet(x)

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
        loss = self.fitnessFunction(pred, y)
        return loss, pred, y

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        return optimizer
