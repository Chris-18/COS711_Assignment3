import pytorch_lightning as pl
import pandas as pd
import torchvision.transforms as transforms
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader

import config
from torch.utils.data import random_split
from dataset import CropDamageDataset


class CropDamageDataModule(pl.LightningDataModule):
    def __init__(self, csv_path, root_dir):
        super().__init__()
        self.csv_path = csv_path
        self.root_dir = root_dir
        self.test_dataset = None
        self.validation_dataset = None
        self.training_dataset = None
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def setup(self, stage: str):
        full_dataset = pd.read_csv(self.csv_path)
        full_dataset = full_dataset.sample(n=config.INPUT_SIZE, random_state=42)

        (
            training_dataset,
            validation_dataset,
            test_dataset,
        ) = random_split(
            full_dataset, [config.TRAIN_SIZE, config.VALIDATION_SIZE, config.TEST_SIZE]
        )
        self.training_dataset = full_dataset.iloc[training_dataset.indices]
        self.validation_dataset = full_dataset.iloc[validation_dataset.indices]
        self.test_dataset = full_dataset.iloc[test_dataset.indices]

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        train_data = CropDamageDataset(
            self.training_dataset, self.root_dir, transform=self.transform
        )
        return DataLoader(
            train_data,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            persistent_workers=True,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        validation_data = CropDamageDataset(
            self.validation_dataset, self.root_dir, transform=self.transform
        )
        return DataLoader(
            validation_data,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            persistent_workers=True,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        test_data = CropDamageDataset(
            self.test_dataset, self.root_dir, transform=self.transform
        )
        return DataLoader(
            test_data,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
        )
