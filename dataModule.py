from collections import Counter

import pytorch_lightning as pl
import pandas as pd
import torchvision.transforms as transforms
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader
import torch

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
        self.probabilities = None
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        self.prep_data()

    def prep_data(self):
        full_dataset = pd.read_csv(self.csv_path)
        full_dataset = full_dataset.sample(n=config.INPUT_SIZE, random_state=42)

        target_length = 1000
        dataset_100 = self.extend_dataframe(
            full_dataset[full_dataset["extent"] == 100], target_length
        )
        dataset_90 = self.extend_dataframe(
            full_dataset[full_dataset["extent"] == 90], target_length
        )
        dataset_80 = self.extend_dataframe(
            full_dataset[full_dataset["extent"] == 80], target_length
        )
        dataset_70 = self.extend_dataframe(
            full_dataset[full_dataset["extent"] == 70], target_length
        )
        dataset_60 = self.extend_dataframe(
            full_dataset[full_dataset["extent"] == 60], target_length
        )
        dataset_50 = self.extend_dataframe(
            full_dataset[full_dataset["extent"] == 50], target_length
        )
        dataset_40 = self.extend_dataframe(
            full_dataset[full_dataset["extent"] == 40], target_length
        )
        dataset_30 = self.extend_dataframe(
            full_dataset[full_dataset["extent"] == 30], target_length
        )
        dataset_20 = self.extend_dataframe(
            full_dataset[full_dataset["extent"] == 20], target_length
        )
        dataset_10 = self.extend_dataframe(
            full_dataset[full_dataset["extent"] == 10], target_length
        )
        dataset_0 = self.extend_dataframe(
            full_dataset[full_dataset["extent"] == 0], target_length
        )

        concatenated_df = pd.concat(
            [
                dataset_0,
                dataset_10,
                dataset_20,
                dataset_30,
                dataset_40,
                dataset_50,
                dataset_60,
                dataset_70,
                dataset_80,
                dataset_90,
                dataset_100,
            ],
            axis=0,
        )

        # Reset the index
        concatenated_df.reset_index(drop=True, inplace=True)
        self.training_dataset = concatenated_df
        self.validation_dataset = concatenated_df.sample(n=50, random_state=42)
        self.test_dataset = concatenated_df.sample(n=25, random_state=30)

        # (
        #     training_dataset,
        #     validation_dataset,
        #     test_dataset,
        # ) = random_split(
        #     full_dataset, [config.TRAIN_SIZE, config.VALIDATION_SIZE, config.TEST_SIZE]
        # )
        # self.training_dataset = full_dataset.iloc[training_dataset.indices]
        # self.validation_dataset = full_dataset.iloc[validation_dataset.indices]
        # self.test_dataset = full_dataset.iloc[test_dataset.indices]

        extent_counts = Counter(self.training_dataset["extent"])
        sort_counts = sorted(extent_counts.items())
        probabilities = []
        total = self.training_dataset.shape[0]
        for sort_count in sort_counts:
            current = sort_count[1]
            probability = current / total
            probabilities.append(probability)
        self.probabilities = torch.tensor(probabilities)

    def extend_dataframe(self, original_df: pd.DataFrame, target_length):
        result_df = pd.DataFrame()
        # Continue adding rows randomly until the target row count is reached
        while result_df.shape[0] < target_length:
            # Randomly sample rows from the original DataFrame
            random_rows = original_df.sample(
                n=min(target_length - result_df.shape[0], original_df.shape[0])
            )

            # Concatenate the sampled rows with the result DataFrame
            result_df = pd.concat([result_df, random_rows])

        # Reset the index of the result DataFrame
        result_df.reset_index(drop=True, inplace=True)
        return result_df

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
            persistent_workers=True,
        )
