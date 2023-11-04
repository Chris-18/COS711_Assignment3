from collections import Counter

import pytorch_lightning as pl
import pandas as pd
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader
import torch

import config as c
from torch.utils.data import random_split
from dataset import CropDamageDataset


class CropDamageDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, tuning=False):
        super().__init__()
        self.csv_path = c.CSV_FILE
        self.root_dir = c.ROOT_DIR
        self.test_dataset = None
        self.validation_dataset = None
        self.training_dataset = None
        self.probabilities = None
        self.batch_size = batch_size
        self.is_tuning = tuning
        self.prep_data()

    def prep_data(self):
        full_dataset = pd.read_csv(self.csv_path)
        full_dataset = full_dataset.sample(n=c.INPUT_SIZE, random_state=42)
        full_dataset = full_dataset[full_dataset["extent"] > 0]

        total_samples = len(full_dataset)

        train_percentage = c.TRAIN_PERCENTAGE  # 80% for training
        test_percentage = c.TEST_PERCENTAGE  # 10% for testing

        train_size = int(train_percentage * total_samples)
        test_size = int(test_percentage * total_samples)
        validation_size = total_samples - train_size - test_size

        training_dataset, validation_dataset, test_dataset = random_split(
            full_dataset, [train_size, validation_size, test_size]
        )

        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset.dataset.iloc[
            validation_dataset.indices
        ]
        self.test_dataset = test_dataset.dataset.iloc[test_dataset.indices]

        if self.is_tuning:
            target_length = 200
        else:
            target_length = 1000
        extended_datasets = []
        for extent_value in range(0, 101, 10):
            extent_dataset = self.extend_dataframe(
                training_dataset.dataset[
                    training_dataset.dataset["extent"] == extent_value
                ],
                target_length,
            )
            extended_datasets.append(extent_dataset)

        concatenated_df = pd.concat(extended_datasets, axis=0)

        # Reset the index
        concatenated_df.reset_index(drop=True, inplace=True)
        self.training_dataset = concatenated_df

    def extend_dataframe(self, original_df: pd.DataFrame, target_length):
        result_df = pd.DataFrame()
        if len(original_df) == 0:
            return result_df
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
        train_data = CropDamageDataset(self.training_dataset, self.root_dir)
        return DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=c.NUM_WORKERS,
            persistent_workers=True,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        validation_data = CropDamageDataset(self.validation_dataset, self.root_dir)
        return DataLoader(
            validation_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=c.NUM_WORKERS,
            persistent_workers=True,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        test_data = CropDamageDataset(self.test_dataset, self.root_dir)
        return DataLoader(
            test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=c.NUM_WORKERS,
            persistent_workers=True,
        )
