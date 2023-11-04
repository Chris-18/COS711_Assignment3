from collections import Counter

import pytorch_lightning as pl
import pandas as pd
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader
import torch

import config
from torch.utils.data import random_split
from dataset import CropDamageDataset


class CropDamageDataModule(pl.LightningDataModule):
    def __init__(self, csv_path, root_dir, test_csv_path, test_dir):
        super().__init__()
        self.csv_path = csv_path
        self.root_dir = root_dir
        self.test_dataset = None
        self.validation_dataset = None
        self.training_dataset = None
        self.probabilities = None
        self.test_csv_path = test_csv_path
        self.test_dir = test_dir
        self.prep_data()

    def prep_data(self):
        full_dataset = pd.read_csv(self.csv_path)
        full_dataset = full_dataset.sample(n=config.INPUT_SIZE, random_state=42)

        train_size = config.TRAIN_SIZE
        validation_size = config.VALIDATION_SIZE
        test_size = config.TEST_SIZE

        training_dataset, validation_dataset, test_dataset = random_split(
            full_dataset, [train_size, validation_size, test_size]
        )

        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset

        target_length = 1000
        extended_datasets = []
        for extent_value in range(0, 101, 10):
            extent_dataset = self.extend_dataframe(
                training_dataset.dataset[training_dataset.dataset["extent"] == extent_value], target_length
            )
            extended_datasets.append(extent_dataset)

        concatenated_df = pd.concat(extended_datasets, axis=0)

        # Reset the index
        concatenated_df.reset_index(drop=True, inplace=True)
        self.training_dataset = concatenated_df
        self.validation_dataset = concatenated_df.sample(n=50, random_state=42)
        self.test_dataset = concatenated_df.sample(n=25, random_state=30)


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
            self.training_dataset, self.root_dir
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
            self.validation_dataset, self.root_dir
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
            self.test_dataset, self.root_dir
        )
        return DataLoader(
            test_data,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            persistent_workers=True,
        )
    