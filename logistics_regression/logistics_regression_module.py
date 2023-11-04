import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, random_split

import config as c
from dataset import CropDamageDataset


class LogisticsRegressionModule(pl.LightningDataModule):
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
        full_dataset["extent"] = (full_dataset["extent"] > 0).astype(int)
        full_dataset = full_dataset.sample(n=c.INPUT_SIZE, random_state=42)

        train_size = c.TRAIN_SIZE
        validation_size = c.VALIDATION_SIZE
        test_size = c.TEST_SIZE

        training_dataset, validation_dataset, test_dataset = random_split(
            full_dataset, [train_size, validation_size, test_size]
        )

        self.training_dataset = training_dataset.dataset.iloc[training_dataset.indices]

        zero_dataset = self.training_dataset[self.training_dataset["extent"] == 0]
        if self.is_tuning:
            zero_dataset = zero_dataset.sample(n=2000, random_state=42)

        one_dataset = self.training_dataset[self.training_dataset["extent"] == 1]

        # Count the number of elements with extent 0/1
        count_extent_0 = len(zero_dataset)
        count_extent_1 = len(one_dataset)

        print("Count of extent 0:", count_extent_0)
        print("Count of extent 1:", count_extent_1)

        extended_datasets = []
        if self.is_tuning:
            if count_extent_1 < 2000:
                extended_dataset = self.extend_dataframe(
                    training_dataset.dataset[training_dataset.dataset["extent"] == 1],
                    count_extent_0 - count_extent_1,
                )
                extended_datasets.append(extended_dataset)
                extended_datasets.append(
                    self.training_dataset[self.training_dataset["extent"] == 0]
                )
            else:
                one_dataset = one_dataset.sample(n=2000, random_state=42)
                extended_datasets.append(one_dataset)
                extended_datasets.append(zero_dataset)
        else:
            extended_dataset = self.extend_dataframe(
                training_dataset.dataset[training_dataset.dataset["extent"] == 1],
                count_extent_0 - count_extent_1,
            )
            extended_datasets.append(extended_dataset)
            extended_datasets.append(
                self.training_dataset[self.training_dataset["extent"] == 0]
            )

        concatenated_df = pd.concat(extended_datasets, axis=0)
        concatenated_df.reset_index(drop=True, inplace=True)
        self.training_dataset = concatenated_df
        self.validation_dataset = validation_dataset.dataset.iloc[
            validation_dataset.indices
        ]
        self.test_dataset = test_dataset.dataset.iloc[test_dataset.indices]

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
