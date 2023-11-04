import csv

import pandas as pd
import pytorch_lightning as pl
import ray
import torch
from ray import tune
from ray.train import CheckpointConfig, RunConfig, ScalingConfig
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from ray.train.torch import TorchTrainer
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from torch.utils.data import DataLoader

import config as c
from dataModule import CropDamageDataModule
from dataset import CropDamageDataset
from model import CropDamageModel
from test_dataset import TestDataset
from ray.tune.integration.pytorch_lightning import TuneReportCallback


def train_crop_model(config, manual=False):
    data_module = CropDamageDataModule(batch_size=config["batch_size"])
    model = CropDamageModel(config)
    metrics = {"val_loss": "val_loss"}
    trainer = pl.Trainer(
        min_epochs=1,
        max_epochs=c.TUNING_NUM_EPOCHS,
        log_every_n_steps=50,
        callbacks=[TuneReportCallback(metrics, on="validation_end")],
    )
    trainer.fit(model, data_module)
    if manual:
        trainer.save_checkpoint("best_model.ckpt")


def tune_crop_model():
    ray.init(local_mode=False)  # You can configure Ray according to your needs

    analysis = tune.run(
        train_crop_model,
        config={
            "batch_size": tune.choice([32, 64]),
            "learning_rate": tune.loguniform(1e-4, 1e-1),
            # Add more hyperparameters to search here
        },
        num_samples=c.TUNING_NUM_SAMPLES,  # Number of trials
        scheduler=ASHAScheduler(
            metric="val_loss", mode="min", max_t=c.TUNING_NUM_EPOCHS
        ),
        # local_dir="./ray_tune_logs",  # Directory to store logs and checkpoints
        name="crop_damage_hyperparameter_tuning",
        resources_per_trial={
            "cpu": 2,
            "gpu": 0,
        },  # Adjust based on your available resources
    )

    best_trial = analysis.get_best_trial("val_loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['val_loss']}")
    return analysis


if __name__ == "__main__":
    csv_path = "data/content/Train.csv"  # Path to your training CSV file
    test_csv_path = "data/Test.csv"  # Path to your test CSV file
    root_dir = "data/content/train"  # Root directory where your images are stored
    test_root_dir = (
        "data/content/test"  # Root directory where your test images are stored
    )
    num_epochs = c.NUM_EPOCHS
    run_type = "tune"

    if run_type == "csv":
        # Open the input CSV file for reading
        input_file = "labels_and_extents.csv"
        output_file = "output.csv"
        with open(input_file, "r", newline="") as csvfile:
            reader = csv.DictReader(csvfile)

            # Define the header for the output CSV file
            fieldnames = reader.fieldnames

            # Open the output CSV file for writing
            with open(output_file, "w", newline="") as output_csvfile:
                writer = csv.DictWriter(output_csvfile, fieldnames=fieldnames)
                writer.writeheader()

                # Loop through the rows in the input CSV file
                for row in reader:
                    extent = float(row["extent"])

                    # Check if the extent is less than 5, and set it to 0 if it is
                    if extent < 10:
                        row["extent"] = 0

                    # Write the modified row to the output CSV file
                    writer.writerow(row)

        print("CSV file processed and saved as 'output.csv'")

    if run_type == "train":
        train_crop_model(config=c.DEFAULT_CONFIG, manual=True)

    if run_type == "tune":
        results = tune_crop_model()
        print(results)
    # make predictions
    if run_type == "predict":
        data_module = CropDamageDataModule(batch_size=c.BATCH_SIZE)
        count = 0
        model = CropDamageModel.load_from_checkpoint("best_model.ckpt")
        model.eval()
        with torch.no_grad():
            for batch in data_module.test_dataloader():
                x, y = batch
                pred = model.forward(x)
                pred = pred.squeeze()
                print(f"Expected value: {y}\nPredicted value: {pred}")
                print("")
                count = count + 1
                if count == 2:
                    break

    if run_type == "test":
        count = 0

        model = CropDamageModel.load_from_checkpoint("best_model.ckpt")
        model.eval()
        predictions = []
        full_dataset = pd.read_csv(test_csv_path)
        test_dataset = TestDataset(full_dataset, test_root_dir)
        test_dataloader = DataLoader(
            test_dataset, batch_size=c.BATCH_SIZE, shuffle=False
        )
        data = []
        with torch.no_grad():
            for batch in test_dataloader:
                x, label = batch
                pred = model.forward(x)
                pred = pred.squeeze()
                predictions.extend(pred.cpu().numpy())
                combined_data = zip(label, pred)
                data.extend(
                    [
                        {"ID": label, "extent": value.item()}
                        for label, value in combined_data
                    ]
                )
                print(data)

        # Define the CSV file name
        csv_file = "labels_and_extents.csv"

        # Write the data to a CSV file
        with open(csv_file, "w", newline="") as csvfile:
            fieldnames = ["ID", "extent"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for row in data:
                writer.writerow(row)

        print(f'CSV file "{csv_file}" has been created with ID and extent columns.')
