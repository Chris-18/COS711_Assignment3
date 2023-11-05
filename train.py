import csv

import pandas as pd
import pytorch_lightning as pl
import ray
import torch
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from torch.utils.data import DataLoader

import config as c
from dataModule import CropDamageDataModule
from logistics_regression.logistics_regression_model import LogisticsRegressionModel
from logistics_regression.logistics_regression_module import LogisticsRegressionModule
from model import CropDamageModel
from test_dataset import TestDataset
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.train import CheckpointConfig
import numpy as np


def train_crop_model(config, tuning, model_type, epochs, seed):
    if model_type == "regression":
        data_module = CropDamageDataModule(
            batch_size=config["batch_size"], seed=seed, tuning=tuning
        )
        model = CropDamageModel(config)
    elif model_type == "logistic":
        data_module = LogisticsRegressionModule(
            batch_size=config["batch_size"], seed=seed, tuning=tuning
        )
        model = LogisticsRegressionModel(config)
    else:
        raise Exception("Invalid model type.")

    metrics = {"val_loss": "val_loss"}
    if tuning:
        callbacks = [TuneReportCallback(metrics, on="validation_end")]
    else:
        callbacks = []

    trainer = pl.Trainer(
        min_epochs=1,
        max_epochs=epochs,
        log_every_n_steps=50,
        callbacks=callbacks,
    )
    trainer.fit(model, data_module)
    if tuning:
        if model_type == "regression":
            trainer.save_checkpoint("best_regression_model.ckpt")
        else:
            trainer.save_checkpoint("best_logistics_model.ckpt")

        train_loss = trainer.callback_metrics["train_loss"]
        val_loss = trainer.callback_metrics["val_loss"]
        print(f"Train_loss: {train_loss}")
        print(f"Val_loss: {val_loss}")

        return {"seed": seed, "train_loss": train_loss, "val_loss": val_loss}


def tune_crop_model(model_type, test_name):
    ray.init(local_mode=False)  # You can configure Ray according to your needs
    trainable = tune.with_parameters(
        train_crop_model,
        tuning=True,
        model_type=model_type,
        epochs=c.TUNING_NUM_EPOCHS,
        seed=42,
    )
    analysis = tune.run(
        trainable,
        config=c.SEARCH_SPACE,
        num_samples=c.TUNING_NUM_SAMPLES,  # Number of trials
        scheduler=ASHAScheduler(
            metric="val_loss", mode="min", max_t=c.TUNING_NUM_EPOCHS
        ),
        # local_dir="./ray_tune_logs",  # Directory to store logs and checkpoints
        checkpoint_config=CheckpointConfig(num_to_keep=2),
        name=f"crop_damage_{model_type}_hyperparameter_tuning",
        resources_per_trial={
            "cpu": 3,
            "gpu": 0,
        },  # Adjust based on your available resources
        reuse_actors=True,
    )

    best_trial = analysis.get_best_trial("val_loss", "min", "last")
    print(f"Best {model_type} trial config: {best_trial.config}")
    print(
        f"Best {model_type} trial final validation loss: {best_trial.last_result['val_loss']}"
    )

    # Save the results to a text file
    with open(f"{model_type}_{test_name}_tuning_results.txt", "w") as file:
        file.write(f"Best {model_type} trial config: {best_trial.config}\n")
        file.write(
            f"Best {model_type} trial final validation loss: {best_trial.last_result['val_loss']}\n"
        )
        file.write("All trial results:\n")
        for trial in analysis.trials:
            file.write(
                f"Trial {trial.trial_id}: \nConfig: {trial.config}\nValidation Loss = {trial.last_result['val_loss']}\n\n"
            )

    return analysis


if __name__ == "__main__":
    csv_path = "data/content/Train.csv"  # Path to your training CSV file
    test_csv_path = "data/Test.csv"  # Path to your test CSV file
    root_dir = "data/content/train"  # Root directory where your images are stored
    test_root_dir = (
        "data/content/test"  # Root directory where your test images are stored
    )

    run_type = "train"
    model = "logistic"

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
        seeds = [42, 54, 6, 8, 4]
        results = []
        for seed in seeds:
            if model == "logistic":
                result = train_crop_model(
                    config=c.LR_DEFAULT_CONFIG,
                    tuning=False,
                    model_type="logistic",
                    epochs=c.NUM_EPOCHS,
                    seed=seed,
                )
            elif model == "regression":
                result = train_crop_model(
                    config=c.R_DEFAULT_CONFIG,
                    tuning=False,
                    model_type="regression",
                    epochs=c.NUM_EPOCHS,
                    seed=seed,
                )
            else:
                raise Exception("Invalid model type given")

            results.append(result)

            # Extract train_loss and val_loss values into separate lists
            train_loss_values = [item["train_loss"] for item in results]
            val_loss_values = [item["val_loss"] for item in results]

            # Calculate the means and standard deviations for train_loss and val_loss
            train_loss_mean = np.mean(train_loss_values)
            val_loss_mean = np.mean(val_loss_values)
            train_loss_std = np.std(train_loss_values)
            val_loss_std = np.std(val_loss_values)

            with open(f"{model}_training_results.txt", "w") as file:
                for r in results:
                    file.write(
                        f"Seed: {r['seed']}\nTraining_loss: {r['train_loss']}\nValidation_loss: {r['val_loss']}\n\n"
                    )
                # write the results
                file.write(f"Mean for Train Loss: {train_loss_mean}\n")
                file.write(f"Standard Deviation for Train Loss: {train_loss_std}\n")
                file.write(f"Mean for Validation Loss: {val_loss_mean}\n")
                file.write(f"Standard Deviation for Validation Loss: {val_loss_std}\n")

    if run_type == "tune":
        tune_crop_model(model_type=model, test_name="optimizer")

    # make predictions
    if run_type == "predict" and model == "logistic":
        data_module = LogisticsRegressionModule(batch_size=c.LR_BATCH_SIZE)
        count = 0
        model = LogisticsRegressionModel.load_from_checkpoint(
            "best_logistics_model.ckpt"
        )
        model.eval()
        correct = 0
        incorrect = 0

        with torch.no_grad():
            for batch in data_module.test_dataloader():
                x, y = batch
                pred = model.forward(x)
                pred = pred.squeeze()
                pred_binary = (pred > 0.5).int()

                # Compare predicted values with ground truth
                num_correct = (pred_binary == y).sum().item()
                num_incorrect = len(y) - num_correct

                correct += num_correct
                incorrect += num_incorrect
        print(f"Number of correct predictions: {correct}")
        print(f"Number of incorrect predictions: {incorrect}")

    if run_type == "predict" and model == "regression":
        data_module = CropDamageDataModule(batch_size=c.R_BATCH_SIZE)
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
            test_dataset, batch_size=c.R_BATCH_SIZE, shuffle=False
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
