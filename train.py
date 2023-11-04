import csv
import pandas as pd
import pytorch_lightning as pl
from dataset import CropDamageDataset
from model import CropDamageModel
from dataModule import CropDamageDataModule
import config
from torch.utils.data import DataLoader
import torch

from test_dataset import TestDataset


if __name__ == "__main__":
    csv_path = "data/content/Train.csv"  # Path to your training CSV file
    test_csv_path = "data/Test.csv"  # Path to your test CSV file
    root_dir = "data/content/train"  # Root directory where your images are stored
    test_root_dir ="data/content/test" # Root directory where your test images are stored
    batch_size = config.BATCH_SIZE
    num_epochs = config.NUM_EPOCHS
    run_type = "csv"

    if(run_type=='csv'):
        # Open the input CSV file for reading
        input_file = 'labels_and_extents.csv'
        output_file = 'output.csv'
        with open(input_file, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            
            # Define the header for the output CSV file
            fieldnames = reader.fieldnames

            # Open the output CSV file for writing
            with open(output_file, 'w', newline='') as output_csvfile:
                writer = csv.DictWriter(output_csvfile, fieldnames=fieldnames)
                writer.writeheader()

                # Loop through the rows in the input CSV file
                for row in reader:
                    extent = float(row['extent'])
                    
                    # Check if the extent is less than 5, and set it to 0 if it is
                    if extent < 10:
                        row['extent'] = 0
                    
                    # Write the modified row to the output CSV file
                    writer.writerow(row)

        print("CSV file processed and saved as 'output.csv'")


    data_module = CropDamageDataModule(csv_path, root_dir, test_csv_path, test_root_dir)

    if run_type == "train":
        model = CropDamageModel()
        trainer = pl.Trainer(min_epochs=1, max_epochs=num_epochs, log_every_n_steps=50)
        trainer.fit(model, data_module)
        # trainer.validate(model, data_module)
        # trainer.test(model, data_module)
        trainer.save_checkpoint("best_model.ckpt")
    # make predictions
    if run_type=='predict':
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
        
    if run_type=='test':
        count = 0
        
        model = CropDamageModel.load_from_checkpoint("best_model.ckpt")
        model.eval()
        predictions = []
        full_dataset = pd.read_csv(test_csv_path)
        test_dataset = TestDataset(full_dataset, test_root_dir)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        data = []
        with torch.no_grad():
            for batch in test_dataloader:
                x, label = batch
                pred = model.forward(x)
                pred = pred.squeeze()
                predictions.extend(pred.cpu().numpy())
                combined_data = (zip(label, pred))
                data.extend([{'ID': label, 'extent': value.item()} for label, value in combined_data])
                print(data)

        # Define the CSV file name
        csv_file = 'labels_and_extents.csv'

        # Write the data to a CSV file
        with open(csv_file, 'w', newline='') as csvfile:
            fieldnames = ['ID', 'extent']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for row in data:
                writer.writerow(row)

        print(f'CSV file "{csv_file}" has been created with ID and extent columns.')
        
        print(predictions)
        


   
