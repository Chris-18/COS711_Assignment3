import pytorch_lightning as pl
from model import CropDamageModel
from dataModule import CropDamageDataModule
import config
import torch


if __name__ == "__main__":
    csv_path = "data/content/Train.csv"  # Path to your training CSV file
    test_csv_path = "test.csv"  # Path to your test CSV file
    root_dir = "data/content/train"  # Root directory where your images are stored
    batch_size = 32
    num_epochs = 10

    model = CropDamageModel(num_classes=config.NUM_CLASSES)
    data_module = CropDamageDataModule(csv_path, root_dir)
    trainer = pl.Trainer(
        min_epochs=1, max_epochs=num_epochs, log_every_n_steps=50, accelerator="gpu"
    )

    trainer.fit(model, data_module)
    # trainer.validate(model, data_module)
    # trainer.test(model, data_module)

    # make predictions
    count = 0
    model.eval()
    for batch in data_module.test_dataloader():
        x, y = batch
        pred = model.forward(x, y)
        # Apply softmax to convert pred to class probabilities
        probabilities = torch.softmax(pred, dim=1)

        # Get the class with the highest probability as the predicted class
        _, predicted_class = torch.max(probabilities, 1)

        results = predicted_class * 10
        print(f"Expected value: {y}\nPredicted value: {results}")
        print("")
        count = count + 1
        if count == 2:
            break


# To make predictions on the test dataset
# model.eval()
# predictions = []
# with torch.no_grad():
#     for batch in test_loader:
#         x, _ = batch
#         logits = model(x)
#         pred = torch.argmax(logits, dim=1)
#         predictions.extend(pred.cpu().numpy())
#
# return predictions
