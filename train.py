import pytorch_lightning as pl
from model import CropDamageModel
from dataModule import CropDamageDataModule
import config


if __name__ == "__main__":
    csv_path = "data/content/train.csv"  # Path to your training CSV file
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
    trainer.test(model, data_module)


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
