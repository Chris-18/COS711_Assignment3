import pandas as pd
import pytorch_lightning as pl
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from dataset import CropDamageDataset
from model import CropDamageModel


def train_and_test(csv_path, root_dir, test_csv_path, batch_size, num_epochs):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    full_dataset = pd.read_csv(csv_path)
    training_dataset = full_dataset.sample(n=1000, random_state=42)

    train_data = CropDamageDataset(training_dataset, root_dir, transform=transform)
    # test_data = CropDamageDataset(test_csv_path, root_dir, transform=transform)

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=9
    )
    # test_loader = DataLoader(test_data, batch_size=batch_size)

    model = CropDamageModel(num_classes=5)
    trainer = pl.Trainer(max_epochs=num_epochs, log_every_n_steps=50)

    trainer.fit(model, train_loader)

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


if __name__ == "__main__":
    csv_path = "data/content/train.csv"  # Path to your training CSV file
    test_csv_path = "test.csv"  # Path to your test CSV file
    root_dir = "data/content/train"  # Root directory where your images are stored
    batch_size = 32
    num_epochs = 10

    predictions = train_and_test(
        csv_path, root_dir, test_csv_path, batch_size, num_epochs
    )
    print(predictions)
