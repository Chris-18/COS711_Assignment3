import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class TestDataset(Dataset):
    def __init__(self, data, root_dir):
        self.data = data
        self.root_dir = root_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_name = self.data.iloc[idx, 1]
        img_name = os.path.join(self.root_dir, file_name)
        image = Image.open(img_name)
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),

            ]
        )

        id = self.data.iloc[idx, 0]
        image = transform(image)

        return image, id
