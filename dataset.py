import os
from torch.utils.data import Dataset
from PIL import Image


class CropDamageDataset(Dataset):
    def __init__(self, data, root_dir, transform=None):
        self.data = data
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_name = self.data.iloc[idx, 1]
        img_name = os.path.join(self.root_dir, file_name)
        image = Image.open(img_name)
        label = int(self.data.iloc[idx, 4])

        if self.transform:
            image = self.transform(image)

        return image, label
