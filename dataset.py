import os
import random
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class CropDamageDataset(Dataset):
    def __init__(self, data, root_dir):
        self.data = data
        self.root_dir = root_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_name = self.data.iloc[idx, 1]
        img_name = os.path.join(self.root_dir, file_name)
        image = Image.open(img_name)
        label = int(self.data.iloc[idx, 4])
        random_transforms = [
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2), 
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.333)),
        ]

        # Choose a random transformation from the list
        selected_transform = random.choice(random_transforms)

        # Define the complete transformation pipeline
        transform = transforms.Compose([
            selected_transform,
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        image = transform(image)

        return image, label
