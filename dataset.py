import torch
from torch.utils.data import Dataset
from torchvision import transforms
from image_loader import load_images_dataset

class CustomDataset(Dataset):
    def __init__(self):
        self.data = load_images_dataset()
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        if not isinstance(image, torch.Tensor):
            image = transforms.ToTensor()(image)
        image = self.transform(image)
        return image, label
