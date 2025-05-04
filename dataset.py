import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import glob

class ChestXrayDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def find_image(self, img_name):
        # Try direct path first
        direct_path = os.path.join(self.img_dir, img_name)
        if os.path.exists(direct_path):
            return direct_path
        # Search all subdirectories
        matches = glob.glob(os.path.join(self.img_dir, '**', img_name), recursive=True)
        if matches:
            return matches[0]
        raise FileNotFoundError(f"Image {img_name} not found in {self.img_dir} or its subdirectories.")

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        img_path = self.find_image(img_name)
        image = Image.open(img_path).convert("RGB")
        label = int(self.data.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label)

def get_dataloaders(train_csv, val_csv, test_csv, img_dir, batch_size=32):
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_dataset = ChestXrayDataset(train_csv, img_dir, train_transform)
    val_dataset = ChestXrayDataset(val_csv, img_dir, eval_transform)
    test_dataset = ChestXrayDataset(test_csv, img_dir, eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
