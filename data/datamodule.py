import os
from typing import Optional
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import pytorch_lightning as pl

import os
import json
from torch.utils.data import Dataset
from PIL import Image

import os
import json
from torch.utils.data import Dataset
from PIL import Image

class SyntheticImageDataset(Dataset):
    def __init__(self, data_dir, transform=None, attributes_to_idx_file='data/attributes_to_idx.json'):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith('.png')]
        
        # Load attributes_to_idx mapping
        with open(attributes_to_idx_file, 'r') as f:
            self.attributes_to_idx = json.load(f)
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Extract attributes from the filename
        parts = os.path.splitext(img_name)[0].split('_')
        # Remove the initial index (first part)
        attributes = parts[1:]  # Excludes edge_color
        attributes_key = '_'.join(attributes)
        
        # Map attributes to class index
        class_idx = self.attributes_to_idx.get(attributes_key)
        if class_idx is None:
            raise ValueError(f"Attributes {attributes} not found in attributes_to_idx mapping.")
        
        return image, class_idx



class SyntheticImageDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32, num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Define transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Using ImageNet means and stds
                                 std=[0.229, 0.224, 0.225])
        ])

    def setup(self, stage: Optional[str] = None):
        # Load the full dataset
        full_dataset = SyntheticImageDataset(self.data_dir, transform=self.transform)
        
        # Calculate lengths for train, val, and test
        total_size = len(full_dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        test_size = total_size - train_size - val_size

        # Split the dataset
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size,
            num_workers=self.num_workers
        )


# Example usage
if __name__ == "__main__":
    data_dir = "enhanced_synthetic_dataset/"
    datamodule = SyntheticImageDataModule(data_dir)
    datamodule.setup()

    # Print some information about the datasets
    print(f"Total images: {len(datamodule.train_dataset) + len(datamodule.val_dataset) + len(datamodule.test_dataset)}")
    print(f"Training images: {len(datamodule.train_dataset)}")
    print(f"Validation images: {len(datamodule.val_dataset)}")
    print(f"Test images: {len(datamodule.test_dataset)}")

    # Test the dataloaders
    train_loader = datamodule.train_dataloader()
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")

    data_dir = "enhanced_synthetic_dataset/"
    dataset = SyntheticImageDataset(data_dir)
    labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        labels.append(label)
    labels = torch.tensor(labels)
    print(f"Labels min: {labels.min()}, max: {labels.max()}")
