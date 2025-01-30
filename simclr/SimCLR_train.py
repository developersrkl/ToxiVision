import torch
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.optim as optim
from PIL import Image
import pandas as pd
from SimCLR_models import SimCLRModel, simclr_loss_function
from torchvision import transforms
import numpy as np

class SimCLRDataset:
    def __init__(self, csv_file, transform=None, simclr_transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.simclr_transform = simclr_transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["img_path"]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.simclr_transform:
            x1 = self.simclr_transform(image)
            x2 = self.simclr_transform(image)
        else:
            x1 = image
            x2 = image
        return x1, x2

csv_file = "csv_bin/train_labels.csv"
simclr_transform = T.Compose([
    T.RandomResizedCrop(200, scale=(0.5, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandomApply([T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2)], p=0.8),
    T.RandomGrayscale(p=0.2),
    T.ToTensor(),
    T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

simclr_dataset = SimCLRDataset(csv_file=csv_file, transform=None, simclr_transform=simclr_transform)
from torch.utils.data import Dataset

class _SimCLRDatasetWrapper(Dataset):
    def __init__(self, simclr_dataset):
        self.simclr_dataset = simclr_dataset
    def __len__(self):
        return len(self.simclr_dataset)
    def __getitem__(self, idx):
        return self.simclr_dataset[idx]

wrapped_dataset = _SimCLRDatasetWrapper(simclr_dataset)
simclr_train_loader = DataLoader(wrapped_dataset, batch_size=32, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_encoder = models.resnet18(pretrained=False)
simclr_model = SimCLRModel(base_encoder=base_encoder, projection_dim=128).to(device)
optimizer = optim.Adam(simclr_model.parameters(), lr=1e-3)
epochs = 5
temperature = 0.07
for epoch in range(epochs):
    simclr_model.train()
    running_loss = 0.0
    for (x1, x2) in simclr_train_loader:
        x1, x2 = x1.to(device), x2.to(device)
        z1 = simclr_model(x1)
        z2 = simclr_model(x2)
        loss = simclr_loss_function(z1, z2, temperature=temperature)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(simclr_train_loader)
torch.save(simclr_model.encoder.state_dict(), "models/simclr_resnet18_encoder.pth")
