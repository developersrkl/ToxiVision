import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import pandas as pd
from torch.utils.data import DataLoader
import torchvision.models as models

import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torch.nn as nn

label_cols = [
    'NR-AR','NR-AR-LBD','NR-AhR','NR-Aromatase','NR-ER',
    'NR-ER-LBD','NR-PPAR-gamma','SR-ARE','SR-ATAD5','SR-HSE','SR-MMP','SR-p53'
]

class create_multi_label_dataset(Dataset):
    def __init__(self, csv_file, label_cols, transform=None):
        self.df = pd.read_csv(csv_file)
        self.label_cols = label_cols
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["img_path"]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        labels = row[self.label_cols].values.astype(float)
        labels_tensor = torch.tensor(labels, dtype=torch.float)
        return image, labels_tensor

def masked_bce_loss(outputs, targets, mask_value=-1):
    mask = (targets != mask_value)
    outputs = outputs[mask]
    targets = targets[mask]
    return nn.BCEWithLogitsLoss()(outputs, targets)

class SimCLRPretrainedResNet18(nn.Module):
    def __init__(self, num_labels=12):
        super().__init__()
        base_model = models.resnet18(pretrained=False)
        num_ftrs = base_model.fc.in_features
        self.encoder = nn.Sequential(*list(base_model.children())[:-1])
        self.fc = nn.Linear(num_ftrs, num_labels)
    def forward(self, x):
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        out = self.fc(features)
        return out
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
downstream_model = SimCLRPretrainedResNet18(num_labels=len(label_cols)).to(device)
pretrained_dict = torch.load("models/simclr_resnet18_encoder.pth", map_location=device)
downstream_model.encoder.load_state_dict(pretrained_dict, strict=False)
transform_downstream = T.Compose([
    T.Resize((200,200)),
    T.ToTensor(),
    T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
train_dataset = create_multi_label_dataset("csv_bin/train_labels.csv", label_cols, transform_downstream)
val_dataset = create_multi_label_dataset("csv_bin/val_labels.csv", label_cols, transform_downstream)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
criterion = masked_bce_loss
optimizer_downstream = optim.Adam(downstream_model.parameters(), lr=1e-3)
num_epochs = 5
for epoch in range(num_epochs):
    downstream_model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = downstream_model(images)
        outputs = torch.clamp(outputs, min=-10, max=10)
        loss = criterion(outputs, labels, mask_value=-1)
        optimizer_downstream.zero_grad()
        loss.backward()
        optimizer_downstream.step()
        running_loss += loss.item()
    train_loss = running_loss / len(train_loader)
    downstream_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = downstream_model(images)
            loss = criterion(outputs, labels, mask_value=-1)
            val_loss += loss.item()
    val_loss /= len(val_loader)
torch.save(downstream_model.state_dict(), "models/simclr_tox21_downstream_model.pth")
