# Author: Shaurya K, Rutgers NB

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
import numpy as np

from create_dataset import create_multi_label_dataset
from model import get_resnet18_multilabel, masked_loss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label_cols = [
        'NR-AR','NR-AR-LBD','NR-AhR','NR-Aromatase','NR-ER',
        'NR-ER-LBD','NR-PPAR-gamma','SR-ARE','SR-ATAD5','SR-HSE','SR-MMP','SR-p53'
    ]

    transform = T.Compose([
        T.Resize((200,200)),
        T.ToTensor(),
        T.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ])

    train_dataset = create_multi_label_dataset("csv_bin/train_labels.csv", label_cols, transform=transform)
    val_dataset = create_multi_label_dataset("csv_bin/val_labels.csv", label_cols, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = get_resnet18_multilabel(num_labels=len(label_cols), pretrained=True).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    epochs = 20
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            outputs = torch.clamp(outputs, min=-10, max=10)

            loss = masked_loss(outputs, labels)
            optimizer.zero_grad()
            loss.backward()

            # gradient clipping for Grad-CAM
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = masked_loss(outputs, labels)
                val_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, "
              f"Train Loss: {running_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss/len(val_loader):.4f}")

    torch.save(model.state_dict(), "models/tox21_multilabel_resnet18.pth")
    print("Model trained and saved")

if __name__ == "__main__":
    main()
