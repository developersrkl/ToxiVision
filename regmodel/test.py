# Author: Shaurya K, Rutgers NB

import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as T
from sklearn.metrics import roc_auc_score

from create_dataset import create_multi_label_dataset
from model import get_resnet18_multilabel

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label_cols = [
        'NR-AR','NR-AR-LBD','NR-AhR','NR-Aromatase','NR-ER',
        'NR-ER-LBD','NR-PPAR-gamma','SR-ARE','SR-ATAD5',
        'SR-HSE','SR-MMP','SR-p53'
    ]

    transform = T.Compose([
        T.Resize((200,200)),
        T.ToTensor(),
        T.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ])

    test_dataset = create_multi_label_dataset("csv_bin/test_labels.csv", label_cols, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Load the trained model (random initialized weights)
    model = get_resnet18_multilabel(num_labels=len(label_cols), pretrained=False)
    model.load_state_dict(torch.load("tox21_multilabel_resnet18.pth", map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    all_preds = np.concatenate(all_preds, axis=0)   # shape [N, 12]
    all_labels = np.concatenate(all_labels, axis=0)  

    roc_aucs = []
    for i in range(len(label_cols)):
        valid_indices = (all_labels[:, i] != -1)
        valid_labels = all_labels[valid_indices, i]
        valid_preds = all_preds[valid_indices, i]

        if len(np.unique(valid_labels)) > 1:
            auc = roc_auc_score(valid_labels, valid_preds)
            roc_aucs.append(auc)
        else:
            roc_aucs.append(None)

    # Print results
    for i, auc in enumerate(roc_aucs):
        name = label_cols[i]
        if auc is None:
            print(f"{name} Error: insufficient data")
        else:
            print(f"{name} Accuracy: {auc:.4f}")

    valid_aucs = [x for x in roc_aucs if x is not None]
    if len(valid_aucs) > 0:
        print(f"Mean Accuracy (computed over all labels): {np.mean(valid_aucs):.4f}")
    else:
        print("Error: insufficient data")

if __name__ == "__main__":
    main()
