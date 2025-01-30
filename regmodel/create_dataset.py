# Author: Shaurya K, Rutgers NB

import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

class create_multi_label_dataset(Dataset):

    def __init__(self, csv_file, label_cols, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.label_cols = label_cols

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
