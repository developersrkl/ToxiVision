import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as T
import pandas as pd
from sklearn.metrics import roc_auc_score
import torchvision.models as models
from downstream import create_multi_label_dataset, label_cols
class SimCLRPretrainedResNet18Eval(models.resnet.ResNet):
    def __init__(self, num_labels=12):
        super().__init__(models.resnet.BasicBlock, [2, 2, 2, 2])
        num_ftrs = self.fc.in_features
        self.fc = torch.nn.Linear(num_ftrs, num_labels)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_eval = SimCLRPretrainedResNet18Eval(num_labels=len(label_cols)).to(device)
model_eval.load_state_dict(torch.load("models/simclr_tox21_downstream_model.pth", map_location=device))
model_eval.eval()
transform_eval = T.Compose([
    T.Resize((200,200)),
    T.ToTensor(),
    T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
test_dataset = create_multi_label_dataset("csv_bin/test_labels.csv", label_cols, transform_eval)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model_eval(images)
        probs = torch.sigmoid(outputs).cpu().numpy()
        all_preds.append(probs)
        all_labels.append(labels.numpy())
all_preds = np.concatenate(all_preds, axis=0)
all_labels = np.concatenate(all_labels, axis=0)
roc_aucs = []
for i, col in enumerate(label_cols):
    valid_indices = (all_labels[:, i] != -1)
    true_vals = all_labels[valid_indices, i]
    pred_vals = all_preds[valid_indices, i]
    if len(np.unique(true_vals)) > 1:
        auc_score = roc_auc_score(true_vals, pred_vals)
        roc_aucs.append(auc_score)
    else:
        roc_aucs.append(None)
for col, val in zip(label_cols, roc_aucs):
    pass
valid_aucs = [v for v in roc_aucs if v is not None]
if len(valid_aucs) > 0:
    pass
