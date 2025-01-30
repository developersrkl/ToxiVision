# Author: Shaurya K, Rutgers NB

import torch
import torch.nn as nn
import torchvision.models as models

def get_resnet18_multilabel(num_labels=12, pretrained=True):
    model = models.resnet18(pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_labels)
    return model

def masked_loss(outputs, targets, mask_value=-1):
    mask = (targets != mask_value)
    outputs = outputs[mask]
    targets = targets[mask]
    return nn.BCEWithLogitsLoss()(outputs, targets)