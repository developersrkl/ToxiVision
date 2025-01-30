import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def get_resnet18_multilabel(num_labels=12, pretrained=True):
    model = models.resnet18(pretrained=pretrained)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_labels)
    return model

class SimCLRDataset(nn.Module):
    pass

class ProjectionHead(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=256, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x):
        return self.net(x)

def simclr_loss_function(z_i, z_j, temperature=0.07):
    batch_size = z_i.size(0)
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)
    similarity_matrix = torch.matmul(z_i, z_j.t())
    similarity_matrix = similarity_matrix / temperature
    labels = torch.arange(batch_size).to(z_i.device)
    loss_i = nn.CrossEntropyLoss()(similarity_matrix, labels)
    loss_j = nn.CrossEntropyLoss()(similarity_matrix.t(), labels)
    loss = (loss_i + loss_j) / 2.0
    return loss

class SimCLRModel(nn.Module):
    def __init__(self, base_encoder=None, projection_dim=128):
        super().__init__()
        if base_encoder is None:
            base_encoder = models.resnet18(pretrained=False)
        num_ftrs = base_encoder.fc.in_features
        self.encoder = nn.Sequential(*list(base_encoder.children())[:-1])
        self.projection_head = ProjectionHead(in_dim=num_ftrs, out_dim=projection_dim)
    def forward(self, x):
        feat = self.encoder(x)
        feat = feat.view(feat.size(0), -1)
        z = self.projection_head(feat)
        return z
