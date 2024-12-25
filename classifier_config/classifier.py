import torch
import torch.nn as nn

class Multiclass_Classification(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.layer0 = nn.Linear(in_features=num_features, out_features=num_classes)

    def forward(self, x):
        return self.layer0(x)