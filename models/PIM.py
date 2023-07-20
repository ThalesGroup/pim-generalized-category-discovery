import torch
import torch.nn as nn

class PIM_partitioner(nn.Module):
    def __init__(self, num_features=512, num_classes=100, temp=25):
        super().__init__()
        self.partitioner = nn.Linear(num_features, num_classes, bias=False)
        self.temp = temp
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        out = self.partitioner(x) * self.temp
        return out