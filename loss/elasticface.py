# elasticface.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class LossFunction(nn.Module):
    def __init__(self, nClasses, nOut=512, margin=0.3, scale=64, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(nClasses, nOut))
        nn.init.xavier_uniform_(self.weight)
        self.margin = margin
        self.scale = scale
        self.nClasses = nClasses

    def forward(self, x, label):
        x = F.normalize(x, p=2, dim=1)
        w = F.normalize(self.weight, p=2, dim=1)
        cos_theta = F.linear(x, w).clamp(-1+1e-7, 1-1e-7)
        one_hot = F.one_hot(label, self.nClasses).float()
        margin = one_hot * self.margin
        cos_theta_m = cos_theta - margin
        logits = cos_theta_m * self.scale
        return F.cross_entropy(logits, label)
