# models/RegNetX400MF.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
try:
    # Newer torchvision (0.13+)
    from torchvision.models import RegNet_X_400MF_Weights
    _HAS_WEIGHTS_ENUM = True
except ImportError:
    # Older torchvision
    RegNet_X_400MF_Weights = None
    _HAS_WEIGHTS_ENUM = False


class RegNetX400MF(nn.Module):
    """
    RegNetX-400MF backbone adapted for face recognition.

    - Uses torchvision.models.regnet_x_400mf as the trunk
    - Removes the original classification head
    - Adds a 512-d embedding head (Dropout + Linear + BN)
    - Returns L2-normalized embeddings
    """
    def __init__(self, nOut=512, pretrained=False, **kwargs):
        super().__init__()

        # --- Load RegNetX-400MF backbone ---
        if _HAS_WEIGHTS_ENUM:
            if pretrained:
                weights = RegNet_X_400MF_Weights.IMAGENET1K_V2
            else:
                weights = None
            backbone = models.regnet_x_400mf(weights=weights)
        else:
            # Fallback for older torchvision (pre-weights API)
            backbone = models.regnet_x_400mf(pretrained=pretrained)

        # Keep stem + trunk as feature extractor
        self.stem = backbone.stem          # initial conv + BN + act
        self.trunk = backbone.trunk_output # main RegNet stages
        self.avgpool = backbone.avgpool    # global avg pool
        
        in_features = backbone.fc.in_features  # 440 for 400MF

        # --- Embedding head ---
        self.embedding = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, nOut, bias=False),
            nn.BatchNorm1d(nOut, eps=1e-5)
        )

        self._initialize_head()

    def _initialize_head(self):
        # Initialize only our embedding head; backbone uses torchvision init.
        for m in self.embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        # x: (B, 3, 224, 224)
        x = self.stem(x)
        x = self.trunk(x)
        x = self.avgpool(x)              # (B, C, 1, 1)
        x = torch.flatten(x, 1)          # (B, C)
        x = self.embedding(x)            # (B, nOut)
        x = F.normalize(x, p=2, dim=1)   # L2-normalized embedding
        return x


def MainModel(nOut=512, **kwargs):
    """
    Factory for compatibility with EmbedNet:
    importlib.import_module('models.'+model).__getattribute__('MainModel')
    """
    return RegNetX400MF(nOut=nOut, **kwargs)
