import torch
import torch.nn as nn
import torchvision.models as models


class FeatureSE(nn.Module):
    """
    Lightweight SE-style attention over the final 512-D feature vector.
    Reweights channels with a small MLP.
    """
    def __init__(self, channels=512, reduction=16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C)
        w = self.fc(x)   # (B, C)
        return x * w     # channel-wise reweighting


class ImprovedResNet(nn.Module):
    def __init__(self, nOut=512, pretrained=False, **kwargs):
        super(ImprovedResNet, self).__init__()
        
        # Base ResNet18 - RANDOM init (no external pretraining)
        backbone = models.resnet18(pretrained=pretrained)
        
        # Remove the final fully connected layer (keep convs + avgpool)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        # features(x) -> (B, 512, 1, 1)

        # Feature-level SE attention on 512-D vector
        self.attention = FeatureSE(channels=512, reduction=16)

        # Add custom embedding layer (goes into AdaFace)
        self.embedding = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(512, nOut, bias=False),
            nn.BatchNorm1d(nOut),
        )
        
        # Weight initialization for embedding head
        for module in self.embedding.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
                
    def forward(self, x):
        # Backbone features
        x = self.features(x)            # (B, 512, 1, 1)
        x = x.view(x.size(0), -1)       # (B, 512)

        # Channel attention on global feature
        x = self.attention(x)           # (B, 512)

        # Embedding head (unnormalized; AdaFace will use norms)
        x = self.embedding(x)           # (B, nOut)
        return x


def MainModel(nOut=512, **kwargs):
    # Make sure we NEVER accidentally use pretrained=True 
    kwargs.pop("pretrained", None)
    return ImprovedResNet(nOut=nOut, pretrained=False, **kwargs)
