import torchvision
import torch.nn as nn

class MobileNetV2(nn.Module):
    def __init__(self, nOut=512, pretrained=True):
        super(MobileNetV2, self).__init__()
        self.model = torchvision.models.mobilenet_v2(pretrained=pretrained)
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.model.last_channel, nOut),
        )

    def forward(self, x):
        return self.model(x)
    
def MainModel(nOut=512, pretrained=True):
    return MobileNetV2(nOut=nOut, pretrained=pretrained)
