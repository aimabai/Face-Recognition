import torchvision

def MainModel(nOut=256, **kwargs):
    
    return torchvision.models.resnet50(num_classes=nOut)