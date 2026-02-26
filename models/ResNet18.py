#! /usr/bin/python
# -*- encoding: utf-8 -*-
# project/models/ResNet18.py
import torchvision

def MainModel(nOut=256, **kwargs):
    
    return torchvision.models.resnet18(num_classes=nOut)
