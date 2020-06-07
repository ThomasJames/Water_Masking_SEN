import torch
import numpy
from torch import nn
import torchvision.models as models

vgg16 = models.vgg16(pretrained=True)

print(vgg16)

vgg16.classifier = "Needs to be changed"