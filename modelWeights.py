import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import numpy as np

resnet = models.resnet18(pretrained=True)
modules = list(resnet.children())[:-1]

print resnet.fc.in_features
# for param in  resnet.parameters():
#     print type(param.data), param.size()