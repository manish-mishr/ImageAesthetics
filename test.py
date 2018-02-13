import torch

import torch.nn as nn
import torchvision.models as models

import torch.optim as optim
from torch.autograd import Variable

import copy
import time
import numpy as np
import dataLoader

######## data loader #####################
output_path = '/home/manish/projects/ResNetModel/modelWeight'
dataloaders, dataset_sizes = dataLoader.getdataLoader()
##############################################

alexnet = models.alexnet(pretrained=True)


params = alexnet.state_dict()
for key,val in params.items():
    print key




