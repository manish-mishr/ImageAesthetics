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

# resnet = models.resnet18(pretrained=True)

use_gpu = False  #torch.cuda.is_available()


class featureColumn(nn.Module):
    def __init__(self, inChannel, outChannel, init=None):
        super(featureColumn,self).__init__()
        self.linear1 = nn.Linear(inChannel,outChannel, bias=True)
        self.bn1 = nn.BatchNorm1d(256, momentum=0.01)
        self.relu1 = nn.LeakyReLU(0.1,inplace=True)
        self.linear2 = nn.Linear(outChannel,1,bias=True)
        self.init_weights(init)

    def init_weights(self,init):
        self.linear1.weight.data.normal_(0.0, 0.02)
        self.linear1.bias.data.fill_(0)
        self.linear2.weight.data = torch.from_numpy(init['weights']).clone()
        self.linear2.bias.data = torch.from_numpy(init['bias']).clone()

    def forward(self,input):
        out = self.linear1(input)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.linear2(out)
        return out

class ospModel(nn.Module):
    def __init__(self, NUM_CLASSES):
        super(ospModel,self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        num_fltrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_fltrs,NUM_CLASSES)
        # self.num_ft = self.resnet.fc.in_features
        # self.resnet.fc = nn.Linear(self.num_ft,512,bias=True)
        # self.bn1 = nn.BatchNorm1d(512,momentum=0.01)
        # self.initWeights = {}
        # self.getpretrainedWts()
        # self.relu = nn.LeakyReLU(0.1,inplace=True)
        # self.BalancingElement = featureColumn(512,256,self.initWeights['BalancingElement'])
        # self.ColorHarmony = featureColumn(512,256,self.initWeights['ColorHarmony'])
        # self.Content = featureColumn(512, 256, self.initWeights['Content'])
        # self.DoF = featureColumn(512, 256, self.initWeights['DoF'])
        # self.Light = featureColumn(512, 256, self.initWeights['Light'])
        # self.MotionBlur = featureColumn(512, 256, self.initWeights['MotionBlur'])
        # self.Object = featureColumn(512, 256, self.initWeights['Object'])
        # self.Repetition = featureColumn(512, 256, self.initWeights['Repetition'])
        # self.RuleOfThirds = featureColumn(512, 256, self.initWeights['RuleOfThirds'])
        # self.Symmetry = featureColumn(512, 256, self.initWeights['Symmetry'])
        # self.VividColor = featureColumn(512, 256, self.initWeights['VividColor'])

    def forward(self,x):
        out = self.resnet(x)
        # out = self.bn1(out)
        # out = self.relu(out)
        # out1 = self.BalancingElement(out)
        # out2 = self.ColorHarmony(out)
        # out3 = self.Content(out)
        # out4 = self.DoF(out)
        # out5 = self.Light(out)
        # out6 = self.MotionBlur(out)
        # out7 = self.Object(out)
        # out8 = self.Repetition(out)
        # out9 = self.RuleOfThirds(out)
        # out10 = self.Symmetry(out)
        # out11 = self.VividColor(out)
        # output = torch.cat([out1, out2, out3, out4, out5, out6,
        #                     out7, out8, out9, out10, out11],1)

        return out



    def getpretrainedWts(self):
        layers = ['BalancingElement', 'ColorHarmony', 'Content', 'DoF', 'Light', 'MotionBlur',
                  'Object', 'Repetition', 'RuleOfThirds', 'Symmetry', 'VividColor']

        for layer in layers:
            wts = 'fc9_' + layer
            net_weights = np.load(output_path + '/' + wts + '_0.npy')
            # net_weights = np.transpose(net_weights, (1, 0))
            net_bias = np.load(output_path + '/' + wts + '_1.npy')
            self.initWeights[layer] = {}
            self.initWeights[layer]['weights'] = net_weights
            self.initWeights[layer]['bias'] = net_bias



def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    global use_gpu
    since = time.time()


    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'validation']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0

            cnt =  0
            # Iterate ovmodules = list(resnet.children())[:-1]er data.
            for data in dataloaders[phase]:

                cnt += 1
                print cnt
                # get the inputs
                inputs, labels = data
                print inputs.size()
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                print inputs.size()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)

                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0] * inputs.size(0)


            epoch_loss = running_loss / dataset_sizes[phase]


            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    print use_gpu
    model_osp = ospModel(2)
    if use_gpu:
        model_osp = model_osp.cuda()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model_osp.parameters())
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer,7)
    best_model = train_model(model_osp,criterion,optimizer,exp_lr_scheduler)