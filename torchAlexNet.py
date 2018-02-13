import torch



import torch.nn as nn

import torch.optim as optim
from torch.autograd import Variable

import copy
import time
import numpy as np
import dataLoader
import getAlexNetWt as alex


######## data loader #####################
output_path = '/home/manish/projects/ResNetModel/modelWeight'
dataloaders, dataset_sizes = dataLoader.getdataLoader()
##############################################

# resnet = models.resnet18(pretrained=True)

use_gpu = False #torch.cuda.is_available()


def getWeights(weight, group=1):
    if weight.ndim == 4:
        wts = np.transpose(weight, (1, 0, 2, 3))
    else:
        wts = np.transpose(weight,(1,0))
    wts = torch.from_numpy(wts)
    if group == 2:
        wts = torch.cat([wts, wts], 0)
    return wts




class featureColumn(nn.Module):
    def __init__(self, inChannel, outChannel, init, name):
        super(featureColumn,self).__init__()
        self.name = name
        self.linear1 = nn.Linear(inChannel,outChannel, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(outChannel,1,bias=True)
        self.init_weights(init)

    def init_weights(self,init):
        layer = 'fc{0}_{1}_{2}'.format
        wts = layer(8,self.name,0)
        bias = layer(8,self.name,1)
        self.linear1.weight.data.copy_(torch.from_numpy(init[wts]))
        self.linear1.bias.data.copy_(torch.from_numpy(init[bias]))

        wts = layer(9, self.name, 0)
        bias = layer(9, self.name, 1)
        self.linear2.weight.data.copy_(torch.from_numpy(init[wts]))
        self.linear2.bias.data.copy_(torch.from_numpy(init[bias]))

    def forward(self,input):
        out = self.linear1(input)
        out = self.relu1(out)
        out1 = self.linear2(out)
        return out1,out






class AlexNet(nn.Module):

    def __init__(self, init):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2)
        self.relu =   nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 =  nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.conv3 =  nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)

        self.fc6 = nn.Linear(256 * 6 * 6, 4096,bias=True)
        self.fc7 = nn.Linear(4096, 4096, bias=True)
        self.fc8 = nn.Linear(4096,512, bias=True)

        self.BalancingElement = featureColumn(4096, 256, init, 'BalancingElement')
        self.ColorHarmony = featureColumn(4096, 256, init, 'ColorHarmony')
        self.Content = featureColumn(4096, 256, init, 'Content')
        self.DoF = featureColumn(4096, 256, init, 'DoF')
        self.Light = featureColumn(4096, 256, init, 'Light')
        self.MotionBlur = featureColumn(4096, 256, init, 'MotionBlur')
        self.Object = featureColumn(4096, 256, init, 'Object')
        self.Repetition = featureColumn(4096, 256, init, 'Repetition')
        self.RuleOfThirds = featureColumn(4096, 256, init, 'RuleOfThirds')
        self.Symmetry = featureColumn(4096, 256, init, 'Symmetry')
        self.VividColor = featureColumn(4096, 256, init, 'VividColor')
        # self.fc10 = nn.Linear(3328,128,bias=True)
        # self.fc11 = nn.Linear(128,1,bias=True)
        self.init_weights(init)

    def init_weights(self,init):


        weight = getWeights(init['conv1_0'])
        self.conv1.weight.data.copy_(weight)
        self.conv1.bias.data.copy_(torch.from_numpy(init['conv1_1']))

        weight = getWeights(init['conv2_0'], 2)
        self.conv2.weight.data.copy_(weight)
        self.conv2.bias.data.copy_(torch.from_numpy(init['conv2_1']))

        weight = getWeights(init['conv3_0'])
        self.conv3.weight.data.copy_(weight)
        self.conv3.bias.data.copy_(torch.from_numpy(init['conv3_1']))

        weight = getWeights(init['conv4_0'], 2)
        self.conv4.weight.data.copy_(weight)
        self.conv4.bias.data.copy_(torch.from_numpy(init['conv4_1']))

        weight = getWeights(init['conv5_0'], 2)
        self.conv5.weight.data.copy_(weight)
        self.conv5.bias.data.copy_(torch.from_numpy(init['conv5_1']))

        weight = getWeights(init['fc6_0'])
        self.fc6.weight.data.copy_(weight)
        self.fc6.bias.data.copy_(torch.from_numpy(init['fc6_1']))

        weight = getWeights(init['fc7_0'])
        self.fc7.weight.data.copy_(weight)
        self.fc7.bias.data.copy_(torch.from_numpy(init['fc7_1']))

        # weight = getWeights(init['fc10_merge_0'])
        # self.fc10.weight.data.copy_(weight)
        # self.fc10.bias.data.copy_(torch.from_numpy(init['fc10_merge_1']))
        #
        # weight = getWeights(init['fc11_score_0'])
        # self.fc11.weight.data.copy_(weight)
        # self.fc11.bias.data.copy_(torch.from_numpy(init['fc11_score_1']))


    def forward(self, x):

        out =  self.conv1(x)
        out = self.relu(out)
        out =   self.maxpool(out)
        out =   self.conv2(out)
        out =  self.relu(out)
        out =  self.maxpool(out)
        out =   self.conv3(out)
        out =   self.relu(out)
        out =   self.conv4(out)
        out =   self.relu(out)
        out =   self.conv5(out)
        out =   self.relu(out)
        out =   self.maxpool(out)

        out  = out.view(out.size(0), 256 * 6 * 6)
        out = self.fc6(out)
        out = self.fc7(out)

        # outNew = self.fc8(out)
        out1, _out1 = self.BalancingElement(out)
        out2, _out2 = self.ColorHarmony(out)
        out3, _out3 = self.Content(out)
        out4, _out4 = self.DoF(out)
        out5, _out5 = self.Light(out)
        out6, _out6 = self.MotionBlur(out)
        out7, _out7 = self.Object(out)
        out8, _out8 = self.Repetition(out)
        out9, _out9 = self.RuleOfThirds(out)
        out10, _out10 = self.Symmetry(out)
        out11, _out11 = self.VividColor(out)
        output = torch.cat([out1, out2, out3, out4, out5, out6,
                            out7, out8, out9, out10, out11], 1)
        # output = self.fc10(output)
        # output = self.fc11(output)
        return output

def train_model(model, criterion, optimizer, scheduler, num_epochs=40):
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
            # Iterate ovmodules = list(resnet.children())[:-1]er data.
            for data in dataloaders[phase]:


                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # print inputs.size()

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

    initDict = alex.getWeights()
    alexnet_osp = AlexNet(initDict)
    if use_gpu:
        alexnet_osp = alexnet_osp.cuda()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(alexnet_osp.parameters())
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer,7)
    best_model = train_model(alexnet_osp,criterion,optimizer,exp_lr_scheduler)