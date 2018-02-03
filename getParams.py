import torchvision.models as models

import numpy as np

def buildDict(words, ind, dict):
    if ind == (len(words)-1):
        return
    if words[ind] not in dict.keys():
        dict[words[ind]] = {}
    buildDict(words,ind+1,dict[words[ind]])


def getWeights():
    resnet = models.resnet18(pretrained=True)
    model_dict = resnet.state_dict()

    keyWeights = {}

    for i, (key,value) in enumerate(model_dict.iteritems()):
        ind = int(i/5)
        rem = i%5
        wt = value.numpy()
        if wt.ndim > 1:
            if wt.ndim > 2:
                wt = np.transpose(wt,(2,3,1,0))
            else:
                wt = np.transpose(wt)


        words = key.split('.')
        buildDict(words,0,keyWeights)

        wtDict = keyWeights
        for i in words[:-1]:
            wtDict = wtDict[i]
        wtDict[words[-1]] = wt
    return keyWeights

if __name__ == '__main__':
    wts  = getWeights()
    for key in wts['layer2'].keys():
        print key


#
