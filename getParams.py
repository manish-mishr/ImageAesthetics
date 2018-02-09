import torchvision.models as models
import numpy as np

output_path = '/home/manish/projects/ResNetModel/modelWeight'

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
    getpretrainedWts(keyWeights)
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




def getpretrainedWts(dict):
    layers = ['BalancingElement', 'ColorHarmony', 'Content', 'DoF', 'Light', 'MotionBlur',
              'Object', 'Repetition', 'RuleOfThirds', 'Symmetry', 'VividColor']

    for layer in layers:
        wts = 'fc9_' + layer
        net_weights = np.load(output_path+'/'+wts+'_0.npy')
        net_weights = np.transpose(net_weights)
        net_bias = np.load(output_path+'/'+wts+'_1.npy')
        dict[layer] = {}
        dict[layer]['wt'] = net_weights
        dict[layer]['bias'] = net_bias






if __name__ == '__main__':

    wts  = getWeights()
    print wts.keys()


#
