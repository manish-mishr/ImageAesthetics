import numpy as np
import os
import torch

def getWeights():
    output_path = '/home/manish/projects/deepImageAestheticsAnalysis/modelWeight'
    files = os.listdir(output_path)

    wtDict = {}
    for file in files:
        path = os.path.join(output_path,file)
        names = file.split('.')
        wts = np.load(path)
        wtDict[names[0]] = wts
    return wtDict


if __name__ == '__main__':
    wtDict = getWeights()

    for key, val in wtDict.items():
        print key, val.shape