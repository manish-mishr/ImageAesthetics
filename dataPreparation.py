import os
import glob
import csv

fileDir = os.path.join(os.getcwd(),'data_dir')
# file = 'imgList{0}Regression.txt'.format
# train = os.path.join(fileDir,file('Train'))
# test = os.path.join(fileDir,file('Test'))
# validation = os.path.join(fileDir,file('Validation'))

# dataDir = os.path.join(os.getcwd(),'datasetImages_warp256')
# fileNames = glob.glob(dataDir+'/*.jpg')
# # print len(fileNames)
#
# output_path = os.path.join(os.getcwd(),'data_dir')
# trainFiles = glob.glob(output_path+'/train/*.jpg')
# # print len(trainFiles)
#
#
# testFiles = glob.glob(output_path+'/test/*.jpg')
# # print len(testFiles)
#
# valFiles = glob.glob(output_path+'/val/*.jpg')
# # print len(valFiles)
#
# cnt = 0
# nameSet = set()
# imgSet = set()
#
# for file in testFiles:
#     f = os.path.basename(file)
#     imgSet.add(f)
#
# print len(imgSet)
# with open(test,'r') as file:
#     for cnt, line in enumerate(file):
#         if cnt == 0:
#             continue
#         words = line.split(',')
#         name = words[0]
#         nameSet.add(name)
#
# print nameSet.difference(imgSet)
csv_fmt = 'imgList{0}Regression.csv'.format
csv_file = os.path.join(fileDir,csv_fmt('Train'))
# txt_file = csv.reader(open(validation,'rb'),delimiter=',')
#
# out_csv = csv.writer(open(csv_file,'wb'))
#
# out_csv.writerows(txt_file)

import pandas as pd
frames = pd.read_csv(csv_file)
print frames.iloc[0,0]

features =  frames.iloc[2,1:].as_matrix()
print features.shape

