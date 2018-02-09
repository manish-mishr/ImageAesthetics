import torch
from torch.utils.data import Dataset

from torchvision import transforms

from skimage import io, transform
from PIL import Image
import os
import pandas as pd
import numpy as np


data_dir = '/home/manish/projects/ResNetModel/data_dir'


class AADBDataset(Dataset):
    '''
    AADB dataset loader
    '''

    def __init__(self, csv_file, root_dir, transform=None):
        self.imgList = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.imgList)

    def __getitem__(self, idx):
        img_file = os.path.join(self.root_dir, self.imgList.iloc[idx, 0])
        image = Image.open(img_file)
        image = image.convert('RGB')
        labels = self.imgList.iloc[idx, 1:].as_matrix()
        labels = labels.astype(dtype=np.float32)
        labels = torch.from_numpy(labels)
        # print image.dtype, labels.dtype
        if self.transform:
            image = self.transform(image)

        return image, labels

class Rescale(object):

    def __init__(self,output_size):
        assert isinstance(output_size,(int,tuple))
        self.output_size = output_size
    def __call__(self, image):

        new_h, new_w = self.output_size

        img = transform.resize(image,(new_h,new_w))

        return img

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image):

        try:
            image = image.transpose((2, 0, 1))
        except:
            print image.shape
            raw_input("check")
        return torch.from_numpy(image).double()

class Normalize(object):

   def __init__(self, mean, std):
        self.mean = mean
        self.std = std

   def __call__(self, tensor):

        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor


# Rescale((224,224))
size = 224,224
def getdataLoader():
    csv_fmt = 'imgList{0}Regression.csv'.format

    data_transforms = transforms.Compose([
            transforms.Scale((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    image_datasets = {}
    dataloaders = {}
    dataset_sizes = {}
    for x in ['train', 'validation']:
        train_csv = os.path.join(data_dir, csv_fmt(x.capitalize()))
        train_dir = os.path.join(data_dir, x)
        image_datasets[x] = AADBDataset(train_csv,train_dir,data_transforms)
        dataloaders[x] = torch.utils.data.DataLoader(image_datasets[x], batch_size=64,
                                                  shuffle=True, num_workers=4)
        dataset_sizes[x] = len(image_datasets[x])

    return dataloaders,dataset_sizes