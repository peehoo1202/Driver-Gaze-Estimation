import os
import glob
import random
import time

from PIL import Image
import numpy as np
from scipy.io import loadmat
import cv2

from torch.utils.data import Dataset
import torchvision.transforms as transforms


def get_classification_data(dataset_root_path, split, activity_classes):
    images = []
    labels = []

    for idx, act in enumerate(activity_classes):
        dir_tmp = os.path.join(dataset_root_path, split, act, '*.*g') # .jpg/jpeg/.png
        tmp = sorted(glob.glob(dir_tmp))
        labels += [idx]*len(tmp)
        images += tmp

    print('Loaded %d eye images!' % len(labels))
    time.sleep(1)
    return images, labels



class GazeDataset(Dataset):
    def __init__(self, dataset_root_path, activity_classes, split='train', random_transforms=False):
        'Initialization'
        print('Preparing '+split+' dataset...')
        self.split = split
        
        self.mean = loadmat(os.path.join(dataset_root_path, 'mean_std.mat'))['mean'][0]
        self.std = loadmat(os.path.join(dataset_root_path, 'mean_std.mat'))['std'][0]
        self.prepare_input = transforms.Compose([transforms.Resize(224), transforms.ToTensor()]) # ToTensor() normalizes image to [0, 1]
        self.normalize = transforms.Normalize(self.mean, self.std)
        if random_transforms:
            self.transforms = transforms.Compose([transforms.Resize(256),
                transforms.RandomRotation((-10, 10)), 
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.ToTensor()]) # ToTensor() normalizes image to [0, 1]
        else:
            self.transforms = None

        self.images, self.labels = get_classification_data(dataset_root_path, self.split, activity_classes)
        print('Finished preparing '+split+' dataset!')

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)

    def __getitem__(self, index):
        'Generates one sample of data'
        y = self.labels[index]
        im = Image.fromarray(cv2.cvtColor(cv2.imread(self.images[index]), cv2.COLOR_BGR2RGB)) #cv2 loads 3 channel image by default

        if self.transforms is None:
            X = self.normalize(self.prepare_input(im))
        else:
            X = self.normalize(self.transforms(im))
        return X, y
