import random
import time
import datetime
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score
from sklearn.utils.multiclass import unique_labels
from visdom import Visdom

from torch.autograd import Variable
import torch


def gan2gaze(tensor, mean, std):
    mean = mean[np.newaxis, ..., np.newaxis, np.newaxis] # (1, nc, 1, 1)
    mean = np.tile(mean, (tensor.size()[0], 1, tensor.size()[2], tensor.size()[3])) # (B, nc, H, W)
    mean = torch.from_numpy(mean.astype(np.float32)).cuda()
    std = std[np.newaxis, ..., np.newaxis, np.newaxis] # (1, nc, 1, 1)
    std = np.tile(std, (tensor.size()[0], 1, tensor.size()[2], tensor.size()[3])) # (B, nc, H, W)
    std = torch.from_numpy(std.astype(np.float32)).cuda()
    return (tensor*0.5+0.5 - mean)/std

def gaze2gan(tensor, mean, std):
    mean = mean[np.newaxis, ..., np.newaxis, np.newaxis] # (1, nc, 1, 1)
    mean = np.tile(mean, (tensor.size()[0], 1, tensor.size()[2], tensor.size()[3])) # (B, nc, H, W)
    mean = torch.from_numpy(mean.astype(np.float32)).cuda()
    std = std[np.newaxis, ..., np.newaxis, np.newaxis] # (1, nc, 1, 1)
    std = np.tile(std, (tensor.size()[0], 1, tensor.size()[2], tensor.size()[3])) # (B, nc, H, W)
    std = torch.from_numpy(std.astype(np.float32)).cuda()
    return (tensor*std+mean - 0.5)/0.5

def tensor2image(tensor, mean, std):
    mean = mean[..., np.newaxis, np.newaxis] # (nc, 1, 1)
    mean = np.tile(mean, (1, tensor.size()[2], tensor.size()[3])) # (nc, H, W)
    std = std[..., np.newaxis, np.newaxis] # (nc, 1, 1)
    std = np.tile(std, (1, tensor.size()[2], tensor.size()[3])) # (nc, H, W)

    image = 255.0*(std*tensor[0].cpu().float().numpy() + mean) # (nc, H, W)
    if image.shape[0] == 1:
        image = np.tile(image, (3, 1, 1))
    return image.astype(np.uint8) # (3, H, W)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)

def plot_confusion_matrix(y_true, y_pred, classes, output_dir=None, normalize=True, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # merge "Eyes Closed" and "Lap" classes
    y_true[y_true == 4] = 0
    y_pred[y_pred == 4] = 0
    # change GT "Shoulder" to "Left Mirror"
    y_true[np.logical_and(y_true == 2, y_pred == 3)] = 3
    # change GT "Shoulder" to "Right Mirror"
    y_true[np.logical_and(y_true == 2, y_pred == 8)] = 8
    # change prediction "Shoulder" to "Left Mirror"
    y_pred[np.logical_and(y_pred == 2, y_true == 3)] = 3
    # change prediction "Shoulder" to "Right Mirror"
    y_pred[np.logical_and(y_pred == 2, y_true == 8)] = 8
    # remove "Shoulder" class
    retain = np.logical_and(y_pred != 2, y_true != 2)
    y_true = y_true[retain]
    y_pred = y_pred[retain]
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    if output_dir is not None:
        fig.savefig(os.path.join(output_dir, 'confusion_matrix.jpg'))
    plt.close(fig)
    return 100.0*accuracy_score(y_true, y_pred), 100.0*balanced_accuracy_score(y_true, y_pred)
