import logging
import os
import shutil
import sys
from time import localtime, strftime
from skimage import io
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm
from training.utils.data_transforms import ToTensor, ToBinaryTensor
from training.utils.dataset import SpaceNetDataset, SpaceNetDatasetBinary
from training.utils.logger import Logger
from training.utils.train_utils import AverageMeter, log_sample_img_gt, render
from torch.nn import Conv2d, MaxPool2d, ReLU, Linear, Softmax, BatchNorm2d
import torch.nn as nn

TRAIN = {
    # hardware and framework parameters
    'use_gpu': True,
    'dtype': torch.float64,

    # paths to data splits
    'data_path_root': '/qfs/projects/sgdatasc/spacenet/', # common part of the path for data_path_train, data_path_val and data_path_test
    'data_path_train': 'Vegas_processed_train/annotations',
    'data_path_val': 'Vegas_processed_val/annotations',
    'data_path_test': 'Vegas_processed_test/annotations',

    # training and model parameters
    'evaluate_only': False,  # Only evaluate the model on the val set once
    'model_choice': 'unet_baseline',  # 'unet_baseline' or 'unet'
    'feature_scale': 1,  # parameter for the Unet

    'num_workers': 4,  # how many subprocesses to use for data loading
    'train_batch_size': 10,
    'val_batch_size': 10,
    'test_batch_size': 10,

    'starting_checkpoint_path': '',  # checkpoint .tar to train from, empty if training from scratch
    'loss_weights': [0.1, 0.8, 0.1],  # weight given to loss for pixels of background, building interior and building border classes
    'learning_rate': 0.5e-3,
    'print_every': 200,  # print every how many steps
    'total_epochs': 20,  # for the walkthrough, we are training for one epoch

    'experiment_name': 'unet_binary_weights', # using weights that emphasize the building interior pixels
}

data_path_root = TRAIN['data_path_root']
data_path_train = os.path.join(data_path_root, TRAIN['data_path_train'])

split_tags = ['trainval', 'test']

dset = SpaceNetDatasetBinary(data_path_train, split_tags,
                             transform=T.Compose([ToBinaryTensor()]))


print(dset[0])
print(dset[1])
print(dset[2])
print(dset[3])
print(dset[4])
