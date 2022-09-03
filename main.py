#!/usr/bin/env python
# coding: utf-8

import argparse
import pickle

parser = argparse.ArgumentParser('Vox2Vox training and validation script', add_help=False)

## training parameters
parser.add_argument('-g', '--gpu', default=0, type=int, help='GPU position')
parser.add_argument('-nc', '--n_classes', default=2, type=int, help='number of classes')
parser.add_argument('-bs', '--batch_size', default=4, type=int, help='batch size')
parser.add_argument('-nch', '--n_channels', default=4, type=int, help='n channels')
parser.add_argument('-a', '--alpha', default=5, type=int, help='alpha weight')
parser.add_argument('-ne', '--num_epochs', default=500, type=int, help='number of epochs')
parser.add_argument('-ct', '--continue_training', action='store_true', help='continue training from the last epoch')

args = parser.parse_args()
gpu = args.gpu
n_classes = args.n_classes
n_channels = args.n_channels
batch_size = args.batch_size
alpha = args.alpha
n_epochs = args.num_epochs
continue_training = args.continue_training


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)
os.environ

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import numpy as np
import nibabel as nib
import glob
import time
from tensorflow.keras.utils import to_categorical
from sys import stdout
import matplotlib.pyplot as plt
import matplotlib.image as mpim
from scipy.ndimage.interpolation import affine_transform
import sklearn.utils.class_weight as class_weight
from sklearn.model_selection import train_test_split

from utils import *
from augmentation import *
from losses import *
from models import *


#Nclasses = 4
Nclasses = 2
classes = np.arange(Nclasses)

data_folder_path = '/content/drive/MyDrive/Fitria/Data Tugas Akhir/'
# images lists
t1_list = sorted(glob.glob(data_folder_path + '*/*Vol.nii'))
seg_list = sorted(glob.glob(data_folder_path + '*/*Seg.nii'))

# create the training and validation sets
Nim = len(t1_list)
idx = np.arange(Nim)

sets = {'train': [], 'valid': [], 'test': []}

if not continue_training:
    idxTrain, idxValid = train_test_split(idx, test_size=0.25)
    idxValid, idxTest = train_test_split(idxValid, test_size=0.5)

    for i in idxTrain:
        sets['train'].append([t1_list[i], seg_list[i]])
    for i in idxValid:
        sets['valid'].append([t1_list[i], seg_list[i]])
    for i in idxTest:
        sets['test'].append([t1_list[i], seg_list[i]])

    pickle.dump(sets, open(data_folder_path + 'data_path_72data3', 'wb'))
else:
    sets = pickle.load(open(data_folder_path + 'data_path_72data3', 'rb'))

print(sets)

print('Preparing data generator...')
train_gen = DataGenerator(sets['train'], batch_size=batch_size, n_channels=n_channels, n_classes=n_classes, augmentation=True)
print('Training data generator created. \tSize:',   len(train_gen))
valid_gen = DataGenerator(sets['valid'], batch_size=batch_size, n_channels=n_channels, n_classes=n_classes, augmentation=True)
print('Validation data generator created. \tSize:', len(valid_gen))
test_gen = DataGenerator(sets['test'], batch_size=batch_size, n_channels=n_channels, n_classes=n_classes, augmentation=True)
print('Test data generator created. \tSize:', len(test_gen))  

print('Input shape:', train_gen[0][0].shape, '| Output shape:', train_gen[0][1].shape)

from train_v2v import *
print("Training the model...")

# train the vox2vox model
h = fit(train_gen, valid_gen, alpha, n_epochs, continue_training)
