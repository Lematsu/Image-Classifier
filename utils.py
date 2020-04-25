#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 
# PROGRAMMER: Leo Tomatsu
# DATE CREATED: 4/21/2020
# DATE REVISED:
# OBJECTIVE: Helper functions for train.py and predict.py
'''
Helper functions for train.py and predict.py
'''
# Import Python libraries
from collections import OrderedDict
import numpy as np
from PIL import Image
# Import ML libraries
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

def load_data(data_dir, train_dir, valid_dir, test_dir):
    ''' Use torchvision to load the data,
        return Dataloaders
    '''
    # Define parameters
    para_rotation = 30
    para_resize = 256
    para_resize_crop = 224
    para_mean = [0.485, 0.456, 0.406]
    para_std = [0.229, 0.224, 0.255]
    para_train_batch = 64
    para_valid_batch = 32
    para_test_batch = 20
    para_shuffle = True
    # Define transforms for the training, validation, 
    # and testing sets
    data_transforms = transforms.Compose([
        transforms.RandomRotation(para_rotation),
        transforms.RandomResizedCrop(para_resize_crop),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(para_mean,para_std)
    ])
    test_transforms = transforms.Compose([
        transforms.Resize(para_resize),
        transforms.CenterCrop(para_resize_crop),
        transforms.ToTensor(),
        transforms.Normalize(para_mean,para_std)
    ])
    valid_transforms = transforms.Compose([
        transforms.Resize(para_resize),
        transforms.CenterCrop(para_resize_crop),
        transforms.ToTensor(),
        transforms.Normalize(para_mean,para_std)
    ])
    # Load the datasets with Image Folder
    train_datasets = datasets.ImageFolder(train_dir,
                          transform=data_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir,
                          transform=valid_transforms)
    test_datasets = datasets.ImageFolder(test_dir,
                          transform=test_transforms)
    # Using the image datasets and transforms,
    # define the dataloaders
    train_loader = DataLoader(train_datasets,
                          batch_size=para_train_batch,
                              shuffle=para_shuffle)
    valid_loader = DataLoader(valid_datasets,
                          batch_size=para_valid_batch)
    test_loader = DataLoader(test_datasets,
                          batch_size=para_test_batch)
    return train_datasets, train_loader, valid_loader, test_loader

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Define parameters
    para_resize = 256
    para_resize_crop = 224
    para_mean = [0.485, 0.456, 0.406]
    para_std = [0.229, 0.224, 0.255]
    # Process a PIL image for use in a PyTorch model
    img_transforms = transforms.Compose([
        transforms.Resize(para_resize),
        transforms.CenterCrop(para_resize_crop),
        transforms.ToTensor(),
        transforms.Normalize(para_mean,para_std)
    ])
    image = img_transforms(Image.open(image))
    return image
