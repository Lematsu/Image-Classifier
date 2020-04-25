#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 
# PROGRAMMER: Leo Tomatsu
# DATE CREATED: 4/21/2020
# DATE REVISED:
# OBJECTIVE: Functions to process models for train.py and predict.py
'''
Functions to process models for train.py and predict.py
'''
# Import Python libraries
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
from time import time, localtime, strftime
# Import ML libraries
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn.functional as F
# Import Custom libraries
import utils

def label_map(path):
    ''' Use json to load mapping,
        returns dictionary
    '''
    # load mapping in file
    with open(path, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def load_pretrain(arch):
    ''' Use torchvision to load pretrained model
        returns model
    '''
    # Define variables
    para_pretrained = True
    # Load pretrained model and get input size
    if arch == 'vgg':
        model = models.vgg16(pretrained=para_pretrained)
        input_size = model.classifier[0].in_features
    elif arch == 'densenet':
        model = models.densenet121(pretrained=para_pretrained)
        input_size = model.classifier.in_features
    else:
        raise Exception('Invalid architecture: {0}'.format(arch))
    return model, input_size

def build_model(arch,dropout=0.5,lr=0.001):
    ''' Build pretrained model and configure,
        return model, criterion, and optimizer
    '''
    # Load pretrained model
    model, input_size = load_pretrain(arch)
    # Define variables
    hidden_layer0 = (input_size, 120)
    hidden_layer1 = (120, 90)
    hidden_layer2 = (90, 80)
    hidden_layer3 = (80, 102)
    for param in model.parameters():
        param.requires_grad = False
    # Freeze parameters to prevent backprop
    for param in model.parameters():
        param.requires_grad = False
    # Configure classifier
    classifier = nn.Sequential(OrderedDict([
        ('dropout', nn.Dropout(dropout)),
        ('inputs', nn.Linear(*hidden_layer0)),
        ('relu1', nn.ReLU()),
        ('hidden_layer1', nn.Linear(*hidden_layer1)),
        ('relu2', nn.ReLU()),
        ('hidden_layer2', nn.Linear(*hidden_layer2)),
        ('relu3', nn.ReLU()),
        ('hidden_layer3', nn.Linear(*hidden_layer3)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    model.classifier = classifier
    # Create criterion
    criterion = nn.NLLLoss()
    # Create optimizer
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    return model, criterion, optimizer

def train_model(model, train_loader, valid_loader,
                  epochs, print_every, criterion, 
                  optimizer, device='cpu'):
    ''' Train model with given hyperparameters
        and prints training metrics
    '''
    # Define variables
    steps = 0
    # Turn on dropout
    model.train()
    # Train model
    model.to(device)
    print('Training with {} architecture'.format(model.__class__.__name__),
          'pre-trained model || epochs={} || device={}'.format(epochs, device))
    # Set up timer
    start_time = time()
    # Iterate over epochs
    for e in range(epochs):
        running_loss = 0
        # Iterate over train_loader
        for index, (inputs, labels) in enumerate(train_loader):
            # Increment steps
            steps += 1
            # Configure inputs and labels
            inputs,labels = inputs.to(device), labels.to(device)
            # Initialize optimizer
            optimizer.zero_grad()
            # Forward pass
            outputs = model.forward(inputs)
            # Backward Pass
            loss = criterion(outputs, labels)
            loss.backward()
            # Increment optimizer
            optimizer.step()
            # Increment running loss
            running_loss += loss.item()
            # Print training information
            if steps % print_every == 0:
                # Turn off drop-out
                model.eval()
                # Define variables
                vlost = 0
                accuracy = 0
                # Iterate over valid loader
                for index2, (inputs2, labels2) in enumerate(valid_loader):
                    # Initialize optimzer
                    optimizer.zero_grad()
                    # Configure inputs, labels, and model
                    inputs2 = inputs2.to(device)
                    labels2 = labels2.to(device)
                    model.to(device)
                    # Increment loss and accuracy
                    with torch.no_grad():
                        outputs2 = model.forward(inputs2)
                        loss2 = criterion(outputs2, labels2)
                        ps = torch.exp(outputs2).data
                        equality = (labels2.data == ps.max(1)[1])
                        accuracy += equality.type_as(torch.FloatTensor()).mean()
                # Calculate metrics
                validation_loss = loss2 / len(valid_loader)
                accuracy = accuracy / len(valid_loader)
                running_loss = running_loss / len(train_loader)
                print('Epoch: {}/{}...'.format(e+1, epochs),
                      'Loss: {:.4f}'.format(running_loss),
                      'Validation Loss {:.4f}'.format(validation_loss),
                      'Validation Accuracy: {:.4f}'.format(accuracy))
                # Re-initialize variables
                running_loss = 0
    # Calculate process time
    end_time = time()
    total_time = strftime('%H:%M:%S',
                          localtime(end_time - start_time))
    print('Total Training Time: {}'.format(total_time))

def check_test_accuracy(model, loader, device):
    ''' Check trained data with test data
        prints accuracy
    '''
    # Define variables
    correct = 0
    total = 0
    model.to(device)
    # Calculate accuracy
    with torch.no_grad():
        # Iterate over loader
        for images, labels in loader:
            # compare outputs between loader and model
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            # Increment total
            total += labels.size(0)
            # Increment correct value if conditions meet
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the',
          'test images: {} %%'.format(100 * correct / total))

def load_checkpoint(filepath):
    ''' Loads a checkpoint and rebuilds the model
        returns model and checkpoint
    '''
    # Load checkpoint
    checkpoint = torch.load(filepath,
                            map_location=lambda storage,
                            loc: storage)
    # Build model
    arch = checkpoint['architecture']
    model,_ = load_pretrain(arch)
    # Store class index and state dict
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model, checkpoint

def predict(image_path, model, device,
            idx_to_class, topk=5):
    ''' Predict the top K classes along with associated probabilities
        returns list of probabilities and classes
    '''
    print('Predicting the top {} classes with'.format(topk),
          '{} pre-trained model'.format(model.__class__.__name__),
          '| device={}.'.format(device))
    # Load an image file
    processed_image = utils.process_image(image_path).squeeze()
    # Turn off drop-out
    model.eval()
    # Change to cuda
    if device == 'gpu':
        model = model.cuda()
    # Predict output
    with torch.no_grad():
        if device == 'gpu':
            output = model(processed_image.float().cuda().unsqueeze_(0))
        else:
            output = model(processed_image.float().unsqueeze_(0))
    # Calculate the class probabilities (softmax) for image
    ps = F.softmax(output,dim=1)
    top = torch.topk(ps,topk)
    probs = top[0][0].cpu().numpy()
    classes = [idx_to_class[i] for i in top[1][0].cpu().numpy()]
    return probs, classes
