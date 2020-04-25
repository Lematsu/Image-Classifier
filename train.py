#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 
# PROGRAMMER: Leo Tomatsu
# DATE CREATED: 4/21/2020
# DATE REVISED:
# OBJECTIVE: cli script to train a new network on a dataset
'''
Basic Usages: 
- Train a new network on a dataset of images
- Print out training loss, validation loss, and validation accuracy

Optional Usages:
- Choose between vgg, alexnet, and densenet architectures
- Set hyperparameters for learning rate, number of hidden units, and training epochs
- Choose to training the model with gpu

Example Usages:
- python train.py flowers --arch 'vgg' --save_dir checkpoints --lr 0.001 --hl 25088 --epoch 5 --gpu
- python train.py flowers --arch 'densenet' --save_dir checkpoints --lr 0.001 --hl 1024 --epoch 5 --gpu
- python train.py flowers --arch 'vgg' --save_dir checkpoints --lr 0.001 --hl 1024 --epoch 1
- python train.py flowers --arch 'densenet' --save_dir checkpoints --lr 0.001 --hl 1024 --epoch 1
'''
# Import Python libraries
import os
from PIL import Image
from time import time, localtime, strftime
import argparse
# Import ML libraries
import torch
from torch import nn, optim
# Import Custom libraries
import model_lib
import utils

def train_network(args):
    ''' Train network with pretrained model
        saves checkpoint
    '''
    # Parse arguments
    data_dir = args.data_dir
    arch = args.arch
    save_dir = args.save_dir
    lr = args.lr
    hl = [args.hl]
    device = 'cuda:0' if args.gpu else 'cpu'
    epochs = args.epochs
    input_size, output_size = None, 102
    dropout = 0.5
    print_every = 40
    # Define parameters
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    # Load data
    train_datasets,train_loader,valid_loader,test_loader = utils.load_data(data_dir,
                                                                           train_dir,
                                                                           valid_dir,
                                                                           test_dir)
    # Build model, criterion, and optimizer
    model, criterion, optimizer = model_lib.build_model(arch,
                                                        dropout,
                                                        lr)
    # Train model
    model_lib.train_model(model, train_loader, valid_loader, epochs,
                      print_every, criterion, optimizer, device)
    # Check accuracy of model
    model_lib.check_test_accuracy(model, test_loader, device)
    # Save checkpoint
    if save_dir is not None:
        # Make sure checkpoint exists
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # Store class index
        model.class_to_idx = train_datasets.class_to_idx
        # Create checkpoint
        checkpoint = {'architecture': arch,
                      'state_dict': model.state_dict(),
                      'class_to_idx': train_datasets.class_to_idx,
                      'classifier': model.classifier}
        # Create checkpoint file based on time to avoid overwrite
        curr_time = strftime('%H_%M_%S', localtime(time()))
        filepath = save_dir + '/{}_{}_checkpoint.pth'.format(arch, curr_time)
        torch.save(checkpoint, filepath)
        print('Model saved successfully as {}'.format(filepath))

def get_input_args():
    ''' Get user arguments,
        returns parsed argument
    '''
    # Create parser object
    parser = argparse.ArgumentParser()
    # Add argument fields
    parser.add_argument('data_dir',
                        type=str,
                        help='Path to data files.')
    parser.add_argument('--arch',
                        type=str.lower,
                        choices=['densenet', 'vgg', 'alexnet'],
                        default='densenet',
                        help='CNN model architecture.')
    parser.add_argument('--save_dir',
                        type=str,
                        help='Path to save checkpoint files.')
    parser.add_argument('--lr',
                        type=float,
                        default=0.001,
                        help='Learning rate for optimizer.')
    parser.add_argument('--hl',
                        type=int,
                        default=1024,
                        help='Number of hidden layers.')
    parser.add_argument('--epochs',
                        type=int,
                        default=1,
                        help='Number of epochs.')
    parser.add_argument('--gpu',
                        action='store_true',
                        help='Utilize gpu to train.')
    # Return parsed arguments
    return parser.parse_args()
        
def main():
    args = get_input_args()
    train_network(args)

if __name__ == '__main__':
    main()
        