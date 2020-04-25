#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 
# PROGRAMMER: Leo Tomatsu
# DATE CREATED: 4/22/2020
# DATE REVISED:
# OBJECTIVE: cli script to read in an image and a checkpoint then prints the most likely image class and it's associated probability
'''
Basic Usages:
- Reads in an image and a checkpoint
- Print top k most likely classes along with associated probabilities

Optional Usages:
- Change values of number top k most likely classes
- Assign label mapping from categories to names
- Use GPU for inference

Example Usages:
- python predict.py flowers/test/1/image_06743.jpg checkpoints/vgg_04_24_37_checkpoint.pth --fp_classmap cat_to_name.json --top_k 5 --gpu
- python predict.py flowers/test/2/image_05100.jpg checkpoints/densenet_18_10_23_checkpoint.pth --fp_classmap cat_to_name.json --top_k 5 --gpu

- python predict.py flowers/test/1/image_06743.jpg checkpoints/vgg_04_24_37_checkpoint.pth --fp_classmap cat_to_name.json --top_k 5
- python predict.py flowers/test/2/image_05100.jpg checkpoints/densenet_18_10_23_checkpoint.pth --fp_classmap cat_to_name.json --top_k 5
'''
# Import Python libraries
import argparse
from PIL import Image
# Import ML libraries
import torch
from torch import nn, optim
import torch.nn.functional as F
# Import Custom libraries
import model_lib
import utils

def get_input_args():
    ''' Get user arguments,
        returns parsed argument
    '''
    # Create parser object
    parser = argparse.ArgumentParser()
    # Add argument fields
    parser.add_argument('fp_image',
                        help='Path to image file.')
    parser.add_argument('fp_checkpoint',
                        help='Path to checkpoint file.')
    parser.add_argument('--fp_classmap', type=str,
                        default='cat_to_name.json',
                        help='Path to classes map file.')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of highest probabilities.')
    parser.add_argument('--gpu', action='store_true',
                        help='Use gpu to train.')
    return parser.parse_args()

def predict_network(args):
    ''' Predict image results with loaded model
        prints results
    '''
    # Define variables
    fp_image = args.fp_image
    fp_checkpoint = args.fp_checkpoint
    fp_classmap = args.fp_classmap
    top_k = args.top_k
    device = 'gpu' if args.gpu else 'cpu'
    # Load model
    model, checkpoint = model_lib.load_checkpoint(fp_checkpoint)
    # Get index to class map
    class_to_idx = checkpoint['class_to_idx']
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    # Predict image
    probs, classes = model_lib.predict(fp_image, model,
                                       device, idx_to_class,
                                       top_k)
    # Print results
    cat_to_name = model_lib.label_map(fp_classmap)
    names = [cat_to_name[c] for c in classes]
    print('Classes           : {}'.format(names))
    print('Probabilities (%):', [float(round(p * 100.0, 2)) for p in probs])
    print('Most probable class: {}'.format(names[0]))

def main():
    args = get_input_args()
    predict_network(args)

if __name__ == '__main__':
    main()