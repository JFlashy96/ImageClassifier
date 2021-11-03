import argparse
import helper
from collections import OrderedDict
import copy
from PIL import Image
from torch import nn
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms, models
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os, random, sys
import time
import torch
import torch.nn.functional as F
import torchvision

"""
Command Line Inteface (CLI) for training model"
"""

parser = argparse.ArgumentParser(
	description = "Parser for train.py"
	) 

parser.add_argument("--data_dir", action="store", default="./flowers/")
parser.add_argument("--save_dir", action='store', default="./checkpoint.pth")
parser.add_argument("--arch", action="store",                    
							 choices=["vgg11", "vgg11_bn", "vgg13", "vgg13_bn",
                             "vgg16", "vgg16_bn", "vgg19", "vgg19_bn"],
                             default="vgg19")
parser.add_argument("--learning_rate", action="store", type=float, default=0.01)
#TODO: Get optimal hidden_unit value and set as default
parser.add_argument('--hidden_units', action="store", dest="hidden_units", type=int, default=512)
parser.add_argument("--num_epochs", action="store", type=int, default=5)
parser.add_argument("--device", action="store", default="cpu")

args = parser.parse_args()
data_dir = args.data_dir
save_dir = args.save_dir
arch = args.arch
learning_rate = args.learning_rate
hidden_units = args.hidden_units
num_epochs = args.num_epochs
device = args.device
criterion = nn.CrossEntropyLoss()
train_loader, val_loader, test_loader, train_data = helper.load_data(data_dir)
dataloaders = [train_loader, val_loader, test_loader]

def main():

	model, optimizer, classifier = helper.setup_network(arch, device, hidden_units, learning_rate)
	model.class_to_idx = train_data.class_to_idx

	# Run training and validation step
	helper.train_model(model, dataloaders, criterion, optimizer, device, num_epochs)
	helper.save_checkpoint(train_data, model, arch, hidden_units, learning_rate, classifier, num_epochs, optimizer)

if __name__ == "__main__":
	main()