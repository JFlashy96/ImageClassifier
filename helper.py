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
Helpfer functions to aid the training of model
and prediction of input image
"""

def load_data(root="./flowers"):

	data_dir = root   
	train_dir = data_dir + "/train"
	valid_dir = data_dir + "/valid"
	test_dir = data_dir + "/test" 
	
	"""
	 Normalize the means and standard deviations of the images to what the network expects
	 #Normalization keeps the weight near zero which tends to make backpropogation more stable.
	 TODO: Understand and be able to apply backpropogation optimization techniques.
	"""
	std = [0.229, 0.224, 0.225]
	means = [0.485, 0.456, 0.406]

	train_transform = transforms.Compose([transforms.RandomRotation(30),
							transforms.RandomResizedCrop(224),
							transforms.RandomHorizontalFlip(),
							transforms.ToTensor(),
							transforms.Normalize(means, std)])

	val_transform = transforms.Compose([transforms.Resize(256),
							transforms.CenterCrop(224),
							transforms.ToTensor(),
							transforms.Normalize(means, std)])

	test_transform = transforms.Compose([transforms.Resize(256),
							transforms.CenterCrop(224),
							transforms.ToTensor(),
	                        transforms.Normalize(means, std)])

	print("Initializing Datasets and DataLoaders")

	# TODO: Load the datasets with ImageFolder
	# Load the datasets with ImageFolder
	train_data = datasets.ImageFolder(train_dir, transform=train_transform)
	val_data = datasets.ImageFolder(valid_dir, transform=val_transform)
	test_data = datasets.ImageFolder(test_dir, transform=test_transform)

	image_datasets = [train_data, val_data, test_data]

	# TODO: Using the image datasets and the trainforms, define the dataloaders
	# Dataloader definitions using the image datasets and the transforms
	train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
	val_loader = torch.utils.data.DataLoader(val_data, batch_size=32)
	test_loader = torch.utils.data.DataLoader(test_data, batch_size=32)

	return train_loader, val_loader, test_loader, train_data

def setup_network(arch, device, hidden_units, learning_rate):

	model = getattr(models, arch)(pretrained=True)

	for param in model.parameters():
		param.requires_grad = False

	# Define a new, untrained feed-forward network as a classifier, using ReLU activations and
	# dropout. 
	classifier = nn.Sequential(OrderedDict([
										('fcl', nn.Linear(25088, 1024)),
										('drop', nn.Dropout(p=0.5)),
										('relu', nn.ReLU()),
										('fc2', nn.Linear(1024, 102)),
										('output', nn.LogSoftmax(dim=1))
										]))

	model.classifier = classifier

	# Train the classifier layers using backpropagation using the pre-trained network to get the features.
	# Track the loss and accuracy on the validation set to determine the best hyperparameters.

	model = model.to(device)
	if torch.cuda.is_available() and device == 'gpu':
		model.cuda()

	# Gather the parameters to be optimized/updated in this run.
	# If we are finetuning we will be updating all parameters. However,
	# if we are doing feature extract method, we will only update
	# the parameters that we have just initialized. i.e., the parameters
	# with requires_grad is True.
	feature_extract = True
	params_to_update = model.parameters()
	print("Param to learn:")
	if feature_extract:
		params_to_update = []
		for name, param in model.named_parameters():
			if param.requires_grad == True:
				params_to_update.append(param)
				print("\t", name)
	else:
		for name, param in model.named_parameters():
			if param.requires_grad == True:
				print("\t", name)

	optimizer = optim.SGD(params_to_update, lr=0.01)

	return model, optimizer, classifier

def save_checkpoint(train_data, model, arch, learning_rate, classifier, num_epochs, optimizer):
	model.class_to_idx = train_data.class_to_idx

	checkpoint = {'input_size': 25088,
				   'output_size': 102,
				    'arch': arch,
				    'learning_rate': learning_rate,
				    'batch_size': 64,
				    'classifier': classifier,
				    'num_epochs': num_epochs,
				    'optimizer': optimizer.state_dict(),
				    'state_dict': model.state_dict(),
				    'class_to_idx': model.class_to_idx}

	torch.save(checkpoint, 'checkpoint.pth')

def load_checkpoint(filename, arch, device):
	checkpoint = torch.load(filename)
	learning_rate = checkpoint['learning_rate']
	model, optimizer, classifier = setup_network(arch,device,learning_rate)

	model = getattr(torchvision.models, checkpoint['arch'])(pretrained=True)
    
	model.num_epochs = checkpoint['num_epochs']
	model.class_to_idx = checkpoint['class_to_idx']
	model.load_state_dict(checkpoint['state_dict'])
                                    
	return model

def train_model(model, dataloaders, criterion, optimizer,device, num_epochs=25):
	since = time.time()

	# list to keep track of model performance accuracy over epochs
	val_acc_history = []
	best_acc = 0.0
	best_model_wts = copy.deepcopy(model.state_dict())

	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)

		train_mode = 0
		valid_mode = 1

		# Each epoch has a training and validation mode
		for mode in [train_mode, valid_mode]:
			if mode == train_mode:
				model.train() # set model to training mode
			else:
				model.eval() # set model to evaluation mode

			running_loss = 0.0
			running_corrects = 0
			pass_count = 0
            
			# Iterate over data.
			for inputs, labels in dataloaders[mode]:
				pass_count += 1
				inputs = inputs.to(device)
				labels = labels.to(device)

				# zero the parameter gradients
				optimizer.zero_grad()

				# forward
				# only track history if in train

                # Get model output and calculate loss
				output = model.forward(inputs)
				loss = criterion(output, labels)

				_, preds = torch.max(output, 1)

                # Backward. Optimize only if in training mode
				if mode == train_mode:
					loss.backward()
					optimizer.step()

                # statistics
				running_loss += loss.item() * inputs.size(0)
				running_corrects += torch.sum(preds == labels.data)

			epoch_acc = running_corrects.double() / len(dataloaders[mode].dataset)

			if mode == train_mode:
				print("\nEpoch: {}/{} ".format(epoch+1, num_epochs),
                      "\nTraining Loss: {:.4f}  ".format(running_loss/pass_count))
			else:
				print("Validation Loss: {:.4f}  ".format(running_loss/pass_count),
                  "Accuracy: {:.4f}".format(epoch_acc))
                
				running_loss = 0

        
	time_elapsed = time.time() - since
	print("\nTotal time: {:.0f}m {:.0f}s".format(time_elapsed//60, time_elapsed % 60))


def predict(image_path, model, topk, device):
	# Predict the class (or classes) of an image using a pretrained deep learing model.

	# TODO: Implement the code to predict the class from an image file
	# move the model to cuda
	if device == "cuda":
		# Move model parameters to the GPU
		model.cuda()
		print("Number of GPUS:", torch.cuda.device_count())
		print("Device name:", torch.cuda.get_device_name(torch.cuda.device_count()-1))
	else:
		model.cpu()

	# turn off dropout
	model.eval()

	# The image
	image = process_image(image_path)

	# transfer to tensor
	image = torch.from_numpy(np.array([image])).float()

	# The image becomes the input
	image = Variable(image)
	if device == "cuda":
		image = image.cuda()
	output = model.forward(image)

	probabilities = torch.exp(output).data

	# getting the topk
	# 0 --> probabilities
	# proabilities is a list of the topk 
	probabilities = torch.topk(probabilities, topk)[0].tolist()[0] 

	return probabilities

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    im = Image.open(image)
    im = im.resize((256, 256))
    value = 0.5 * (256-224)
    im = im.crop((value, value, 256-value, 256-value))
    im = np.array(im)/255

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    im = (im - mean) / std 

    return im.transpose(2,0,1)
