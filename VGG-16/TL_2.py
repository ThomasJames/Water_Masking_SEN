from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torch import save
import torch
from torch import optim

"""""           
Set the device   
"""""
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
print(f"device is: {device}")

"""""           
Applying Transformation to the data   
"""""
image_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        # transforms.RandomRotation(degrees=15),
        # transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

"""""           
Loading in the data  
train
validation
test 

"""""
train_directory = "/Users/tj/PycharmProjects/CNN_for_EO/data/train_set/"
valid_directory = "/Users/tj/PycharmProjects/CNN_for_EO/data/Validate/"
test_directory = "/Users/tj/PycharmProjects/CNN_for_EO/data/test_set/"

# Batch size
bs = 20

# Number of classes
num_classes = 2

# Load Data from folders
data = {
    'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
    'valid': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid']),
    'test': datasets.ImageFolder(root=test_directory, transform=image_transforms['test'])
}

# Size of Data, to be used for calculating Average Loss and Accuracy
train_data_size = len(data['train'])
valid_data_size = len(data['valid'])
test_data_size = len(data['test'])

# Create iterators for the Data loaded using DataLoader module
train_data = DataLoader(data['train'], batch_size=bs, shuffle=False)
valid_data = DataLoader(data['valid'], batch_size=bs, shuffle=False)
test_data = DataLoader(data['test'], batch_size=bs, shuffle=False)

# Print the train, validation and test set data sizes
print(f"Train size: {train_data_size}, \n"
      f"validation size: {valid_data_size},\n"
      f"Test size:  {test_data_size}")

# Iterate through the dataloader once
trainiter = iter(train_data)
features, labels = next(trainiter)
print(f"Features shape: {features.shape}, \n"
      f"Labels shape: {labels.shape}")



"""""
Loading a pre-trained Vgg-16 network
Change the classifer layer
"""""

model = models.vgg16(pretrained=True)













