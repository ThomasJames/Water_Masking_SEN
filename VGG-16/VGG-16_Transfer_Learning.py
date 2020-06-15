import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.optim import Adam
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy


torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Confirm access to a cpu or a gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# VGG-16 mean and std
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Transform the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# Load in the data

trainset = "data/test_set"
testset = "data/train_set"

print(len(trainset))


trainloader = DataLoader(trainset, batch_size=98, shuffle=True)
testloader = DataLoader(trainset, batch_size=98, shuffle=True)

for images, labels in trainloader:
    print(images.size(), labels.size())
    break

# Ask PyTorch for the pre-trained VGG-16 model
vgg16 = models.vgg16(pretrained=True)

# Set optimiser to Adam
optimiser = Adam(vgg16.parameters())

print(vgg16.features)
print(vgg16.classifier)

# When treating the network as a fixed feature extractor, we need to freeze the network
# Grab all the parameters, and set the requires grad to false.
for param in vgg16.parameters():
    param.requires_grad = False

# Then remove the models last fully connected layer, and use the fixed feature extractor
# THen add a linear classifier.
# Set the classes to 2
vgg16.classifier[-1] = nn.Sequential(
    nn.Linear(in_features=4096, out_features=2),
    nn.LogSoftmax(dim=1)
)

# Check again the architecture of the network
print(vgg16.features)
print(vgg16.classifier)

# Change the criterion to Sigmoid, for binary classification tasks
criterion = nn.Sigmoid()

num_epochs = 1   # Define the number of iterations
batch_loss = 0
cum_epoch_loss = 0  # Sets a tracker for loss






"""""
Running data through the network
"""""

model = vgg16.to(device)
optimiser = Adam(vgg16.parameters(()))

for e in range(num_epochs):    # for each predefined epoch
  cum_epoch_loss = 0
  for batch, (images, labels) in enumerate(trainloader, 1):
    images = images.to(device)  # Load the images to the device
    labels = labels.to(device)  # Load the labels to the device

    optimiser.zero_grad() # Zero out the gradients at each iter
    logps = vgg16(images)  # Run the batch through the VGG-16 model, to see what predictions the model will provide
    loss = criterion(logps, labels) # Calculate the loss
    loss.backward() # Run the backwards pass
    optimiser.step() # Update the weights using the loss values

    batch_loss += loss.item()
    print(f"Epoch({e}/{num_epochs} : number ({batch}/{len(trainloader)}))) Batch loss : {loss.item()}")

    print(f"Training loss : {batch_loss/len(trainloader)}")


"""
Evaluate the mode
During training, the output layer randomly sets some of its inputs to zero which effectivley erases them from the 
network
This makes the finley trained network more robust

"""

vgg16.to("cpu")
vgg16.eval()

with torch.no_grad():
    images, labels = next(iter(testloader))
    logps = vgg16(images)

    output = torch.exp(logps)
    print(output) # Prints the probabilities

    pred = torch.argmax(output, 1)





