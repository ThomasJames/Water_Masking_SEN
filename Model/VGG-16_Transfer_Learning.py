import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Assign a device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
trainset = ""
testset = ""

vgg16 = models.vgg16(pretrained=True)

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


