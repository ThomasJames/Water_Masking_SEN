""""
Jain et al. (2020) suggested transfer learning on a
Testing the original DenseNet on internet images
"""""

import io
from PIL import Image
import requests
from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms as transforms

DenseNet = models.DenseNet()  # This may take a few minutes.

# Random cat img taken from Google
IMG_URL = 'https://images.pexels.com/photos/104827/cat-pet-animal-domestic-104827.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=650&w=940'
# Class labels used when training VGG as json, courtesy of the 'Example code' link above.
LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'

# Let's get our class labels for the output.
response = requests.get(LABELS_URL)  # Make an HTTP GET request and store the response.
labels = {int(key): value for key, value in response.json().items()}

# Let's get the cat img.
response = requests.get(IMG_URL)
img = Image.open(io.BytesIO(response.content))  # Read bytes and store as an img.

# We can do all this preprocessing using a transform pipeline.
min_img_size = 224  # The min size, as noted in the PyTorch pretrained models doc, is 224 px.
transform_pipeline = transforms.Compose([transforms.Resize(min_img_size),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])])
img = transform_pipeline(img)

# PyTorch pretrained models expect the Tensor dims to be (num input imgs, num color channels, height, width).
# Currently however, we have (num color channels, height, width); let's fix this by inserting a new axis.
img = img.unsqueeze(0)  # Insert the new axis at index 0 i.e. in front of the other axes/dims.

# Variable; PyTorch models expect inputs to be Variables. A PyTorch Variable is a
# wrapper around a PyTorch Tensor.
img = Variable(img)


prediction = DenseNet(img)  # Returns a Tensor of shape (batch, num class labels)
prediction = prediction.data.numpy().argmax()  # Our prediction will be the index of the class images with the largest value.
print("Models predicts: ", labels[prediction])  # Converts the index to a string using our labels dict