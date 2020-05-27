import numpy as np
import matplotlib.pyplot as plt

import math
from PIL import Image


SAR = np.load("Shanghai/SAR-Region4.npy")
SAR = SAR[-1][:, :, 0]
# SAR[SAR > 0.1] = 1
# SAR[SAR < 0.1] = 0

MSI = np.load("Shanghai/MSI-Region4.npy")
blue = MSI[-1][:, :, 1]
green = MSI[-1][:, :, 2]
red = MSI[-1][:, :, 3]
NIR = MSI[-1][:, :, 7]
SWIR1 = MSI[-1][:, :, 10]
SWIR2 = MSI[-1][:, :, 11]

NDBI = (SWIR2 - NIR)/ (SWIR2 + NIR)
NDBI[NDBI > 0] = 255
NDBI[NDBI < 0] = 0

plt.imshow(NDBI)
plt.show()


# Decide on an index
PI = (((SWIR2 - NIR) / (SWIR2 + NIR)) + ((green - SWIR2) / (green + SWIR2)) + ((green - NIR) / (green + NIR)))

# Interpolate the index

PI[PI > 0] = 255
PI[PI < 0] = 0

plt.imshow(PI)
plt.show()


TC = np.load("Shanghai/TC-Region4.npy")
print(TC.shape)
TC = TC[-1]


plt.imsave("PI-shangai.png", PI, cmap="Blues")
plt.imsave("TC-shangai.png", TC)
