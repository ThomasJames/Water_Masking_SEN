import numpy as np
import matplotlib.pyplot as plt
import math
import numpy as np
from tifffile import imsave
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
import cv2
from PIL import Image, ImageOps
import random

"""
dim_pix - Dimensions of the target image (Model input)
im - 2D numpy array to be split
Location - A string to save the location of the image
dytpe - String to describe (Label or input)
"""


def split_image(dim_pix, im, location, dtype, filename):
    # Find the number of sub-images that fit in rows
    rows = []
    for i in range((math.floor(im.shape[0] / dim_pix))):
        rows.append(i)
    # Find the number of sub-images that fit in rows
    columns = []
    for i in range((math.floor(im.shape[1] / dim_pix))):
        columns.append(i)

    # Numerically identify the sub-image
    a = 0
    for i in rows:
        for j in columns:

            # Check for 244 x 244 (Mask) or 244 x 244 x 3 (TC image)
            if im[0 + (dim_pix * j): dim_pix + (dim_pix * j),
               0 + dim_pix * i: dim_pix + (dim_pix * i)].shape == \
                    (dim_pix, dim_pix) or (dim_pix, dim_pix, 3):

                # Save the 244 x 244 as an tiff file.
                imsave(f"{filename}/{location}_{region}_{a}_{dtype}.tiff",
                       im[0 + (dim_pix * j): dim_pix + (dim_pix * j),
                       0 + dim_pix * i: dim_pix + (dim_pix * i)])
                a += 1
            else:
                print("no data")


"""
Proposed Water index (WI)
i - NBDI (Zha et al., 2007)
j - PI (Jain et al., 2020)
k - PI (Jain et al., 2020)
l - SAR Scalar 

"""


def WI(SWIR2, NIR, green, SAR, i, j, k, l):
    # Extract MSI bands
    MSI = np.load(f"MSI-Region{region}.npy")
    blue = MSI[-1][:, :, 1]
    green = MSI[-1][:, :, 2]
    red = MSI[-1][:, :, 3]
    NIR = MSI[-1][:, :, 7]
    SWIR1 = MSI[-1][:, :, 10]
    SWIR2 = MSI[-1][:, :, 11]

    WI = ((i * ((SWIR2 - NIR) / (SWIR2 + NIR))) +
          (j * ((green - SWIR2) / (green + SWIR2))) +
          (k * ((blue - NIR) / (blue + NIR))))
    WI[WI > 0] = 1
    WI[WI < 0] = 0
    return WI


def NDBI(SWIR2, NIR):
    NDBI = (SWIR2 - NIR) / (SWIR2 + NIR)
    NDBI[NDBI > 0] = 1
    NDBI[NDBI < 0] = 0
    return NDBI


# Function by: Md. Rezwanul Haque (stolen from stack overflow)
def sp_noise(image, prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


if __name__ == "__main__":
    region = "3"
    location = "Colombo"

    # Extract True Color image
    TC = np.load(f"TC-Region{region}.npy")
    TC = TC[-1]

    # Extract SAR IW
    # SAR = np.load(f"SAR-Region{region}.npy")
    # SAR = SAR[-1][:, :, 0]
    #
    # plt.imshow((SAR))
    # plt.show()
    #
    # bs = 0.006
    # SAR[SAR >= bs] = 1
    # SAR[SAR < bs] = 0

    # Extract MSI bands
    MSI = np.load(f"MSI-Region{region}.npy")
    blue = MSI[-1][:, :, 1]
    green = MSI[-1][:, :, 2]
    red = MSI[-1][:, :, 3]
    NIR = MSI[-1][:, :, 7]
    SWIR1 = MSI[-1][:, :, 10]
    SWIR2 = MSI[-1][:, :, 11]

    # Water Index
    i = 0.5  # NDBI
    j = 1.0  # MDWI.1
    k = 1.0  # MDWI.2
    l = 0.0  # SAR

    WI = ((i * ((SWIR2 - NIR) / (SWIR2 + NIR))) +
          (j * ((green - SWIR2) / (green + SWIR2))) +
          (k * ((green - NIR) / (green + NIR))))
    WI[WI > 0] = 1
    WI[WI < 0] = 0

    plt.imsave("WI.png", WI, cmap="Blues")
    #  plt.imsave("SAR.png", SAR, cmap="Blues")
    plt.imsave("TC.png", TC)
    # NORMAL
    split_image(dim_pix=244, im=WI, location=location, dtype=f"Mask", filename=f"Region_{region}")
    split_image(dim_pix=244, im=TC, location=location, dtype=f"TC", filename=f"Region_{region}")

    # Horizontal Flip
    TC_Hflip = np.flip(TC, 1)
    WI_Hflip = np.flip(WI, 1)
    Hflip = "_Hflip"
    split_image(dim_pix=244, im=WI_Hflip, location=location, dtype=f"Mask", filename=f"Region_{region}{Hflip}")
    split_image(dim_pix=244, im=TC_Hflip, location=location, dtype=f"TC", filename=f"Region_{region}{Hflip}")

    # Vertical Flip
    TC_Vflip = np.flip(TC, 0)
    WI_Vflip = np.flip(WI, 0)
    Vflip = "_Vflip"
    split_image(dim_pix=244, im=WI_Vflip, location=location, dtype=f"Mask", filename=f"Region_{region}{Vflip}")
    split_image(dim_pix=244, im=TC_Vflip, location=location, dtype=f"TC", filename=f"Region_{region}{Vflip}")

    # Blur filter
    TC_Blur = cv2.medianBlur(TC, 5)
    Blur = "_Blur"
    split_image(dim_pix=244, im=WI, location=location, dtype=f"Mask", filename=f"Region_{region}{Blur}")
    split_image(dim_pix=244, im=TC_Blur, location=location, dtype=f"TC", filename=f"Region_{region}{Blur}")

    # Noise Filter
    noise = sp_noise(TC, 0.05)
    TC_noise = noise + TC
    Noise = "_Noise"
    split_image(dim_pix=244, im=WI, location=location, dtype=f"Mask", filename=f"Region_{region}{Noise}")
    split_image(dim_pix=244, im=TC_noise, location=location, dtype=f"TC", filename=f"Region_{region}{Noise}")
