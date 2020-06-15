import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import ImageFilter
from scipy.ndimage import filters
from scipy.ndimage import measurements
import math
import random
from s2cloudless import S2PixelCloudDetector, CloudMaskRequest
import torch
from torch.nn import Softmax

# Lee filter - Set to 7 x 7 window
def lee_filter(img, size=7):
    img_mean = filters.uniform_filter(img, (size, size))
    img_sqr_mean = filters.uniform_filter(img ** 2, (size, size))
    img_variance = img_sqr_mean - img_mean ** 2
    overall_variance = measurements.variance(img)
    img_weights = img_variance / (img_variance + overall_variance)
    img_output = img_mean + img_weights * (img - img_mean)
    return img_output


def split_image(dim_pix, im, location, dtype, filename, region):
    # Find the number of sub-images that fit in rows
    rows = []
    for i in range((math.floor(im.shape[0] / dim_pix))):
        rows.append(i)
    # Find the number of sub-images that fit in rows
    columns = []
    for i in range((math.floor(im.shape[1] / dim_pix))):
        columns.append(i)

    # Numerically identify the sub-labels
    a = 0
    for i in rows:
        for j in columns:

            # Check for 244 x 244 (Mask) or 244 x 244 x 3 (TC labels)
            if im[0 + (dim_pix * j): dim_pix + (dim_pix * j),
               0 + dim_pix * i: dim_pix + (dim_pix * i)].shape == \
                    (dim_pix, dim_pix) or (dim_pix, dim_pix, 3):

                # Save the 244 x 244 as an tiff file.
                plt.imsave(f"{filename}/{location}_{region}_{a}_{dtype}.tiff",
                           im[0 + (dim_pix * j): dim_pix + (dim_pix * j),
                           0 + dim_pix * i: dim_pix + (dim_pix * i)])
                a += 1
            else:
                print("no data")


# Function by: Md. Rezwanul Haque (stolen from stack overflow)
def sp_noise(image, prob):
    '''
    Add salt and pepper noise to labels
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

    print("Image Labelling Program")
    print(" ")
    region = input(str("Region: "))
    region_number = input(str("Region number: "))
    apply_sar = input(str("Is SAR avaliable?: "))
    output = input(str("Shall I output the tiles on this run?: "))
    project = "DataLabelling"

    # True Color for reference
    TC = np.load(f"/Users/tj/PycharmProjects/{project}/{region}/TC-Region{region_number}.npy")[-1]
    plt.imshow(TC)
    plt.show()
    plt.imsave("Index_samples/TC.png", TC)

    """"" 
    Water masking with Multispectral imagery (MSI) 
    """""
    print(f"Loading {region} MSI data")
    MSI = np.load(f"{region}/MSI-Region{region_number}.npy")
    blue =   MSI[-1][:, :, 1]
    green =  MSI[-1][:, :, 2]
    red =    MSI[-1][:, :, 3]
    NIR =    MSI[-1][:, :, 7]
    SWIR1 =  MSI[-1][:, :, 10]
    SWIR2 =  MSI[-1][:, :, 11]

    """""                                                
    Cloud Masking      
    """""

    all_bands = True

    bands_num = ((MSI[-1].shape)[2])
    if bands_num == 13:
        bands_num = True
    elif bands_num == 12:
        bands_num = False
    else:
        print("Cloud mask wont compute")


    need_cloud_filter = input(str("Is a cloud filter needed?: "))

    if need_cloud_filter == "yes":
        print("Applying cloud mask...")
        cloud_detector = S2PixelCloudDetector(threshold=0.4,
                                              average_over=4,
                                              dilation_size=2,
                                              all_bands=all_bands)

        cloud_mask = (cloud_detector.get_cloud_masks(MSI))[-1]
        plt.imshow(cloud_mask, cmap="Blues")
        plt.show()


    """""                                                
    Calculating Spectral Indicies      
    """""

    print("Creating Spectral Indicies...")
    # Normalized Difference Water Index (NDWI) McFeeters (1996)
    NDWI = ((green - NIR) / (green + NIR))
    NDWI[NDWI < 0] = 0
    NDWI[NDWI > 0] = 1

    #  Modified NDWI Xu (2006)
    MNDWI = ((green - SWIR2) / (green + SWIR2))
    MNDWI[MNDWI < 0] = 0
    MNDWI[MNDWI > 0] = 1

    # Index I Yang et al., (2017)
    I = ((green - NIR) / (green + NIR)) + ((blue - NIR) / (blue + NIR))
    I[I < 0] = 0
    I[I > 0] = 1

    # Proposed Index - (Jain et al., 2020)
    PI = ((green - SWIR2) / (green + SWIR2)) + ((blue - NIR) / (blue + NIR))
    PI[PI < 0] = 0
    PI[PI > 0] = 1

    # Normalised built up index
    NBDI = ((SWIR2 - NIR) / (SWIR2 + NIR))
    NBDI[NBDI < 0] = 0
    NBDI[NBDI > 0] = 1

    # Normalised Vegetation difference index (NVDI)
    NVDI = ((NIR - red) / (NIR + red))
    NVDI[NVDI < 0] = 0
    NVDI[NVDI > 0] = 1

    # Modified NBDI (BU)
    BU = ((SWIR2 - NIR) / (SWIR2 + NIR)) - ((NIR - red) / (NIR + red))
    BU[BU < 0] = 0
    BU[BU > 0] = 1

    print("Plotting...")
    # Display and output
    plt.imshow(NDWI, cmap="Blues")
    plt.title("NDWI")
    plt.show()
    plt.imsave("Index_samples/NDWI.png", NDWI, cmap="Blues")

    plt.imshow(MNDWI, cmap="Blues")
    plt.title("MNDWI")
    plt.show()
    plt.imsave("Index_samples/MNDWI.png", MNDWI, cmap="Blues")

    plt.imshow(I, cmap="Blues")
    plt.title("I")
    plt.show()
    plt.imsave("Index_samples/I.png", I, cmap="Blues")

    plt.imshow(PI, cmap="Blues")
    plt.title("PI")
    plt.show()
    plt.imsave("Index_samples/PI.png", PI, cmap="Blues")

    plt.imshow(NBDI, cmap="Blues")
    plt.title("NBDI")
    plt.show()
    plt.imsave("Index_samples/NBDI.png", NBDI, cmap="Blues")

    plt.imshow(NVDI, cmap="Blues")
    plt.title("NVDI")
    plt.show()
    plt.imsave("Index_samples/NVDI.png", NVDI, cmap="Blues")

    plt.imshow(BU, cmap="Blues")
    plt.title("BU")
    plt.show()
    plt.imsave("Index_samples/BU.png", BU, cmap="Blues")

    """"" 
    Water masking with Synthetic Apeture Radar (SAR)
    """""
    if apply_sar == "yes":
        print(f"Loading {region} SAR data...")
        SAR = np.load(f"{region}/SAR-Region{region_number}.npy")[-1]
        sar_hh = SAR[:, :, 0]
        sar_vv = SAR[:, :, 1]

        sar_hh = lee_filter(sar_hh)
        plt.imsave("sar_hh.png", sar_hh, cmap="cividis")  # Saves to the local directory
        sar_hh = cv2.imread("sar_hh.png", 0)
        blur_hh = cv2.GaussianBlur(sar_hh, (5, 5), 0)
        ret3_hh, th3_hh = cv2.threshold(blur_hh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        plt.imshow(th3_hh, cmap="cividis")
        plt.title("HH")
        plt.show()

        sar_vv = lee_filter(sar_vv)
        plt.imsave("sar_vv.png", sar_vv, cmap="cividis")  # Saves to the local directory
        sar_vv = cv2.imread("sar_vv.png", 0)
        blur_vv = cv2.GaussianBlur(sar_vv, (5, 5), 0)
        ret3_vv, th3_vv = cv2.threshold(blur_vv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        plt.imshow(th3_vv, cmap="cividis")
        plt.title("VV")
        plt.show()

    """""
    Water Index fusion algorithm 
    """""

    # Set the parameters
    a = 1  # NBDI
    b = 1  # PI

    if need_cloud_filter == "yes":
        print("Calculating custom water index...")
        UWI = ((a * PI) + (b * BU)) - cloud_mask
        UWI[UWI < 0] = 0
        UWI[UWI > 0] = 1
        plt.imshow(UWI, cmap="Blues")
        plt.title("Custom Water Index")
        plt.show()
        plt.imsave("/Users/tj/PycharmProjects/DataLabelling/Index_samples/UWI.png", UWI)
    else:
        print("Calculating custom water index...")
        UWI = ((a * PI) + (b * BU))
        UWI[UWI < 0] = 0
        UWI[UWI > 0] = 1
        plt.imshow(UWI, cmap="Blues")
        plt.title("Custom Water Index")
        plt.show()
        plt.imsave("/Users/tj/PycharmProjects/DataLabelling/Index_samples/UWI.png", UWI)

    """""
    White Roof Filter
    """""

    need_roof_filer = input(str("need roof filter?: "))

    if need_roof_filer == "yes":
        print("Cutting out white roofs")
        RGB = red + green + blue / 3
        RGB[RGB > 0.2] = 1
        RGB[RGB < 0.2] = 0
        plt.imshow(RGB)
        plt.show()
        UWI = UWI - RGB
        UWI[UWI < 0] = 0
        plt.imshow(UWI, cmap="Blues")
        plt.title("Custom Water Index")
        plt.show()
        plt.imsave("/Users/tj/PycharmProjects/DataLabelling/Index_samples/UWI.png", UWI)

    radar_filter = input(str("Shall I apply the radar density filter?: "))

    """""
    SAR Filter
    """""
    if apply_sar == "yes":
        if radar_filter == "yes":
            polarisation = input(str("is VV or HH more suitable?: "))
            if polarisation == "VV":
                polarisation = sar_vv
            elif polarisation == "HH":
                polarisation = sar_hh
            else:
                print("no polarisation has been chosen")

            print("Applying the Radar Filter")
            UWI = (UWI - polarisation) / (UWI + polarisation)
            print(np.amax(UWI), np.amin(UWI))
            plt.imshow(UWI, cmap="Blues")
            plt.title("Custom Water Index_radar")
            plt.show()
            plt.imsave("/Users/tj/PycharmProjects/DataLabelling/Index_samples/UWI.png", UWI)




    if output == "yes":
        # NORMAL
        split_image(dim_pix=244, im=UWI, location=region, dtype=f"Mask",
                    filename=f"Region_{region_number}")
        split_image(dim_pix=244, im=TC, location=region, dtype=f"TC",
                    filename=f"Region_{region_number}")

        # Horizontal Flip
        TC_Hflip = np.flip(TC, 1)
        WI_Hflip = np.flip(UWI, 1)
        Hflip = "Hflip"
        split_image(dim_pix=244, im=WI_Hflip, location=region,
                    dtype=f"Mask", filename=f"Region_{region_number}_{Hflip}")
        split_image(dim_pix=244, im=TC_Hflip, location=region, dtype=f"TC",
                    filename=f"Region_{region_number}_{Hflip}")

        # Vertical Flip
        TC_Vflip = np.flip(TC, 0)
        WI_Vflip = np.flip(UWI, 0)
        Vflip = "Vflip"
        split_image(dim_pix=244, im=WI_Vflip, location=region,
                    dtype=f"Mask", filename=f"Region_{region_number}_{Vflip}")
        split_image(dim_pix=244, im=TC_Vflip, location=region, dtype=f"TC",
                    filename=f"Region_{region_number}_{Vflip}")

        # Blur filter
        TC_Blur = cv2.medianBlur(TC, 5)
        Blur = "Blur"
        split_image(dim_pix=244, im=UWI, location=region, dtype=f"Mask",
                    filename=f"Region_{region}_{Blur}")
        split_image(dim_pix=244, im=TC_Blur, location=region, dtype=f"TC",
                    filename=f"Region_{region}_{Blur}")

        # Noise Filter
        noise = sp_noise(TC, 0.05)
        TC_noise = noise + TC
        Noise = "Noise"
        split_image(dim_pix=244, im=UWI, location=region, dtype=f"Mask",
                    filename=f"Region_{region_number}_{Noise}")
        split_image(dim_pix=244, im=TC_noise, location=region, dtype=f"TC",
                    filename=f"Region_{region_number}_{Noise}")
    else:
        print("You asked for no outputs")
