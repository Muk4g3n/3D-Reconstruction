import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import tensorflow as tf

import os
from skimage.metrics import mean_squared_error
# import cv2
# from scipy.ndimage import rotate


# read Data from raw File

def read_data(file_path, array_shape=(1000, 1000, 1000)):
    raw_data = np.fromfile(file_path, dtype=np.uint8)
    return raw_data.reshape(array_shape).astype("float32")

# -----------------------------------------------Preprocessing-------------------------------------


# create_sub_voxels
def extract_subvolumes(cube, subvol_size=250):
    subvolumes = []
    cube_size = cube.shape[0]
    for z in range(0, cube_size, subvol_size):
        for x in range(0, cube_size, subvol_size):
            for y in range(0, cube_size, subvol_size):
                subvol = cube[z:z+subvol_size, x:x +
                              subvol_size, y:y+subvol_size]
                subvol = subvol.reshape((250, 250, 250, 1))
                paddedSubVolumes = add_padding(subvol.shape[0], subvol)
                subvolumes.append(paddedSubVolumes)

    return np.array(subvolumes, dtype="float32")


# split images into blocks

def splitImg(img, numOfBlocks=4):

    # Get the size of the image
    height, width = 1000, 1000

    # Define the size of each block
    block_size = (width // numOfBlocks, height // numOfBlocks)
    # Create a list to store the blocks
    blocks = []

    # Split the image into blocks
    for i in range(numOfBlocks):
        for j in range(numOfBlocks):
            x1, y1 = j * block_size[0], i * block_size[1]
            x2, y2 = x1 + block_size[0], y1 + block_size[1]
            block = img[y1:y2, x1:x2]
            blocks.append(block)
    return blocks

# create an array from the blocks


def get_split_images(data):
    images = []

    for image in data:
        blocks = splitImg(image)

        for block in blocks:
            images.append(block)

    return np.array(images, dtype="float32").reshape((len(images), 250, 250))

#

# add padding to the image


def add_padding(Range, data, pad_size=3):

    final = []
    for i in range(Range):
        img = np.pad(data[i], ((pad_size, pad_size),
                     (pad_size, pad_size), (0, 0)), mode='constant')
#         print(img.shape)
        final.append(img)

    return np.array(final)

# remove padding from the image


def remove_padding(data, pad_size=3):
    unpadded_imgs = []
    for i in range(data.shape[0]):
        unpadded_imgs.append(data[i][pad_size:-pad_size, pad_size:-pad_size])
    return np.array(unpadded_imgs)



<<<<<<< Updated upstream
#     x4 = m.conv_4(x3)
#     x4d = m.batchNorm_4(x4)
#     x4 = m.maxPool_4(x4d)
=======


def porosity(im):
    threshold = 0.5  # Adjust the threshold as needed

    # Calculate the porosity using NumPy operations
    porosity = 1 - np.mean(im <= threshold)
    return porosity
>>>>>>> Stashed changes


def calculate_mae(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("Both lists must have the same length.")
    
    n = len(list1)
    # mae = sum(abs(y1 - y2) for y1, y2 in zip(list1, list2)) / n
    mae = [abs(list1[i] - list2[i]) for i in range(n) ]
    return mae

def calculate_mse(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("Both lists must have the same length.")

<<<<<<< Updated upstream
#     flat = layers.Flatten()(x5)
#     latentDim = m.latentDense(flat)
#     encodedFeatures = m.batchNorm_6(latentDim)
#     return encodedFeatures.numpy()
=======
    mse = [mean_squared_error(img1, img2) for img1, img2 in zip(list1, list2)]
    return mse
>>>>>>> Stashed changes
