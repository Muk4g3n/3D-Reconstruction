import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt 
import tensorflow as tf

import os
import cv2
from scipy.ndimage import rotate


# read Data from raw File

def read_data(file_path,array_shape=(1000, 1000,1000)):
    raw_data = np.fromfile(file_path, dtype=np.uint8)
    return raw_data.reshape(array_shape).astype("float32")

# -----------------------------------------------Preprocessing-------------------------------------



## create_sub_voxels
def extract_subvolumes(cube, subvol_size=250):
    subvolumes = []
    cube_size = cube.shape[0]
    for z in range(0, cube_size, subvol_size):
        for x in range(0, cube_size, subvol_size):
            for y in range(0, cube_size, subvol_size):
                subvol = cube[z:z+subvol_size, x:x+subvol_size, y:y+subvol_size]
                subvol = subvol.reshape((250,250,250,1))
                paddedSubVolumes = add_padding(subvol.shape[0],subvol)
                subvolumes.append(paddedSubVolumes)
                
    return np.array(subvolumes,dtype="float32")


## split images into blocks

def splitImg(img,numOfBlocks = 4):

    # Get the size of the image
    height, width = 1000,1000

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

## create an array from the blocks

def get_split_images(data):
    images = []
    
    for image in data:
        blocks = splitImg(image)
        
        for block in blocks:
            images.append(block)
       
    return np.array(images,dtype="float32").reshape((len(images), 250, 250))

# 

## add padding to the image

def add_padding(Range,data, pad_size = 3):
    
    final = []
    for i in range(Range):
        img = np.pad(data[i], ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='constant')
#         print(img.shape)
        final.append(img)

    return np.array(final)

## remove padding from the image

def remove_padding(data, pad_size = 3):
    unpadded_imgs = []
    for i in range(data.shape[0]):
        unpadded_img.append(data[i][pad_size:-pad_size, pad_size:-pad_size])
    return np.array(unpadded_imgs)

# Generate Features

def getFeatures(m,X):
    x1 = m.conv_1(X)
    x1d = m.batchNorm_1(x1)
    x1 = m.maxPool_1(x1d)


    x2 = m.conv_2(x1)
    x2d = m.batchNorm_2(x2)
    x2 = m.maxPool_2(x2d)


    x3 = m.conv_3(x2)
    x3d = m.batchNorm_3(x3)
    x3 = m.maxPool_3(x3d)



    x4 = m.conv_4(x3)
    x4d = m.batchNorm_4(x4)
    x4 = m.maxPool_4(x4d)


    x5 = m.conv_5(x4)
    x5d = m.batchNorm_5(x5)
    x5 = m.maxPool_5(x5d)


    flat = layers.Flatten()(x5)
    latentDim = m.latentDense(flat)
    encodedFeatures = m.batchNorm_6(latentDim)
    return encodedFeatures.numpy()