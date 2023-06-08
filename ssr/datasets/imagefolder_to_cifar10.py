"""
**********************************************************************************
 * Autonomous Training in X-Ray Imaging Systems
 * 
 * Training a deep learning model based on noisy labels from a rule based algorithm.
 * 
 * Copyright 2023 Nikodem Czarlinski
 * 
 * Licensed under the Attribution-NonCommercial 3.0 Unported (CC BY-NC 3.0)
 * (the "License"); you may not use this file except in compliance with 
 * the License. You may obtain a copy of the License at
 * 
 *     https://creativecommons.org/licenses/by-nc/3.0/
 * 
**********************************************************************************
"""


"""Convery imagefolder torch dataset to cifar10 torch dataset."""

import os
import pickle
import shutil
import sys
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def create_meta_data(class_labels, destination):
    """Create meta data for cifar10 dataset."""

    # Create meta data
    meta = {}
    meta["label_names"] = class_labels
    # meta["num_vis"] = 3072

    # Save meta data
    with open(os.path.join(destination, "batches.meta"), "wb") as f:
        pickle.dump(meta, f)

    return

def open_batch(destination, batch, dataset_type='train'):
    """Open batch file."""


    # Open batch file
    file = open(os.path.join(destination, dataset_type), "wb")

    return file

def close_batch(file, data, labels):
    """Close batch file."""
    # Pickle the 'data' and '
    pickle.dump({"data": data, "labels": labels}, file)

    # Close batch file
    file.close()

    return

def label_to_index(class_labels, folder):
    """Convert class label to index."""

    return class_labels.index(folder)

def image_to_byte_array(image, class_index, size):
    """Convert image to byte array."""

    # Open image
    img = Image.open(image)

    # Resize image
    img = img.resize(size)

    # Convert image to 3 dimensional array
    img_array = np.array(img)

    # Convert 3 dimensional array into row major order
    img_array_R = img_array[:, :, 0].flatten()
    img_array_G = img_array[:, :, 1].flatten()
    img_array_B = img_array[:, :, 2].flatten()
    class_index = [class_index]

    # Turn row-major array into bytes
    img_byte_array = np.array(
        list(img_array_R) + list(img_array_G) + list(img_array_B),
        np.uint8,
    )  # Turn into row-major byte array

    return img_byte_array




def process_image_dataset(source, destination, size, batch, dataset_type='train'):
    """Convery imagefolder torch dataset to cifar10 python dataset."""

    # Create destination directory
    Path(destination).mkdir(parents=True, exist_ok=True)

    # Create batches
    CURRENT_BATCH = 1

    file = open_batch(destination, CURRENT_BATCH, dataset_type)
    
    image_array = []
    class_array = []
    

    # Create class labels
    class_labels = []
    for folder in os.listdir(source):
        class_labels.append(folder)

    # Create meta data
    create_meta_data(class_labels, destination)

    # Create batches
    for folder in os.listdir(source):
        class_index = label_to_index(class_labels, folder)
        for image in tqdm(os.listdir(os.path.join(source, folder))):

            img_byte_array = image_to_byte_array(
                os.path.join(source, folder, image), class_index, size
            )
            image_array.append(img_byte_array)
            class_array.append(class_index)


    close_batch(file, image_array, class_array)

    print("Done")

    return


dataset_type = 'val' # 'train' or 'val'
use_rba = True

if use_rba:
    rule = 'rba'
else:
    rule = 'true'

# Set square dimensions of images
size = (32, 32)  # 32 by 32 pixels

# Set number of batches
batch = 5 # not used anymore

# Source of image dataset (Use absolute path)
# source = 'C:\\Users\\Nikodem\\Desktop\\Code\\Rule Based Algorithm\\extracted_cherry_images\\'+rule+'\\'+dataset_type

source = 'C:\\Users\\Nikodem\\Desktop\\Code\\SSR-own\\datasets\\square-circle-png\\' + dataset_type

# Destination of processed dataset (use absolute path)
# destination = 'C:\\Users\\Nikodem\\Desktop\\Code\\SSR-own\\datasets\\cifar-runtime-real'
destination = 'C:\\Users\\Nikodem\\Desktop\\Code\\SSR-own\\datasets\\square-circle-cifar'

# Process dataset
process_image_dataset(source, destination, size, batch, dataset_type=dataset_type)




