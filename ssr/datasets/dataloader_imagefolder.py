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


from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random
from torchvision.datasets.cifar import *
from typing import Any, Callable, Optional, Tuple
import torch


def unpickle(file):
    import dill as pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class cifar_dataset(Dataset):
    def __init__(self, dataset_dir, transform, dataset_mode='train'):

        self.transform = transform
        self.mode = dataset_mode

        # load the cifar data
        cifar_dict = unpickle(dataset_dir)
        self.cifar_data = cifar_dict['data']
        self.size = len(self.cifar_data)

        # convert to numpy array
        if not isinstance(self.cifar_data, np.ndarray):
            self.cifar_data = np.array(self.cifar_data)

        print('INFO: data size: %d' % self.size)

        # reshape from (N, 3072) to (N, 3, 32, 32)
        self.cifar_data = self.cifar_data.reshape(self.size, 3, 32, 32)

        # transpose from (N, 3, 32, 32) to (N, 32, 32, 3)
        # to make it compatible with PIL
        self.cifar_data = self.cifar_data.transpose(0, 2, 3, 1)

        self.cifar_label = cifar_dict['labels']

    def update_labels(self, labels):
        self.cifar_label = labels.cpu()

    def __getitem__(self, index):
        img = self.cifar_data[index]
        img = Image.fromarray(img)
        img = self.transform(img)
        label = self.cifar_label[index]
        return img, label, index
    
    def __len__(self):
        return self.size
    
    def __repr__(self):
        return f'dataset_mode: {self.mode}, size: {self.size} \n'
    
    

