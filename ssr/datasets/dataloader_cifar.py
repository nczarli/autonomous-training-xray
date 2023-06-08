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


from torch.utils.data import Dataset, Subset, DataLoader
import torchvision.transforms as transforms
import json
import random
from torchvision.datasets.cifar import *
from typing import Any, Callable, Optional, Tuple
import torch


def unpickle(file):
    import dill as pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

class cifar_dataset(Dataset):


    def __init__(self, dataset, noisy_dataset, root_dir, noise_data_dir, transform,  noise_mode='sym',
                 dataset_mode='train', noise_ratio=0.5, open_ratio=0.5, noise_file=None):

        self.r = noise_ratio  # total noise ratio
        self.on = open_ratio  # proportion of open noise
        self.transform = transform
        self.mode = dataset_mode
        self.transition = {0: 0, 2: 0, 4: 7, 7: 7, 1: 1, 9: 1, 3: 5, 5: 3, 6: 6, 8: 8}  # class transition for asymmetric noise
        self.open_noise = None
        self.closed_noise = None
        
        self.train_size = 7534
        self.test_size = 1510


        if self.mode == 'test':
            cifar_dic = unpickle('%s\\test_batch' % root_dir)
            self.cifar_data = cifar_dic['data']
            self.test_size = len(self.cifar_data)
            if not isinstance(self.cifar_data, np.ndarray):
                self.cifar_data = np.array(self.cifar_data)
           
            print(self.test_size, 'test_size')
            # reshape from 3072 to 32*32*3
            self.cifar_data = self.cifar_data.reshape((self.test_size, 3, 32, 32))
            self.cifar_data = self.cifar_data.transpose((0, 2, 3, 1))


            self.cifar_label = cifar_dic['labels']
            print(len(self.cifar_label), 'cifar_label')
            

        elif self.mode == 'train':
   
            cifar_data = []
            cifar_label = []
            for n in range(1, 6):
                dpath = '%s\\data_batch_%d' % (root_dir, n)
                data_dic = unpickle(dpath)
                cifar_data.append(data_dic['data'])
                cifar_label = cifar_label + data_dic['labels']
                
            if not isinstance(cifar_data, np.ndarray):
                cifar_data = np.array(cifar_data)
            self.cifar_data = np.concatenate(cifar_data)
            self.train_size = len(self.cifar_data)
            self.cifar_data = self.cifar_data.reshape((self.train_size, 3, 32, 32))
            self.cifar_data = self.cifar_data.transpose((0, 2, 3, 1))


            print(len(cifar_label), 'cifar_label')
    
            self.clean_label = cifar_label
            if noisy_dataset == 'imagenet32':
                noise_data = None
            else:
                # noise_data = unpickle('%s/cifar-100-python/train' % noise_data_dir)['data'].reshape((self.train_size, 3, 32, 32)).transpose(
                #     (0, 2, 3, 1))
                noise_data = self.cifar_data

            # if os.path.exists(noise_file):
            #     noise = json.load(open(noise_file, "r"))
            #     noise_labels = noise['noise_labels']
            #     self.open_noise = noise['open_noise']
            #     # self.open_id = np.array(self.open_noise)[:, 0] if len(self.open_noise) !=0 else None
            #     self.closed_noise = noise['closed_noise']
            #     for cleanIdx, noisyIdx in noise['open_noise']:
            #         if noisy_dataset == 'imagenet32':
            #             self.cifar_data[cleanIdx] = np.asarray(
            #                 Image.open('{}/{}.png'.format(noise_data_dir, str(noisyIdx + 1).zfill(7)))).reshape((32, 32, 3))
            #             # set the groundtruth outliers label to be -1
            #             self.clean_label[cleanIdx] = 10
            #         else:
            #             self.cifar_data[cleanIdx] = noise_data[noisyIdx]
            #             self.clean_label[cleanIdx] = 10
            #     self.cifar_label = noise_labels
            # else:
            # inject noise
            noise_labels = []  # all labels (some noisy, some clean)
            print(self.train_size)
            train_size_int = int(self.train_size)
            print('test_size_int: ', train_size_int)
            idx = list(range(train_size_int))  # indices of cifar dataset
            random.shuffle(idx)
            num_total_noise = int(self.r * train_size_int)  # total amount of noise
            num_open_noise = int(self.on * num_total_noise)  # total amount of noisy/openset images
            print('Statistics of synthetic noisy CIFAR dataset: ', 'num of clean samples: ', train_size_int - num_total_noise,
                    ' num of closed-set noise: ', num_total_noise - num_open_noise, ' num of open-set noise: ', num_open_noise)
            # print(num_open_noise, num_total_noise)
            if noisy_dataset == 'imagenet32':  # indices of openset source images
                target_noise_idx = list(range(train_size_int))
            else:
                target_noise_idx = list(range(train_size_int))
            random.shuffle(target_noise_idx)
            self.open_noise = list(
                zip(idx[:num_open_noise], target_noise_idx[:num_open_noise]))  # clean sample -> openset sample mapping
            self.closed_noise = idx[num_open_noise:num_total_noise]  # closed set noise indices
            # populate noise_labels
            for i in range(train_size_int):
                if i in self.closed_noise:
                    if noise_mode == 'sym':
                        if dataset == 'cifar10':
                            noiselabel = random.randint(0, 1)
                    elif noise_mode == 'asym':
                        noiselabel = self.transition[cifar_label[i]]
                    noise_labels.append(noiselabel)
                else:
                    noise_labels.append(cifar_label[i])
            # populate openset noise images
            for cleanIdx, noisyIdx in self.open_noise:
                if noisy_dataset == 'imagenet32':
                    self.cifar_data[cleanIdx] = np.asarray(
                        Image.open('{}/{}.png'.format(noise_data_dir, str(noisyIdx + 1).zfill(7)))).reshape((32, 32, 3))
                    self.clean_label[cleanIdx] = train_size_int

                else:
                    self.cifar_data[cleanIdx] = noise_data[noisyIdx]
                    self.clean_label[cleanIdx] = train_size_int

            # write noise to a file, to re-use
            noise = {'noise_labels': noise_labels, 'open_noise': self.open_noise, 'closed_noise': self.closed_noise}
            print("save noise to %s ..." % noise_file)
            json.dump(noise, open(noise_file, "w"))
            self.cifar_label = noise_labels
            # self.open_id = np.array(self.open_noise)[:, 0] if len(self.open_noise) !=0 else None

        else:
            raise ValueError(f'Dataset mode should be train or test rather than {self.dataset_mode}!')


    def update_labels(self, new_label):
        self.cifar_label = new_label.cpu()

    def __getitem__(self, index):
        # print(index)
        if self.mode == 'train':
            img = self.cifar_data[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            target = self.cifar_label[index]
            clean_target = self.clean_label[index]
            
            return img, target, clean_target, index
        else:
            img = self.cifar_data[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            target = self.cifar_label[index]
            return img, target, index

    def __len__(self):
        return len(self.cifar_data)

    def get_noise(self):
        return (self.open_noise, self.closed_noise)

    def __repr__(self):
        return f'dataset_mode: {self.mode}, dataset number: {len(self)} \n'


# class cifar_dataset_fixmatch(cifar_dataset):
#     '''
#     Four different dataset modes:
#         all: all train dataset with noisy labels
#         subset: part of train dataset selected by subset_id with noisy labels
#         test: all test dataset with clean labels
#         memory: all train dataset with clean labels
#     '''
#
#     def __init__(self, dataset, root_dir, strong_transform, weak_transform, dataset_mode='train', noise_ratio=0.5, noise_mode='sym',
#                  noise_file=None):
#         '''
#         :param dataset: cifar10 or cifar100
#         :param root_dir: dataset directory
#         :param transform: transform made to images
#         :param dataset_mode: 'train' or 'test' or 'eval'
#         :param noise_ratio: generated sample noise ratio
#         :param noise_mode: 'sym' or 'asym' or 'mixed'
#         :param noise_file: generated noise file path
#         '''
#         # airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks
#         super(cifar_dataset_fixmatch, self).__init__(dataset, root_dir, None, dataset_mode, noise_ratio, noise_mode, noise_file)
#         self.strong_transform = strong_transform
#         self.weak_transform = weak_transform
#
#     def __getitem__(self, index):
#         img, target, index = self.cifar_data[index]
#         strong_img = self.strong_transform(img)
#         weak_img = self.weak_transform(img)
#         return weak_img, strong_img, target, index
#
#     def __len__(self):
#         return len(self.cifar_data)

# if __name__ == '__main__':
#     for mode in ['train', 'memory', 'test']:
#         testdata = cifar_dataset(dataset='cifar10',
#                                  root_dir='/Users/chen/Desktop/SSL_noise_reduction/dataset/cifar10',
#                                  dataset_mode=mode, transform=transforms.ToTensor(),
#                                  noise_file='/Users/chen/Desktop/SSL_noise_reduction/cifar10_noise_file.json')
#
#         print(len(testdata))
#         print(testdata[0][1:])
