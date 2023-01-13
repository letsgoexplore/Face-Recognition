
import os
import random

import numpy as np
import torch
import torchvision.datasets as datasets
from PIL import Image
from torch.utils.data.dataset import Dataset

from .utils import cvtColor, preprocess_input, resize_image


def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

class FacenetDataset(Dataset):
    def __init__(self, input_shape, lines, num_classes, random):
        self.input_shape    = input_shape
        self.lines          = lines
        self.length         = len(lines)
        self.num_classes    = num_classes
        self.random         = random
        
        self.paths  = []
        self.labels = []

        self.load_dataset()
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        #------------------------------------#
        #   创建全为零的矩阵
        #------------------------------------#
        images = np.zeros((3, 3, self.input_shape[0], self.input_shape[1]))
        labels = np.zeros((3))

        #------------------------------#
        #   先获得两张同一个人的人脸
        #   用来作为anchor和positive sample
        #   方法是从labels中招具有两个及以上元素的label
        #------------------------------#
        c               = random.randint(0, self.num_classes - 1)
        selected_path   = self.paths[self.labels[:] == c]
        while len(selected_path) < 2:
            c               = random.randint(0, self.num_classes - 1)
            selected_path   = self.paths[self.labels[:] == c]

        #------------------------------------#
        #   随机选择两张
        #------------------------------------#
        image_indexes = np.random.choice(range(0, len(selected_path)), 2)
        #------------------------------------#
        #   打开图片并放入矩阵
        #------------------------------------#
        image = cvtColor(Image.open(selected_path[image_indexes[0]]))
        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        if self.rand()<.5 and self.random: 
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        image = resize_image(image, [self.input_shape[1], self.input_shape[0]], letterbox_image = True)
        image = preprocess_input(np.array(image, dtype='float32'))
        image = np.transpose(image, [2, 0, 1])
        images[0, :, :, :] = image
        labels[0] = c
        
        image = cvtColor(Image.open(selected_path[image_indexes[1]]))
        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        if self.rand()<.5 and self.random: 
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        image = resize_image(image, [self.input_shape[1], self.input_shape[0]], letterbox_image = True)
        image = preprocess_input(np.array(image, dtype='float32'))
        image = np.transpose(image, [2, 0, 1])
        images[1, :, :, :] = image
        labels[1] = c

        #------------------------------#
        #   取出另外一个人的人脸
        #------------------------------#
        different_c         = list(range(self.num_classes))
        different_c.pop(c)
        different_c_index   = np.random.choice(range(0, self.num_classes - 1), 1)
        current_c           = different_c[different_c_index[0]]
        selected_path       = self.paths[self.labels == current_c]
        while len(selected_path)<1:
            different_c_index   = np.random.choice(range(0, self.num_classes - 1), 1)
            current_c           = different_c[different_c_index[0]]
            selected_path       = self.paths[self.labels == current_c]

        #------------------------------#
        #   随机选择一张
        #------------------------------#
        image_indexes       = np.random.choice(range(0, len(selected_path)), 1)
        image               = cvtColor(Image.open(selected_path[image_indexes[0]]))
        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        if self.rand()<.5 and self.random: 
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        image = resize_image(image, [self.input_shape[1], self.input_shape[0]], letterbox_image = True)
        image = preprocess_input(np.array(image, dtype='float32'))
        image = np.transpose(image, [2, 0, 1])
        images[2, :, :, :]  = image
        labels[2]           = current_c

        return images, labels

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a
    
    def load_dataset(self):
        for path in self.lines:
            path_split = path.split(";")
            self.paths.append(path_split[1].split()[0])
            self.labels.append(int(path_split[0]))
        self.paths  = np.array(self.paths,dtype=object)
        self.labels = np.array(self.labels)


#---------------------------------------------------------#
# 供给torch.utils.data.DataLoader中collate_fn使用
# 作用是将每个batch里的数据转换成DataLoader需要的格式
#---------------------------------------------------------#       
def dataset_collate(batch):
    images = []
    labels = []
    for img, label in batch:
        images.append(img)
        labels.append(label)

    images1 = np.array(images)[:, 0, :, :, :]
    images2 = np.array(images)[:, 1, :, :, :]
    images3 = np.array(images)[:, 2, :, :, :]
    images = np.concatenate([images1, images2, images3], 0)
    
    labels1 = np.array(labels)[:, 0]
    labels2 = np.array(labels)[:, 1]
    labels3 = np.array(labels)[:, 2]
    labels = np.concatenate([labels1, labels2, labels3], 0)
    
    images  = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    labels  = torch.from_numpy(np.array(labels)).long()
    return images, labels


