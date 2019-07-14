import glob
import cv2
import numpy as np
import torch
# from multiprocessing import Pool
from torch.utils.data import Dataset
import torch
import os
from PIL import Image
import sys

sys.path.insert(0, '../data/coco/cocoapi-master/PythonAPI')

from pycocotools.coco import COCO
import skimage.io as io
import matplotlib.pyplot as plt
import pylab

class COCO_dataset(torch.utils.data.Dataset):
    
    def __init__(self, root, train=True, color=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.train_folder = 'images/train2017'
        self.val_folder = 'images/val2017'
        self.test_folder = 'images/test2017'
        self.gt_train = 'annotations/instances_train2017.json'
        self.gt_val   = 'annotations/instances_val2017.json'

        if self.train:
            input_path = os.path.join(self.root, self.val_folder)
            gt_path    = os.path.join(self.root, self.gt_val)
            self.data, self.target = datagenerator(input_path, gt_path)
        
        else:
            path = os.path.join(self.root, self.test_folder)
            images = [cv2.imread(file) for file in sorted(glob.glob(path + '/*'), key=lambda f: int(''.join(filter(str.isdigit, f))))]
            self.data = images

        if color == False:
            self.data = np.expand_dims(self.data, axis=3)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.train:
            input, gt = self.data[index], self.target[index]

            # return PIL image
            input = Image.fromarray(np.uint8(input)), 

            if self.transform is not None:
                input = self.transform(input)
            return input, gt     

        else:
            # return PIL image
            input = Image.fromarray(np.uint8(self.data[index]))

            if self.transform is not None:
                input = self.transform(input)
            return input


def datagenerator(input_data_dir, gt_data_dir):
    
    coco = COCO(gt_data_dir)

    # get name list of all image files
    input_file_list = sorted(glob.glob(input_data_dir+'/*'), key=lambda f: int(''.join(filter(str.isdigit, f))))  
    # initialize
    data = []
    target = []
    for i in range(len(input_file_list)):
        input_img = cv2.imread(input_file_list[i])
        # get corresponding gt information
        ids = ''.join(list(map(lambda f: ''.join(filter(str.isdigit, f)), os.path.basename(input_file_list[i]))))
        img_id = reduce_zero(ids)
        gt        = coco.loadAnns(coco.getAnnIds(imgIds=img_id, iscrowd=None))

        data.append(input_img), target.append(gt)
    #data = np.array(data, dtype='uint8')
    #data = data.reshape((data.shape[0]*data.shape[1], data.shape[2], data.shape[3], 3))
    print('training data finished')
    return data, target


def reduce_zero(n):
    n = str(n)
    while n[0] == '0':
        n = n[1:]
    return int(n)


def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    return [data, target]