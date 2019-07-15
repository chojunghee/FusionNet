import glob
import cv2
import numpy as np
# from multiprocessing import Pool
from torch.utils.data import Dataset
import torch
import os

from PIL import Image

class stanford_dataset(torch.utils.data.Dataset):
    
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.train_image_folder = 'train_images'
        self.train_label_folder = 'train_labels'
        self.test_image_folder = 'test_images'
        self.test_label_folder = 'test_labels'

        if self.train:
            input_path = os.path.join(self.root, self.train_image_folder)
            label_path = os.path.join(self.root, self.train_label_folder)

            self.images = [cv2.imread(file) for file in sorted(glob.glob(input_path+'/*'), key=lambda f: int(''.join(filter(str.isdigit, f))))]
            self.labels = [np.loadtxt(file, dtype='long') for file in sorted(glob.glob(label_path+'/*.regions.txt'), key=lambda f: int(''.join(filter(str.isdigit, f))))]
        
        else:
            path = os.path.join(self.root, self.test_image_folder)
            self.images = [cv2.imread(file) for file in sorted(glob.glob(path + '/*'), key=lambda f: int(''.join(filter(str.isdigit, f))))]
            self.filename = [os.path.basename(file) for file in sorted(glob.glob(path + '/*'), key=lambda f: int(''.join(filter(str.isdigit, f))))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if self.train:
            input, label = self.images[index], self.labels[index]

            # return PIL image
            input = Image.fromarray(np.uint8(input))
            
            if self.transform is not None:
                input = self.transform(input)
            return input, label     

        else:
            # return PIL image
            input, filename = Image.fromarray(np.uint8(self.images[index])), self.filename[index]

            if self.transform is not None:
                input = self.transform(input)
            return input, filename

