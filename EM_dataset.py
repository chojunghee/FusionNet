import glob
import cv2
import numpy as np
# from multiprocessing import Pool
from torch.utils.data import Dataset
import torch
import os

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from scipy import interpolate
from PIL import Image

class EM_dataset(torch.utils.data.Dataset):
    
    def __init__(self, root, sigma, train=True, color=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.sigma = sigma
        self.data = []
        self.input_folder = 'input'
        self.gt_folder = 'gt_files'
        self.test_folder = 'test'

        if self.train:
            input_path = os.path.join(self.root, self.input_folder)
            gt_path    = os.path.join(self.root, self.gt_folder)
            self.data = datagenerator(input_path, gt_path, self.sigma)
        
        else:
            path = os.path.join(self.root, self.test_folder)
            images = [cv2.imread(file) for file in sorted(glob.glob(path + '/*'), key=lambda f: int(''.join(filter(str.isdigit, f))))]
            self.data = np.stack(images, axis=0)

        if color == False:
            self.data = np.expand_dims(self.data, axis=3)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.train:
            input, gt = np.split(self.data[index], 2, axis=1)

            # return PIL image
            input, gt = Image.fromarray(np.uint8(input)), Image.fromarray(np.uint8(gt))

            if self.transform is not None:
                input, gt = self.transform(input), self.transform(gt)
            return input, gt     

        else:
            # return PIL image
            input = Image.fromarray(np.uint8(self.data[index]))

            if self.transform is not None:
                input = self.transform(input)
            return input


def data_aug(img):
    # data augmentation using rotation and horizontal reflection
    augmented = []
    augmented.append(img)
    augmented.append(np.fliplr(img))
    augmented.append(np.rot90(img))
    augmented.append(np.fliplr(np.rot90(img)))
    augmented.append(np.rot90(img, k=2))
    augmented.append(np.fliplr(np.rot90(img, k=2)))
    augmented.append(np.rot90(img, k=3))
    augmented.append(np.fliplr(np.rot90(img, k=3)))
    return augmented


def datagenerator(input_data_dir, gt_data_dir, sigma):
    # get name list of all .jpg files
    input_file_list = sorted(glob.glob(input_data_dir+'/*'), key=lambda f: int(''.join(filter(str.isdigit, f))))  
    gt_file_list = sorted(glob.glob(gt_data_dir+'/*'), key=lambda f: int(''.join(filter(str.isdigit, f))))
    # initialize
    data = []
    # augment data
    for i in range(len(input_file_list)):
        input_img = cv2.imread(input_file_list[i])
        gt_img    = cv2.imread(gt_file_list[i])
        augmented_input, augmented_gt = data_aug(input_img), data_aug(gt_img)
        for j in range(len(augmented_input)):
            augmented_input[j] = augmented_input[j] + np.random.randn(*augmented_input[j].shape)*sigma  # add random noise
            data.append(np.concatenate((augmented_input[j], augmented_gt[j]), axis=1))
    #data = np.array(data, dtype='uint8')
    #data = data.reshape((data.shape[0]*data.shape[1], data.shape[2], data.shape[3], 3))
    np.stack(data, axis=0)
    print('training data finished')
    return data


def make_perturbed_grid(w, h, size=12):

    # generate random vector field on 12x12 grid
    vf_x = 0.1*(np.random.rand(size, size) - 0.5)
    vf_y = 0.1*(np.random.rand(size, size) - 0.5)
    vf_x[0,:] = vf_x[:,0] =  vf_x[-1,:] =  vf_x[:,-1] = 0   # flow at boundary is 0
    vf_y[0,:] = vf_y[:,0] =  vf_y[-1,:] =  vf_y[:,-1] = 0

    # interpolate vector field of size x size on W x H grid 
    Y, X = np.linspace(0,h,size), np.linspace(0,w,size)
    fx = interpolate.interp2d(X,Y,vf_x)
    fy = interpolate.interp2d(X,Y,vf_y)

    new_X = np.arange(w)
    new_Y = np.arange(h)

    # get interpolated vector field
    new_vf_x = fx(new_X,new_Y)
    new_vf_y = fy(new_X,new_Y)

    # resulted random vector field
    vf = torch.tensor(np.stack([new_vf_x, new_vf_y], axis=2), dtype=torch.float32)
    
    # To use nn.functional.grid_sample, make a normalized grid whose left top coordinate is (-1,-1) and
    # right bottom coordinate is (1, 1)
    xv, yv = torch.meshgrid([torch.linspace(-1,1,w), torch.linspace(-1,1,h)])
    grid = torch.stack((yv,xv),dim=2)

    # perturb the grid by random vector field
    out = grid + vf
    return out