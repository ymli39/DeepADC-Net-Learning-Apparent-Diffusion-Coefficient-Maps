#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 00:02:08 2020

@author: ym
"""

import numpy as np
from torch.utils.data import Dataset
import torch
import os
import pdb
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.measure import label

'''
This function reads radial sampled DWI MRI data and outputs the original resoution and the down-sampled resoution image
output image is stacked in dimension: batch*Channel*H*W*2, where 2 is real and imaginary parts

no use of adc_dir
'''

class DWI_loader(Dataset):
    
    def __init__(self, list_dir, data_4x_dir, adc_4x_dir, phase):

        list_name = phase + '_list'
        data_list = os.path.join(list_dir, list_name)
        
        self.image_slices = []
        self.down_names = []
        
        self.down_adcs = []
        with open(data_list, 'r') as f:
            for line in f:
                img = np.load(os.path.join(data_4x_dir, line.rstrip() + '_cx_image_data_downsampled.npy'))
                dim, _, _, _ = img.shape
                for i in range(dim):
                    self.down_names.append(os.path.join(data_4x_dir, line.rstrip() + '_cx_image_data_downsampled.npy'))
                    self.down_adcs.append(os.path.join(adc_4x_dir, line.rstrip() + '_Diffusion_Fits_2param.npy'))
                    self.image_slices.append(i)

        print('Finished initializing ' + phase + ' data loader!')


    def __getitem__(self, idx):
        image_slice = self.image_slices[idx]
        down_name = self.down_names[idx]
        down_adc_name = self.down_adcs[idx]

        down_img = np.absolute(np.load(down_name)[image_slice])
        down_adc = np.load(down_adc_name)[1][image_slice]

        cur_name = down_name.split('/')[-1][:-4] + "{:02d}".format(image_slice)
        
        upper_thres = 0.0032
        lower_thres = 0.000032
        
        down_adc[down_adc>=upper_thres] = 0
        down_adc = ndimage.median_filter(down_adc, size=3)

        #normalize simulated data
        down_b1 = down_img[0, :, :]
        down_b1 = np.reshape(down_b1, (96 * 96, 1))
        temp_simu = np.unique(down_b1)
        down_b1_max = temp_simu[int(len(temp_simu) * 0.99)]
        
        down_img_map =np.clip(down_img, 0, down_b1_max)
        down_img_map = down_img_map/down_b1_max
        
        
        down_adc = (down_adc - np.min(down_adc)) / (np.max(down_adc) - np.min(down_adc))
        down_adc = down_adc[np.newaxis,:,:]
        
        sample = {'b1_max': down_b1_max,
                  'down_img': torch.from_numpy(down_img_map),
                  'down_adc': torch.from_numpy(down_adc),
                  'name': cur_name, 'img_idx': image_slice}
        
        return sample
        
        
        
    def __len__(self):
        return len(self.down_names)
        
    