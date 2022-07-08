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

'''
This function reads radial sampled DWI MRI data and outputs the original resoution and the down-sampled resoution image
output image is stacked in dimension: batch*Channel*H*W*2, where 2 is real and imaginary parts

no use of adc_dir
'''

class DWI_loader(Dataset):
    
    def __init__(self, list_dir, data_dir, data_4x_dir, adc_dir, adc_4x_dir, phase):

        list_name = phase + '_list'
        data_list = os.path.join(list_dir, list_name)
        
        self.image_slices = []
        self.image_names = []
        self.down_names = []
        
        self.adc_names = []
        self.down_adcs = []
        with open(data_list, 'r') as f:
            for line in f:
                img = np.load(os.path.join(data_dir, line.rstrip() + '_cx_image_data.npy'))
                dim, _, _, _ = img.shape
                for i in range(dim):
                    self.image_names.append(os.path.join(data_dir, line.rstrip() + '_cx_image_data.npy'))
                    self.down_names.append(os.path.join(data_4x_dir, line.rstrip() + '_cx_image_data_downsampled.npy'))
                    self.adc_names.append(os.path.join(adc_dir, line.rstrip() + '_Diffusion_Fits_2param.npy'))
                    self.down_adcs.append(os.path.join(adc_4x_dir, line.rstrip() + '_Diffusion_Fits_2param.npy'))
                    self.image_slices.append(i)

        print('Finished initializing ' + phase + ' data loader!')


    def __getitem__(self, idx):
        image_slice = self.image_slices[idx]
        image_name = self.image_names[idx]
        down_name = self.down_names[idx]
        adc_name = self.adc_names[idx]
        down_adc_name = self.down_adcs[idx]

        orig_img = np.absolute(np.load(image_name)[image_slice])
        down_img = np.absolute(np.load(down_name)[image_slice])
        orig_adc = np.load(adc_name)[1][image_slice]
        down_adc = np.load(down_adc_name)[1][image_slice]

        orig_S0 = np.load(adc_name)[0][image_slice]/100
        
        cur_name = image_name.split('/')[-1][:-4] + "{:02d}".format(image_slice)
        
        #calculate mask
        upper_thres = 0.0032
        lower_thres = 0.000032
        
        orig_adc[orig_adc>=upper_thres] = 0
        orig_adc = ndimage.median_filter(orig_adc, size=3)

        down_adc[down_adc>=upper_thres] = 0
        down_adc = ndimage.median_filter(down_adc, size=3)

        bool_min = orig_adc>lower_thres
        bool_max = orig_adc<upper_thres
        mask2 = (bool_min * bool_max).astype(int)
        mask2 = mask2[np.newaxis,:,:]
        
        bool_b12 = orig_img[0,:,:] > orig_img[1,:,:]
        bool_b23 = orig_img[1,:,:] > orig_img[2,:,:]
        mask1 = (bool_b12 * bool_b23).astype(int)
        mask1 = mask1[np.newaxis,:,:]
        
    
        #normalize simulated data
        orig_b1 = orig_img[0,:,:]
        orig_b1 = np.reshape(orig_b1, (96*96, 1))
        temp_simu = np.unique(orig_b1)
        orig_b1_max = temp_simu[int(len(temp_simu)*0.99)]

        down_b1 = down_img[0, :, :]
        down_b1 = np.reshape(down_b1, (96 * 96, 1))
        temp_simu = np.unique(down_b1)
        down_b1_max = temp_simu[int(len(temp_simu) * 0.99)]
        
        orig_image_map =np.clip(orig_img, 0, orig_b1_max)
        orig_image_map = orig_image_map/orig_b1_max
        
        down_img_map =np.clip(down_img, 0, down_b1_max)
        down_img_map = down_img_map/down_b1_max
        
        orig_adc = (orig_adc - np.min(orig_adc)) / (np.max(orig_adc) - np.min(orig_adc))
        down_adc = (down_adc - np.min(down_adc)) / (np.max(down_adc) - np.min(down_adc))
        
        orig_adc = orig_adc[np.newaxis,:,:]
        down_adc = down_adc[np.newaxis,:,:]
        orig_S0 = orig_S0[np.newaxis,:,:]
        
        mask = mask1 * mask2
        
        sample = {'mask': torch.from_numpy(mask), 'b1_max': orig_b1_max,
                  'orig_img': torch.from_numpy(orig_image_map) , 'down_img': torch.from_numpy(down_img_map),
                  'orig_adc': torch.from_numpy(orig_adc), 'down_adc': torch.from_numpy(down_adc),
                  'orig_S0': torch.from_numpy(orig_S0),
                  'name': cur_name, 'img_idx': image_slice}
        
        return sample
        
        
        
    def __len__(self):
        return len(self.image_names)
        
    