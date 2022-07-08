#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 18:33:23 2020

@author: ym

the image with simualted down sampled is named as _4x, the image with real down sampled is named as _down
"""

import os
import torch as t
from config_adc_test import opt
from data.DWI_loader_test import DWI_loader
from torch.utils.data import DataLoader
from models.DenseUnet_4x_MHSA import Backbone
from tqdm import tqdm
import numpy as np
import pdb
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt



def validate(**kwargs):
    opt._parse(kwargs)

    model = Backbone()
    
    load_model = os.path.join(opt.load_model_path, opt.load_saved_model)
    if load_model:
        checkpoint = t.load(load_model)
        model.load_state_dict(checkpoint['state_dict'])
        print('finsihed loading model: ' + load_model)
    model.to(opt.device)

    test_data = DWI_loader(opt.list_dir, opt.data_4x_dir, opt.adc_4x_dir, 'vali')
    test_dataloader = DataLoader(test_data, opt.val_batch_size,
                                 shuffle=False, num_workers=opt.num_workers)

    model.eval()
    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)

    save_visual = os.path.join(opt.load_model_path, 'visualization_test/')

    if not os.path.exists(save_visual):
        os.makedirs(save_visual)

    tbar = tqdm(test_dataloader)

    for ii, sample_batched in enumerate(tbar):
        down_imgs = sample_batched['down_img']
        down_adcs = sample_batched['down_adc']
        name = sample_batched['name'][0][:-15]
        img_idx = sample_batched['img_idx']
        img_idx = img_idx.data.cpu().numpy()

        input_ = t.cat((down_imgs, down_adcs), 1)
        input_ = input_.to(opt.device).float()

        pred_ADC, pred_S0 = model(input_)

        ###########################model output-save ADC######################################
        pred_ADC_npy = np.squeeze(pred_ADC.data.cpu().numpy(), axis=1)

        pred_ADC_npy = pred_ADC_npy * 0.0032

        for ii in range(len(img_idx)):
            cur_idx = img_idx[ii]
            np.save(save_visual + '/pred_ADC_' + name + str(cur_idx) +  '.npy', pred_ADC_npy[ii])

            plot_all = False
            if plot_all:
                plt.switch_backend('agg')
                plt.figure(dpi=600)

                # save intermediate ADC results
                fig, ax1 = plt.subplots(1, 1, figsize=(50, 50))
                im_in = ax1.imshow(pred_ADC_npy[ii], "hot", vmin = 0, vmax = 0.0035)
                ax1.set_xticks([]), ax1.set_yticks([]), ax1.set_title("down_ADC", size=80)
                divider = make_axes_locatable(ax1)
                cax = divider.append_axes("right", size="5%", pad=2)
                cbar = fig.colorbar(im_in, cax=cax)
                cbar.ax.tick_params(labelsize=80)

                plt.savefig(os.path.join(save_visual + '/' + name + str(cur_idx) + 'adc.jpg'))
        ##############################################################################



if __name__ == '__main__':
    validate()
