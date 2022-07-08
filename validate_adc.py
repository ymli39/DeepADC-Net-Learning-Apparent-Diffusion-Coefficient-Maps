#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 18:33:23 2020

@author: ym
"""

import os
import torch as t
from config_adc_train import opt
from data.DWI_loader_train import DWI_loader
from torch.utils.data import DataLoader
from models.DenseUnet_4x_MHSA import Backbone
from tqdm import tqdm
from torchnet import meter
import numpy as np
import pdb
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.metrics import structural_similarity as ssim


def validate(**kwargs):
    opt._parse(kwargs)

    model = Backbone()

    load_model = os.path.join(opt.load_model_path, opt.load_saved_model)
    if load_model:
        checkpoint = t.load(load_model)
        model.load_state_dict(checkpoint['state_dict'])
        print('finsihed loading model: ' + load_model)
    model.to(opt.device)

    test_data = DWI_loader(opt.list_dir, opt.data_dir, opt.data_4x_dir, opt.adc_dir, opt.adc_4x_dir, 'vali')
    test_dataloader = DataLoader(test_data, opt.val_batch_size,
                                 shuffle=False, num_workers=opt.num_workers)

    model.eval()

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)

    save_visual = os.path.join(opt.load_model_path, 'visualization_train/')
    if not os.path.exists(save_visual):
        os.makedirs(save_visual)

    tbar = tqdm(test_dataloader)

    total_corr = []
    total_nmse = []
    total_psnr = []
    total_ssim = []
    ##############################################################################

    for ii, sample_batched in enumerate(tbar):
        mask = sample_batched['mask']
        down_imgs = sample_batched['down_img']
        orig_adcs = sample_batched['orig_adc']
        down_adcs = sample_batched['down_adc']
        img_idx = sample_batched['img_idx']
        img_idx = img_idx.data.cpu().numpy()

        input_ = t.cat((down_imgs, down_adcs), 1)

        input_ = input_.to(opt.device).float()

        pred_ADC, pred_S0 = model(input_)

        ##############################################################################
        pred_ADC_npy = np.squeeze(pred_ADC.data.cpu().numpy(), axis=1)
        orig_ADC_npy = np.squeeze(orig_adcs.data.cpu().numpy(), axis=1)
        mask = np.squeeze(mask.data.cpu().numpy(), axis=1)

        mask[orig_ADC_npy < 0.01] = 0

        pred_ADC_npy = pred_ADC_npy * 0.0032
        orig_ADC_npy = orig_ADC_npy * 0.0032

        ##############################################################################
        total_corr.append(np.mean(b_corf(orig_ADC_npy, pred_ADC_npy, mask)))
        total_nmse.append(np.mean(b_nmse(orig_ADC_npy, pred_ADC_npy, mask)))
        total_psnr.append(np.mean(b_psnr(orig_ADC_npy, pred_ADC_npy, mask)))
        total_ssim.append(np.mean(b_ssim(orig_ADC_npy, pred_ADC_npy, mask)))
        ##############################################################################

        name = sample_batched['name']
        for jj in range(len(name)):
            name_list = name[jj]

            pred_ADC_npy = np.squeeze(pred_ADC[jj].cpu().data.numpy())
            orig_adcs_npy = np.squeeze(orig_adcs[jj].cpu().data.numpy())

            plot_all = False
            if plot_all:
                plt.switch_backend('agg')
                plt.figure(dpi=600)

                pred_ADC_npy = pred_ADC_npy * 0.0032
                orig_adcs_npy = orig_adcs_npy * 0.0032

                np.save(save_visual + '/pred_ADC_' + name_list + str(img_idx[jj]) + '.npy', pred_ADC_npy)

                # save intermediate ADC results
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(100, 100))

                im_in = ax1.imshow(pred_ADC_npy, "hot", vmin=0, vmax=0.0035)
                ax1.set_xticks([]), ax1.set_yticks([]), ax1.set_title("Reconstructed ADC", size=80)
                divider = make_axes_locatable(ax1)
                cax = divider.append_axes("right", size="5%", pad=2)
                cbar = fig.colorbar(im_in, cax=cax)
                cbar.ax.tick_params(labelsize=80)

                im_in = ax2.imshow(orig_adcs_npy, "hot", vmin=0, vmax=0.0035)
                ax2.set_xticks([]), ax2.set_yticks([]), ax2.set_title("Full sampled ADC", size=80)
                divider = make_axes_locatable(ax2)
                cax = divider.append_axes("right", size="5%", pad=2)
                cbar = fig.colorbar(im_in, cax=cax)
                cbar.ax.tick_params(labelsize=80)

                plt.savefig(os.path.join(save_visual + '/' + name_list + str(img_idx[jj]) + 'adc.jpg'))

    total_corr = np.asarray(total_corr)
    total_nmse = np.asarray(total_nmse)
    total_psnr = np.asarray(total_psnr)
    total_ssim = np.asarray(total_ssim)

    print("mean: overall coeff {},\n nmse {},\n psnr {},\n ssim {}.".format(np.mean(total_corr),
                                                                            np.mean(total_nmse),
                                                                            np.mean(total_psnr),
                                                                            np.mean(total_ssim)))
    print("std: overall coeff {},\n nmse {},\n psnr {},\n ssim {}.".format(np.std(total_corr),
                                                                           np.std(total_nmse),
                                                                           np.std(total_psnr),
                                                                           np.std(total_ssim)))


def tv_loss(img, tv_weight=1):
    """
    Compute total variation loss.
    Inputs:
    - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.
    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """

    w_variance = t.sum(t.pow(img[:, :, :, :-1] - img[:, :, :, 1:], 2))
    h_variance = t.sum(t.pow(img[:, :, :-1, :] - img[:, :, 1:, :], 2))
    loss = tv_weight * (h_variance + w_variance)
    return loss


def b_corf(gt, pred, mask):
    ''' Compute Correlation coefficient'''
    batch_size = gt.shape[0]
    batch_corf = []
    for i in range(batch_size):
        cur_gt = gt[i].flat[np.where(mask[i].flat == 1)]
        cur_pred = pred[i].flat[np.where(mask[i].flat == 1)]
        cm = np.corrcoef(cur_pred, cur_gt)
        batch_corf.append(cm[0, 1])
    return np.asarray(batch_corf)


def b_nmse(gt, pred, mask):
    """ Compute Normalized Mean Squared Error (NMSE) """
    batch_size = gt.shape[0]
    batch_nmse = []
    for i in range(batch_size):
        cur_gt = gt[i].flat[np.where(mask[i].flat == 1)]
        cur_pred = pred[i].flat[np.where(mask[i].flat == 1)]
        batch_nmse.append(np.linalg.norm(cur_gt - cur_pred) ** 2 / np.linalg.norm(cur_gt) ** 2)
    return np.asarray(batch_nmse)


def b_psnr(gt, pred, mask):
    """ Compute Peak Signal to Noise Ratio metric (PSNR) """
    batch_size = gt.shape[0]
    batch_psnr = []
    for i in range(batch_size):
        cur_gt = gt[i].flat[np.where(mask[i].flat == 1)]
        cur_pred = pred[i].flat[np.where(mask[i].flat == 1)]
        mse = ((cur_gt - cur_pred) ** 2).mean()
        max_i = cur_gt.max()
        s_psnr = 10 * np.log10((max_i ** 2) / mse)
        batch_psnr.append(s_psnr)
    return np.asarray(batch_psnr)


def b_ssim(gt, pred, mask):
    """ Compute Structural Similarity Index Metric (SSIM). """
    batch_size = gt.shape[0]
    batch_ssim = []
    for i in range(batch_size):
        cur_gt = gt[i].flat[np.where(mask[i].flat == 1)]
        cur_pred = pred[i].flat[np.where(mask[i].flat == 1)]
        s_ssim = ssim(cur_gt * 10, cur_pred * 10)
        batch_ssim.append(s_ssim)
    return np.asarray(batch_ssim)


if __name__ == '__main__':
    validate()