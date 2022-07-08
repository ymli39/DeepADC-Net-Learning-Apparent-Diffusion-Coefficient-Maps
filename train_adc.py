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


def train(**kwargs):
    opt._parse(kwargs)

    # step1: configure model
    model = Backbone()
    
    if opt.load_saved_model:
        load_model = os.path.join(opt.load_model_path, opt.load_saved_model)
        checkpoint = t.load(load_model)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        best_error = checkpoint['best_error']
        print('finsihed loading model: ' + load_model)
    else:
        print('Initialize the model!')
        start_epoch = 0
        best_error = 0

    model.to(opt.device)
    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)
    
    # step2: data
    train_data = DWI_loader(opt.list_dir, opt.data_dir, opt.data_4x_dir, opt.adc_dir, opt.adc_4x_dir, 'train')
    val_data = DWI_loader(opt.list_dir, opt.data_dir, opt.data_4x_dir, opt.adc_dir, opt.adc_4x_dir, 'vali')
    train_dataloader = DataLoader(train_data, opt.batch_size,
                                  shuffle=True, num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data, opt.val_batch_size,
                                shuffle=False, num_workers=opt.num_workers)

    # step3: criterion and optimizer
    criterion = t.nn.L1Loss(reduction='mean')

    optimizer = t.optim.AdamW(model.parameters(), lr=opt.lr,
                             weight_decay=opt.weight_decay)
    
    # step4: meters
    loss_meter = meter.AverageValueMeter()

    save_visual = os.path.join(opt.load_model_path, 'visualization_train/')
    if not os.path.exists(save_visual):
        os.makedirs(save_visual)

    # train
    for epoch in range(start_epoch, opt.max_epoch):

        '''
        **********************************************************************************************************************
            start to train the model
        **********************************************************************************************************************
        '''
        tbar_train = tqdm(train_dataloader)
        train_loss = 0.0
        model.train()

        for ii, sample_batched in enumerate(tbar_train):
            # train model
            
            orig_imgs = sample_batched['orig_img']
            down_imgs = sample_batched['down_img']
            orig_adcs = sample_batched['orig_adc']
            down_adcs = sample_batched['down_adc']
            mask = sample_batched['mask']
            orig_S0 = sample_batched['orig_S0']

            input_ = t.cat((down_imgs, down_adcs), 1)

            orig_imgs = orig_imgs.to(opt.device).float()
            orig_adcs = orig_adcs.to(opt.device).float()
            input_ = input_.to(opt.device).float()
            orig_S0 = orig_S0.to(opt.device).float()

            optimizer.zero_grad()
            
            pred_ADC, pred_S0 = model(input_)

            score_adc = pred_ADC
            target_adc = orig_adcs
            
            ###################dwi###################################################
            b_array = opt.b_array.view(1,5,1,1).to(opt.device).float()
            fit_data = pred_S0 * t.exp(-b_array * pred_ADC.repeat(1,5,1,1) * 0.0032)

            score_dwi = fit_data # * 10000
            target_dwi = orig_imgs  # * 10000

            score_S0 = pred_S0
            target_S0 = orig_S0
            ###################dwi###################################################

            adc_loss = criterion(score_adc,target_adc)
            dwi_loss = criterion(score_dwi,target_dwi)
            S0_loss = criterion(score_S0,target_S0)

            loss = adc_loss + dwi_loss*0.5 + S0_loss*0.1

            loss.backward()
            optimizer.step()

            loss_meter.add(loss.item())
            train_loss += loss.item() / opt.batch_size
            tbar_train.set_description('Train loss: %.3f' % (train_loss / (ii + 1)))
            tbar_train.set_description('Min ADC: %.3f' % (t.min(pred_ADC)))

        print("==== Epoch [" + str(opt.max_epoch) + " / " + str(epoch) + "] DONE ====")
        print('Loss: %.3f' % train_loss)

        '''
        **********************************************************************************************************************
            start to validate the model
        **********************************************************************************************************************
        '''
        tbar_vali = tqdm(val_dataloader)
        model.eval()

        total_corr = []

        for ii, sample_batched in enumerate(tbar_vali):
            mask = sample_batched['mask']

            down_imgs = sample_batched['down_img']
            orig_adcs = sample_batched['orig_adc']
            down_adcs = sample_batched['down_adc']

            input_ = t.cat((down_imgs, down_adcs), 1)
            input_ = input_.to(opt.device).float()

            optimizer.zero_grad()
            pred_ADC, _ = model(input_)

            ##############################################################################
            pred_ADC_npy = np.squeeze(pred_ADC.data.cpu().numpy(), axis=1)  # /10000
            orig_ADC_npy = np.squeeze(orig_adcs.data.cpu().numpy(), axis=1)
            mask = np.squeeze(mask.data.numpy(), axis=1)
            
            mask[orig_ADC_npy<0.01] = 0

            total_corr.append(b_corf(orig_ADC_npy, pred_ADC_npy, np.squeeze(mask)))

            ##############################################################################
        #uncomment this part if want to see temp results
        # name = sample_batched['name']
        # for jj in range(len(name)):
        #     name_list = name[jj]
        #     img_idx = sample_batched['img_idx'][0]
        #     img_idx = img_idx.data.cpu().numpy()
        #
        #     pred_ADC_npy = np.squeeze(pred_ADC[jj].cpu().data.numpy())
        #     orig_adcs_npy = np.squeeze(orig_adcs[jj].cpu().data.numpy())
        #
        #     plt.switch_backend('agg')
        #     plt.figure(dpi=600)
        #
        #     pred_ADC_npy = pred_ADC_npy * 0.0032
        #     orig_adcs_npy = orig_adcs_npy * 0.0032
        #
        #     # save intermediate ADC results
        #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(100, 100))
        #
        #     im_in = ax1.imshow(pred_ADC_npy, "hot", vmin=0, vmax=0.0035)
        #     ax1.set_xticks([]), ax1.set_yticks([]), ax1.set_title("Reconstructed ADC", size=80)
        #     divider = make_axes_locatable(ax1)
        #     cax = divider.append_axes("right", size="5%", pad=2)
        #     cbar = fig.colorbar(im_in, cax=cax)
        #     cbar.ax.tick_params(labelsize=80)
        #
        #     im_in = ax2.imshow(orig_adcs_npy, "hot", vmin=0, vmax=0.0035)
        #     ax2.set_xticks([]), ax2.set_yticks([]), ax2.set_title("Full sampled ADC", size=80)
        #     divider = make_axes_locatable(ax2)
        #     cax = divider.append_axes("right", size="5%", pad=2)
        #     cbar = fig.colorbar(im_in, cax=cax)
        #     cbar.ax.tick_params(labelsize=80)
        #
        #     plt.savefig(os.path.join(save_visual + '/' + name_list + str(img_idx) + 'adc.jpg'))

        total_corr = np.concatenate(total_corr)
        print("overall coeff {}.".format(np.mean(total_corr)))
        avg_corr = np.mean(total_corr)

        save_model = os.path.join('./checkpoints/', opt.env)
        if not os.path.exists(save_model):
            os.makedirs(save_model)

        if avg_corr > best_error:
            t.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_error': avg_corr
            }, os.path.join(save_model, '4x_checkpoint.pth.tar'))
            print("save model on epoch %d" % epoch)
            best_error = avg_corr

            file = open(save_model + "/checkpoint_count.txt", "w")
            file.write(str(epoch) + ': overall coeff {}'.format(np.mean(total_corr)))
            file.close()

        for param_group in optimizer.param_groups:
            if epoch < 800:
                param_group['lr'] = opt.lr_list[epoch]
            else:
                param_group['lr'] = opt.lr_list[799]


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
    train()
