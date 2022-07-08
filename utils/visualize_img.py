import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_img():
    # Unet = '/home/ym/Desktop/research/titanserver/home/yli/DWI_project/DWI_new_process/checkpoints/Unet/visualization/tumor'
    # DUnet = '/home/ym/Desktop/research/titanserver/home/yli/DWI_project/DWI_new_process/checkpoints/DenseUnet/visualization/tumor'
    # DUnetADC = '/home/ym/Desktop/research/titanserver/home/yli/DWI_project/DWI_new_process/checkpoints/DenseADC/visualization/tumor'
    # DUnetDWI = '/home/ym/Desktop/research/titanserver/home/yli/DWI_project/DWI_new_process/checkpoints/DenseDWI/visualization/tumor'
    # DUnetBOTH = '/home/ym/Desktop/research/titanserver/home/yli/DWI_project/DWI_new_process/checkpoints/DenseBOTH_Weighted/visualization/tumor'
    # Att_Unet = '/home/ym/Desktop/research/titanserver/home/yli/DWI_project/DWI_new_process/checkpoints/Att_Unet/visualization/tumor'
    # FBPNet = '/home/ym/Desktop/research/titanserver/home/yli/DWI_project/DWI_new_process/checkpoints/FBPConvNet/visualization/tumor'
    # Down = '/home/ym/Desktop/research/titanserver/home/yli/DWI_project/DWI_new_process/checkpoints/DenseADC/visualization/tumor_4x'
    #
    # ours = '/home/ym/Desktop/research/titanserver/home/yli/DWI_project/DWI_new_process/checkpoints/DenseTransformer/visualization/tumor'


    Unet = '/home/yli/DWI_project/DWI_new_process/checkpoints/Unet/visualization/tumor'
    DUnet = '/home/yli/DWI_project/DWI_new_process/checkpoints/DenseUnet/visualization/tumor'
    DUnetADC = '/home/yli/DWI_project/DWI_new_process/checkpoints/DenseADC/visualization/tumor'
    DUnetDWI = '/home/yli/DWI_project/DWI_new_process/checkpoints/DenseDWI/visualization/tumor'
    DUnetBOTH = '/home/yli/DWI_project/DWI_new_process/checkpoints/DenseBOTH_Weighted/visualization/tumor'
    Att_Unet = '/home/yli/DWI_project/DWI_new_process/checkpoints/Att_Unet/visualization/tumor'
    FBPNet = '/home/yli/DWI_project/DWI_new_process/checkpoints/FBPConvNet/visualization/tumor'
    Down = '/home/yli/DWI_project/DWI_new_process/checkpoints/DenseADC/visualization/tumor_4x'

    ours = '/home/yli/DWI_project/DWI_new_process/checkpoints/DenseTransformer/visualization/tumor'


    name = 'M97_cx_image_data01.npy'


    orig_name = 'orig_' + name
    pred_name = 'pred_' + name
    mask_name = 'mask_' + name

    orig_adc = np.load(os.path.join(Unet, orig_name))
    mask = np.load(os.path.join(Unet, mask_name))

    Unet_pred_adc = np.load(os.path.join(Unet, pred_name)) #Unet
    DUnet_pred_adc = np.load(os.path.join(DUnet, pred_name)) #DUnet
    DUnetADC_pred_adc = np.load(os.path.join(DUnetADC, pred_name)) #DUnetADC
    DUnetDWI_pred_adc = np.load(os.path.join(DUnetDWI, pred_name)) #DUnetDWI
    DUnetBOTH_pred_adc = np.load(os.path.join(DUnetBOTH, pred_name)) #DUnetBOTH
    ours_pred_adc = np.load(os.path.join(ours, pred_name)) #ours
    Att_pred_adc = np.load(os.path.join(Att_Unet, pred_name)) #Att_Unet
    FBP_pred_adc = np.load(os.path.join(FBPNet, pred_name))
    Down_pred_adc = np.load(os.path.join(Down, pred_name)) # Downsampled

    # Down_pred_adc = (Down_pred_adc - Down_pred_adc.min()) / (Down_pred_adc.max() - Down_pred_adc.min())
    # orig_adc = (orig_adc - orig_adc.min()) / (orig_adc.max() - orig_adc.min())
    # Att_pred_adc = (Att_pred_adc - Att_pred_adc.min()) / (Att_pred_adc.max() - Att_pred_adc.min())
    # FBP_pred_adc = (FBP_pred_adc - FBP_pred_adc.min()) / (FBP_pred_adc.max() - FBP_pred_adc.min())
    # Unet_pred_adc = (Unet_pred_adc - Unet_pred_adc.min())/ (Unet_pred_adc.max()-Unet_pred_adc.min())
    # DUnet_pred_adc = (DUnet_pred_adc - DUnet_pred_adc.min()) / (DUnet_pred_adc.max() - DUnet_pred_adc.min())
    # DUnetADC_pred_adc = (DUnetADC_pred_adc - DUnetADC_pred_adc.min()) / (DUnetADC_pred_adc.max() - DUnetADC_pred_adc.min())
    # DUnetDWI_pred_adc = (DUnetDWI_pred_adc - DUnetDWI_pred_adc.min()) / (DUnetDWI_pred_adc.max() - DUnetDWI_pred_adc.min())
    # DUnetBOTH_pred_adc = (DUnetBOTH_pred_adc - DUnetBOTH_pred_adc.min()) / (DUnetBOTH_pred_adc.max() - DUnetBOTH_pred_adc.min())
    # ours_pred_adc = (ours_pred_adc - ours_pred_adc.min()) / (ours_pred_adc.max() - ours_pred_adc.min())

    #%%
    plt.switch_backend('agg')
    fig, ((ax1, ax2, ax3, ax4, ax5, ax6, ax7),
    (ax8, ax9, ax10, ax11, ax12, ax13, ax14)) = plt.subplots(2, 7, figsize=(100, 100))

    vmax = 0.0032
    # vmax = 1

    im_in = ax1.imshow(orig_adc, cmap='hot', vmin=0, vmax=vmax)
    ax1.set_xticks([]), ax1.set_yticks([]), ax1.set_title("Original ADC", size=80)
    # divider = make_axes_locatable(ax1)
    # cax = divider.append_axes("right", size="5%", pad=2)
    # cbar = fig.colorbar(im_in, cax=cax)
    # cbar.ax.tick_params(labelsize=80)

    ax2.imshow(Unet_pred_adc, cmap='hot', vmin=0, vmax=vmax)
    ax2.set_xticks([]), ax2.set_yticks([]), ax2.set_title("Unet_pred_adc", size=80)

    ax3.imshow(DUnet_pred_adc, cmap='hot', vmin=0, vmax=vmax)
    ax3.set_xticks([]), ax3.set_yticks([]), ax3.set_title("DUnet_pred_adc", size=80)

    ax4.imshow(DUnetADC_pred_adc, cmap='hot', vmin=0, vmax=vmax)
    ax4.set_xticks([]), ax4.set_yticks([]), ax4.set_title("DUnetADC_pred_adc", size=80)

    ax5.imshow(DUnetDWI_pred_adc, cmap='hot', vmin=0, vmax=vmax)
    ax5.set_xticks([]), ax5.set_yticks([]), ax5.set_title("DUnetDWI_pred_adc", size=80)

    ax6.imshow(DUnetBOTH_pred_adc, cmap='hot', vmin=0, vmax=vmax)
    ax6.set_xticks([]), ax6.set_yticks([]), ax6.set_title("DUnetBOTH_pred_adc", size=80)

    ax7.imshow(ours_pred_adc, cmap='hot', vmin=0, vmax=vmax)
    ax7.set_xticks([]), ax7.set_yticks([]), ax7.set_title("ours_pred_adc", size=80)

    im_in = ax8.imshow(np.abs(orig_adc - orig_adc), cmap='jet', vmin=0, vmax=vmax)
    ax8.set_xticks([]), ax8.set_yticks([])
    # divider = make_axes_locatable(ax8)
    # cax = divider.append_axes("right", size="5%", pad=2)
    # cbar = fig.colorbar(im_in, cax=cax)
    # cbar.ax.tick_params(labelsize=80)

    ax9.imshow(np.abs(orig_adc - Unet_pred_adc), cmap='jet', vmin=0, vmax=vmax)
    ax9.set_xticks([]), ax9.set_yticks([])

    ax10.imshow(np.abs(orig_adc - DUnet_pred_adc), cmap='jet', vmin=0, vmax=vmax)
    ax10.set_xticks([]), ax10.set_yticks([])

    ax11.imshow(np.abs(orig_adc - DUnetADC_pred_adc), cmap='jet', vmin=0, vmax=vmax)
    ax11.set_xticks([]), ax11.set_yticks([])

    ax12.imshow(np.abs(orig_adc - DUnetDWI_pred_adc), cmap='jet', vmin=0, vmax=vmax)
    ax12.set_xticks([]), ax12.set_yticks([])

    ax13.imshow(np.abs(orig_adc - DUnetBOTH_pred_adc), cmap='jet', vmin=0, vmax=vmax)
    ax13.set_xticks([]), ax13.set_yticks([])

    ax14.imshow(np.abs(orig_adc - ours_pred_adc), cmap='jet', vmin=0, vmax=vmax)
    ax14.set_xticks([]), ax14.set_yticks([])

    # plt.savefig('/home/ym/Desktop/research/titanserver/home/yli/DWI_project/DWI_new_process/tumor_adc.jpg')
    plt.savefig('/home/yli/DWI_project/DWI_new_process/tumor_adc.jpg')

    # %%
    plt.switch_backend('agg')
    fig, ((ax1, ax2, ax3, ax4, ax5, ax6, ax7),
          (ax8, ax9, ax10, ax11, ax12, ax13, ax14)) = plt.subplots(2, 7, figsize=(100, 100))

    im_in = ax1.imshow(orig_adc, cmap='hot', vmin=0, vmax=vmax)
    ax1.set_xticks([]), ax1.set_yticks([]), ax1.set_title("Original ADC", size=80)
    # divider = make_axes_locatable(ax1)
    # cax = divider.append_axes("right", size="5%", pad=2)
    # cbar = fig.colorbar(im_in, cax=cax)
    # cbar.ax.tick_params(labelsize=80)

    ax2.imshow(Unet_pred_adc, cmap='hot', vmin=0, vmax=vmax)
    ax2.set_xticks([]), ax2.set_yticks([]), ax2.set_title("Unet_pred_adc", size=80)

    ax3.imshow(DUnet_pred_adc, cmap='hot', vmin=0, vmax=vmax)
    ax3.set_xticks([]), ax3.set_yticks([]), ax3.set_title("DUnet_pred_adc", size=80)

    ax4.imshow(Down_pred_adc, cmap='hot', vmin=0, vmax=vmax)
    ax4.set_xticks([]), ax4.set_yticks([]), ax4.set_title("Down_pred_adc", size=80)

    ax5.imshow(Att_pred_adc, cmap='hot', vmin=0, vmax=vmax)
    ax5.set_xticks([]), ax5.set_yticks([]), ax5.set_title("Att_pred_adc", size=80)

    ax6.imshow(FBP_pred_adc, cmap='hot', vmin=0, vmax=vmax)
    ax6.set_xticks([]), ax6.set_yticks([]), ax6.set_title("FBP_pred_adc", size=80)

    ax7.imshow(ours_pred_adc, cmap='hot', vmin=0, vmax=vmax)
    ax7.set_xticks([]), ax7.set_yticks([]), ax7.set_title("ours_pred_adc", size=80)

    im_in = ax8.imshow(np.abs(orig_adc - orig_adc), cmap='jet', vmin=0, vmax=vmax * 0.75)
    ax8.set_xticks([]), ax8.set_yticks([])
    # divider = make_axes_locatable(ax8)
    # cax = divider.append_axes("right", size="5%", pad=2)
    # cbar = fig.colorbar(im_in, cax=cax)
    # cbar.ax.tick_params(labelsize=80)

    ax9.imshow(np.abs(orig_adc - Unet_pred_adc), cmap='jet', vmin=0, vmax=vmax * 0.75)
    ax9.set_xticks([]), ax9.set_yticks([])

    ax10.imshow(np.abs(orig_adc - DUnet_pred_adc), cmap='jet', vmin=0, vmax=vmax * 0.75)
    ax10.set_xticks([]), ax10.set_yticks([])

    ax11.imshow(np.abs(orig_adc - Down_pred_adc), cmap='jet', vmin=0, vmax=vmax * 0.75)
    ax11.set_xticks([]), ax11.set_yticks([])

    ax12.imshow(np.abs(orig_adc - Att_pred_adc), cmap='jet', vmin=0, vmax=vmax * 0.75)
    ax12.set_xticks([]), ax12.set_yticks([])

    ax13.imshow(np.abs(orig_adc - FBP_pred_adc), cmap='jet', vmin=0, vmax=vmax * 0.75)
    ax13.set_xticks([]), ax13.set_yticks([])

    ax14.imshow(np.abs(orig_adc - ours_pred_adc), cmap='jet', vmin=0, vmax=vmax * 0.75)
    ax14.set_xticks([]), ax14.set_yticks([])

    # plt.savefig('/home/ym/Desktop/research/titanserver/home/yli/DWI_project/DWI_new_process/tumor_adc.jpg')
    plt.savefig('/home/yli/DWI_project/DWI_new_process/tumor_adc_SOTAs.jpg')


    ###
    fig, ((ax1, ax2, ax3, ax4, ax5, ax6, ax7),
          (ax8, ax9, ax10, ax11, ax12, ax13, ax14)) = plt.subplots(2, 7, figsize=(100, 100))

    ax1.imshow(orig_adc * mask, cmap='hot', vmin=0, vmax=vmax)
    ax1.set_xticks([]), ax1.set_yticks([]), ax1.set_title("Original ADC", size=80)

    ax2.imshow(Unet_pred_adc * mask, cmap='hot', vmin=0, vmax=vmax)
    ax2.set_xticks([]), ax2.set_yticks([]), ax2.set_title("Unet_pred_adc", size=80)

    ax3.imshow(DUnet_pred_adc * mask, cmap='hot', vmin=0, vmax=vmax)
    ax3.set_xticks([]), ax3.set_yticks([]), ax3.set_title("DUnet_pred_adc", size=80)

    ax4.imshow(DUnetADC_pred_adc * mask, cmap='hot', vmin=0, vmax=vmax)
    ax4.set_xticks([]), ax4.set_yticks([]), ax4.set_title("DUnetADC_pred_adc", size=80)

    ax5.imshow(DUnetDWI_pred_adc * mask, cmap='hot', vmin=0, vmax=vmax)
    ax5.set_xticks([]), ax5.set_yticks([]), ax5.set_title("DUnetDWI_pred_adc", size=80)

    ax6.imshow(DUnetBOTH_pred_adc * mask, cmap='hot', vmin=0, vmax=vmax)
    ax6.set_xticks([]), ax6.set_yticks([]), ax6.set_title("DUnetBOTH_pred_adc", size=80)

    ax7.imshow(ours_pred_adc * mask, cmap='hot', vmin=0, vmax=vmax)
    ax7.set_xticks([]), ax7.set_yticks([]), ax7.set_title("ours_pred_adc", size=80)

    ax8.imshow(np.abs(orig_adc - orig_adc) * mask, cmap='jet', vmin=0, vmax=vmax)
    ax8.set_xticks([]), ax8.set_yticks([])

    im_in = ax9.imshow(np.abs(orig_adc - Unet_pred_adc) * mask, cmap='jet', vmin=0, vmax=vmax * 0.25)
    ax9.set_xticks([]), ax9.set_yticks([])
    # divider = make_axes_locatable(ax8)
    # cax = divider.append_axes("right", size="5%", pad=2)
    # cbar = fig.colorbar(im_in, cax=cax)
    # cbar.ax.tick_params(labelsize=80)

    ax10.imshow(np.abs(orig_adc - DUnet_pred_adc) * mask, cmap='jet', vmin=0, vmax=vmax * 0.25)
    ax10.set_xticks([]), ax10.set_yticks([])

    ax11.imshow(np.abs(orig_adc - DUnetADC_pred_adc) * mask, cmap='jet', vmin=0, vmax=vmax * 0.25)
    ax11.set_xticks([]), ax11.set_yticks([])

    ax12.imshow(np.abs(orig_adc - DUnetDWI_pred_adc) * mask, cmap='jet', vmin=0, vmax=vmax * 0.25)
    ax12.set_xticks([]), ax12.set_yticks([])

    ax13.imshow(np.abs(orig_adc - DUnetBOTH_pred_adc) * mask, cmap='jet', vmin=0, vmax=vmax * 0.25)
    ax13.set_xticks([]), ax13.set_yticks([])

    ax14.imshow(np.abs(orig_adc - ours_pred_adc) * mask, cmap='jet', vmin=0, vmax=vmax * 0.25)
    ax14.set_xticks([]), ax14.set_yticks([])

    # plt.savefig('/home/ym/Desktop/research/titanserver/home/yli/DWI_project/DWI_new_process/tumor_adc.jpg')
    plt.savefig('/home/yli/DWI_project/DWI_new_process/tumor_adc_tumor.jpg')

    ###
    fig, ((ax1, ax2, ax3, ax4, ax5, ax6, ax7),
          (ax8, ax9, ax10, ax11, ax12, ax13, ax14)) = plt.subplots(2, 7, figsize=(100, 100))

    ax1.imshow(orig_adc * mask, cmap='hot', vmin=0, vmax=vmax)
    ax1.set_xticks([]), ax1.set_yticks([]), ax1.set_title("Original ADC", size=80)

    ax2.imshow(Unet_pred_adc * mask, cmap='hot', vmin=0, vmax=vmax)
    ax2.set_xticks([]), ax2.set_yticks([]), ax2.set_title("Unet_pred_adc", size=80)

    ax3.imshow(DUnet_pred_adc * mask, cmap='hot', vmin=0, vmax=vmax)
    ax3.set_xticks([]), ax3.set_yticks([]), ax3.set_title("DUnet_pred_adc", size=80)

    ax4.imshow(Down_pred_adc * mask, cmap='hot', vmin=0, vmax=vmax)
    ax4.set_xticks([]), ax4.set_yticks([]), ax4.set_title("Down_pred_adc", size=80)

    ax5.imshow(Att_pred_adc * mask, cmap='hot', vmin=0, vmax=vmax)
    ax5.set_xticks([]), ax5.set_yticks([]), ax5.set_title("Att_pred_adc", size=80)

    ax6.imshow(FBP_pred_adc * mask, cmap='hot', vmin=0, vmax=vmax)
    ax6.set_xticks([]), ax6.set_yticks([]), ax6.set_title("FBP_pred_adc", size=80)

    ax7.imshow(ours_pred_adc * mask, cmap='hot', vmin=0, vmax=vmax)
    ax7.set_xticks([]), ax7.set_yticks([]), ax7.set_title("ours_pred_adc", size=80)

    ax8.imshow(np.abs(orig_adc - orig_adc) * mask, cmap='jet', vmin=0, vmax=vmax)
    ax8.set_xticks([]), ax8.set_yticks([])

    im_in = ax9.imshow(np.abs(orig_adc - Unet_pred_adc) * mask, cmap='jet', vmin=0, vmax=vmax*0.25)
    ax9.set_xticks([]), ax9.set_yticks([])
    # divider = make_axes_locatable(ax8)
    # cax = divider.append_axes("right", size="5%", pad=2)
    # cbar = fig.colorbar(im_in, cax=cax)
    # cbar.ax.tick_params(labelsize=80)

    ax10.imshow(np.abs(orig_adc - DUnet_pred_adc) * mask, cmap='jet', vmin=0, vmax=vmax*0.25)
    ax10.set_xticks([]), ax10.set_yticks([])

    ax11.imshow(np.abs(orig_adc - Down_pred_adc) * mask, cmap='jet', vmin=0, vmax=vmax*0.25)
    ax11.set_xticks([]), ax11.set_yticks([])

    ax12.imshow(np.abs(orig_adc - Att_pred_adc) * mask, cmap='jet', vmin=0, vmax=vmax*0.25)
    ax12.set_xticks([]), ax12.set_yticks([])

    ax13.imshow(np.abs(orig_adc - FBP_pred_adc) * mask, cmap='jet', vmin=0, vmax=vmax*0.25)
    ax13.set_xticks([]), ax13.set_yticks([])

    ax14.imshow(np.abs(orig_adc - ours_pred_adc) * mask, cmap='jet', vmin=0, vmax=vmax*0.25)
    ax14.set_xticks([]), ax14.set_yticks([])

    # plt.savefig('/home/ym/Desktop/research/titanserver/home/yli/DWI_project/DWI_new_process/tumor_adc.jpg')
    plt.savefig('/home/yli/DWI_project/DWI_new_process/tumor_adc_tumor_SOTAs.jpg')

    print('saved images')


def plot_img_SOTA():

    organ = 'tumor'

    Unet = '/home/yli/DWI_project/DWI_new_process/checkpoints/Unet/visualization/' + organ
    DUnet = '/home/yli/DWI_project/DWI_new_process/checkpoints/DenseUnet/visualization/' + organ
    Att_Unet = '/home/yli/DWI_project/DWI_new_process/checkpoints/Att_Unet/visualization/' + organ
    FBPNet = '/home/yli/DWI_project/DWI_new_process/checkpoints/FBPConvNet/visualization/' + organ
    Down = '/home/yli/DWI_project/DWI_new_process/checkpoints/DenseTransformer/visualization/' + organ + '_down'
    ours = '/home/yli/DWI_project/DWI_new_process/checkpoints/DenseTransformer/visualization/' + organ

    # Unet = '/home/yli/DWI_project/DWI_new_process/checkpoints/Unet_8x/visualization/' + organ + '_down'
    # DUnet = '/home/yli/DWI_project/DWI_new_process/checkpoints/DenseUnet_8x/visualization/' + organ + '_down'
    # Att_Unet = '/home/yli/DWI_project/DWI_new_process/checkpoints/Att_Unet_8x/visualization/' + organ + '_down'
    # FBPNet = '/home/yli/DWI_project/DWI_new_process/checkpoints/FBPConvNet_8x/visualization/' + organ + '_down'
    # Down = '/home/yli/DWI_project/DWI_new_process/checkpoints/curve_fit_8x/visualization/' + organ + '_down'
    # ours = '/home/yli/DWI_project/DWI_new_process/checkpoints/DenseTransformer_8x/visualization/' + organ + '_down'

    name = 'ADC_M97_cx_image_data01.npy'
        # 'ADC_M239_cx_image_data02.npy' #-kidney
    #'ADC_M47_cx_image_data03.npy'-muscle
    #'ADC_M73_cx_image_data08.npy' -tumor


    orig_name = 'orig_' + name
    pred_name = 'pred_' + name
    mask_name = 'mask_' + name

    orig_adc = np.load(os.path.join(Unet, orig_name))
    mask = np.load(os.path.join(Unet, mask_name))

    Unet_pred_adc = np.load(os.path.join(Unet, pred_name)) #Unet
    DUnet_pred_adc = np.load(os.path.join(DUnet, pred_name)) #DUnet
    ours_pred_adc = np.load(os.path.join(ours, pred_name)) #ours
    Att_pred_adc = np.load(os.path.join(Att_Unet, pred_name)) #Att_Unet
    FBP_pred_adc = np.load(os.path.join(FBPNet, pred_name))
    Down_pred_adc = np.load(os.path.join(Down, pred_name)) # Downsampled

    # Down_pred_adc = (Down_pred_adc - Down_pred_adc.min()) / (Down_pred_adc.max() - Down_pred_adc.min())
    # orig_adc = (orig_adc - orig_adc.min()) / (orig_adc.max() - orig_adc.min())
    # Att_pred_adc = (Att_pred_adc - Att_pred_adc.min()) / (Att_pred_adc.max() - Att_pred_adc.min())
    # FBP_pred_adc = (FBP_pred_adc - FBP_pred_adc.min()) / (FBP_pred_adc.max() - FBP_pred_adc.min())
    # Unet_pred_adc = (Unet_pred_adc - Unet_pred_adc.min())/ (Unet_pred_adc.max()-Unet_pred_adc.min())
    # DUnet_pred_adc = (DUnet_pred_adc - DUnet_pred_adc.min()) / (DUnet_pred_adc.max() - DUnet_pred_adc.min())

    vmax = 0.0035

    # %%
    plt.switch_backend('agg')
    fig, ((ax1, ax2, ax3, ax4, ax5, ax6, ax7),
          (ax8, ax9, ax10, ax11, ax12, ax13, ax14)) = plt.subplots(2, 7, figsize=(100, 100))

    im_in = ax1.imshow(orig_adc, cmap='hot', vmin=0, vmax=vmax)
    ax1.set_xticks([]), ax1.set_yticks([]), ax1.set_title("Original ADC", size=80)
    # divider = make_axes_locatable(ax1)
    # cax = divider.append_axes("right", size="5%", pad=2)
    # cbar = fig.colorbar(im_in, cax=cax)
    # cbar.ax.tick_params(labelsize=80)

    ax2.imshow(Unet_pred_adc, cmap='hot', vmin=0, vmax=vmax)
    ax2.set_xticks([]), ax2.set_yticks([]), ax2.set_title("Unet_pred_adc", size=80)

    ax3.imshow(DUnet_pred_adc, cmap='hot', vmin=0, vmax=vmax)
    ax3.set_xticks([]), ax3.set_yticks([]), ax3.set_title("DUnet_pred_adc", size=80)

    ax4.imshow(Down_pred_adc, cmap='hot', vmin=0, vmax=vmax)
    ax4.set_xticks([]), ax4.set_yticks([]), ax4.set_title("Down_pred_adc", size=80)

    ax5.imshow(Att_pred_adc, cmap='hot', vmin=0, vmax=vmax)
    ax5.set_xticks([]), ax5.set_yticks([]), ax5.set_title("Att_pred_adc", size=80)

    ax6.imshow(FBP_pred_adc, cmap='hot', vmin=0, vmax=vmax)
    ax6.set_xticks([]), ax6.set_yticks([]), ax6.set_title("FBP_pred_adc", size=80)

    ax7.imshow(ours_pred_adc, cmap='hot', vmin=0, vmax=vmax)
    ax7.set_xticks([]), ax7.set_yticks([]), ax7.set_title("ours_pred_adc", size=80)

    im_in = ax8.imshow(np.abs(orig_adc - orig_adc), cmap='rainbow', vmin=0, vmax=vmax * 0.75)
    ax8.set_xticks([]), ax8.set_yticks([])
    # divider = make_axes_locatable(ax8)
    # cax = divider.append_axes("right", size="5%", pad=2)
    # cbar = fig.colorbar(im_in, cax=cax)
    # cbar.ax.tick_params(labelsize=80)

    ax9.imshow(np.abs(orig_adc - Unet_pred_adc), cmap='rainbow', vmin=0, vmax=vmax * 0.75)
    ax9.set_xticks([]), ax9.set_yticks([])

    ax10.imshow(np.abs(orig_adc - DUnet_pred_adc), cmap='rainbow', vmin=0, vmax=vmax * 0.75)
    ax10.set_xticks([]), ax10.set_yticks([])

    ax11.imshow(np.abs(orig_adc - Down_pred_adc), cmap='rainbow', vmin=0, vmax=vmax * 0.75)
    ax11.set_xticks([]), ax11.set_yticks([])

    ax12.imshow(np.abs(orig_adc - Att_pred_adc), cmap='rainbow', vmin=0, vmax=vmax * 0.75)
    ax12.set_xticks([]), ax12.set_yticks([])

    ax13.imshow(np.abs(orig_adc - FBP_pred_adc), cmap='rainbow', vmin=0, vmax=vmax * 0.75)
    ax13.set_xticks([]), ax13.set_yticks([])

    ax14.imshow(np.abs(orig_adc - ours_pred_adc), cmap='rainbow', vmin=0, vmax=vmax * 0.75)
    ax14.set_xticks([]), ax14.set_yticks([])

    # plt.savefig('/home/ym/Desktop/research/titanserver/home/yli/DWI_project/DWI_new_process/tumor_adc.jpg')
    # plt.savefig('/home/yli/DWI_project/DWI_new_process/' + organ + '_adc_SOTAs.jpg')
    plt.savefig('/home/yli/DWI_project/DWI_new_process/tumor_adc_SOTAs_good.jpg')



    ###
    fig, ((ax1, ax2, ax3, ax4, ax5, ax6, ax7),
          (ax8, ax9, ax10, ax11, ax12, ax13, ax14)) = plt.subplots(2, 7, figsize=(100, 100))

    ax1.imshow(orig_adc * mask, cmap='hot', vmin=0, vmax=vmax)
    ax1.set_xticks([]), ax1.set_yticks([]), ax1.set_title("Original ADC", size=80)

    ax2.imshow(Unet_pred_adc * mask, cmap='hot', vmin=0, vmax=vmax)
    ax2.set_xticks([]), ax2.set_yticks([]), ax2.set_title("Unet_pred_adc", size=80)

    ax3.imshow(DUnet_pred_adc * mask, cmap='hot', vmin=0, vmax=vmax)
    ax3.set_xticks([]), ax3.set_yticks([]), ax3.set_title("DUnet_pred_adc", size=80)

    ax4.imshow(Down_pred_adc * mask, cmap='hot', vmin=0, vmax=vmax)
    ax4.set_xticks([]), ax4.set_yticks([]), ax4.set_title("Down_pred_adc", size=80)

    ax5.imshow(Att_pred_adc * mask, cmap='hot', vmin=0, vmax=vmax)
    ax5.set_xticks([]), ax5.set_yticks([]), ax5.set_title("Att_pred_adc", size=80)

    ax6.imshow(FBP_pred_adc * mask, cmap='hot', vmin=0, vmax=vmax)
    ax6.set_xticks([]), ax6.set_yticks([]), ax6.set_title("FBP_pred_adc", size=80)

    ax7.imshow(ours_pred_adc * mask, cmap='hot', vmin=0, vmax=vmax)
    ax7.set_xticks([]), ax7.set_yticks([]), ax7.set_title("ours_pred_adc", size=80)

    im_in = ax8.imshow(np.abs(orig_adc - orig_adc) * mask, cmap='rainbow', vmin=0, vmax=vmax)
    ax8.set_xticks([]), ax8.set_yticks([])
    # divider = make_axes_locatable(ax8)
    # cax = divider.append_axes("right", size="5%", pad=2)
    # cbar = fig.colorbar(im_in, cax=cax)
    # cbar.ax.tick_params(labelsize=80)

    ax9.imshow(np.abs(orig_adc - Unet_pred_adc) * mask, cmap='rainbow', vmin=0, vmax=vmax*0.25)
    ax9.set_xticks([]), ax9.set_yticks([])

    ax10.imshow(np.abs(orig_adc - DUnet_pred_adc) * mask, cmap='rainbow', vmin=0, vmax=vmax*0.25)
    ax10.set_xticks([]), ax10.set_yticks([])

    ax11.imshow(np.abs(orig_adc - Down_pred_adc) * mask, cmap='rainbow', vmin=0, vmax=vmax*0.15)
    ax11.set_xticks([]), ax11.set_yticks([])

    ax12.imshow(np.abs(orig_adc - Att_pred_adc) * mask, cmap='rainbow', vmin=0, vmax=vmax*0.25)
    ax12.set_xticks([]), ax12.set_yticks([])

    ax13.imshow(np.abs(orig_adc - FBP_pred_adc) * mask, cmap='rainbow', vmin=0, vmax=vmax*0.25)
    ax13.set_xticks([]), ax13.set_yticks([])

    ax14.imshow(np.abs(orig_adc - ours_pred_adc) * mask, cmap='rainbow', vmin=0, vmax=vmax*0.25)
    ax14.set_xticks([]), ax14.set_yticks([])

    # plt.savefig('/home/ym/Desktop/research/titanserver/home/yli/DWI_project/DWI_new_process/tumor_adc.jpg')
    # plt.savefig('/home/yli/DWI_project/DWI_new_process/' + organ + '_adc_tumor_SOTAs.jpg')
    plt.savefig('/home/yli/DWI_project/DWI_new_process/tumor_adc_tumor_SOTAs_good.jpg')

    print('saved images')

if __name__ == '__main__':
    plot_img_SOTA()