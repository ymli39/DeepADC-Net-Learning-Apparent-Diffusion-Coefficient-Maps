# HKS 11/27/20 Fit diffusion
# HKS 12/7/20 diffusion_fit_ALL: Batch processing, set limits on diffusion values, plot all images/b-values

print("Start program!")

import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
import math
from datetime import datetime
from scipy import signal
import cmath
from scipy.optimize import curve_fit
import matplotlib.backends.backend_pdf
import pandas as pd
import os

#from PIL import Image
# R-G-B-A (A = alpha or opacity: 1=opaque, 0=transparent)

# Define the diffusion function to fit
# def func(x, a, b, c):
#     return a * np.exp(-b * x) + c

def func(x, a, b):
    return a * np.exp(-b * x)
    
print("Start =", datetime.now())

xres = 96
yres = 96
bvalues = 5
b_array = np.array([24.25, 536.86, 1072.62, 1482.07, 2144.69]) # b-values


datalist = 'DIRECTORY/image_list' #directory of image_list
data_dir = 'DIRECTORY/4x_downsampled_data' #directory of dataset
save_dir = 'DIRECTORY/reconstructed_adcs_4x/' #directory of generated dataset
slice_table = 'DIRECTORY/DWI_dimensions.xlsx' #directory of excel table


namefile = open(datalist, 'r')
datanames = namefile.readlines()
dir_to_analyze = [f[:-1] for f in datanames]

#load excel to read slice numbers
dfs = pd.read_excel(slice_table, sheet_name='Sheet1')
img_num = dfs['M#'].values
img_num = [int(f[1:])-1 for f in img_num]
slice_num = dfs['slices']


if not os.path.exists(save_dir):
    os.makedirs(save_dir)
                    
for k in range(0, len(dir_to_analyze)):

    current_dir = os.path.join(data_dir, dir_to_analyze[k])
    input_file = current_dir + "_AllSlicesBvalues.bin"


    # out_file = save_dir + dir_to_analyze[k] + "_Diffusion_Fits_2param.npy"
    # if os.path.exists(out_file):
    #     print('skipped ' + dir_to_analyze[k])
    #     continue

    name_id = int(current_dir.split('/')[-1][1:])
    slices = int(slice_num[name_id-1])

    tmpdat = np.fromfile(input_file, dtype=np.float64, count = xres*yres*bvalues*slices)
    imgarr = np.reshape(tmpdat,(slices,bvalues,yres,xres))


    # Fit the diffusion

    fitval = np.zeros((4,slices,yres,xres)) # Fit all slices

    for sl in range(0, slices):

        print("Slice = ", sl)
        img = imgarr[sl,:,:,:]

        maxval = np.amax(np.absolute(img))
        img = img/maxval * 100 # Normalize to 100 since absolute numbers are not meaningful

#        noise_mean = np.average(img[0:10,0:10,0])
        noise_mean = np.average(img[0,0:10,0:10])
        # fit_threshold = 5*noise_mean
        fit_threshold = 10 * noise_mean

        error_count = 0
        for j in range(0,yres):
            for i in range(0,xres):
                yn = img[:,j,i]
#                yn = img[i,j,:]
                if (yn[0] > fit_threshold):
                    init_bval = -np.log(yn[1]/yn[0])/(b_array[1]-b_array[0])
                    try:
                        #3 parameters
                        # popt, pcov = curve_fit(func, b_array, yn, p0 = [(yn[0]-yn[4]), init_bval, yn[4]])
                        # fitval[0:3,sl,j,i] = popt
                        #2 parameters-5 b-values
                        popt, pcov = curve_fit(func, b_array, yn, p0=[(yn[0] - yn[4]), init_bval])  # this is for 3 b-values
                        fitval[0:2, sl, j, i] = popt

                    except RuntimeError:
                        #print("Error - curve_fit failed. i,j = ", i, j)
                        error_count += 1
                        fitval[:,sl,j,i] = [0,0,0,0.5] # Assign to certain color if fit doesn't converge
                else: fitval[:,sl,j,i] = [0,0,0,0.8] # Assign to certain color if signal is too low to fit


    # Set min/max to reasonable values
    tmparr = fitval[:,:,:,0]
    fitval[0,:,:,:] = np.clip(fitval[0,:,:,:],0,100) # Max of 100% of first b-value signal for coefficient
    fitval[1,:,:,:] = np.clip(fitval[1,:,:,:],0,0.005) # Max diffusion value of 0.005
    fitval[2,:,:,:] = np.clip(fitval[2,:,:,:],0,100) # Max of 100% of first b-value for baseline
    
    # Save results to file
    out_file = save_dir + dir_to_analyze[k] + "_Diffusion_Fits_2param.npy"
    # fitval.tofile(out_file)
    np.save(out_file, fitval)

    
    # View the original images
    plt.switch_backend('agg')
    plt.figure(num=1, figsize=(6,5))
    plt.subplots_adjust(left=0.01, bottom=0.07, right=0.99, top=0.99, wspace=0.02, hspace=0.02)
    for i in range(0, slices):
        plt.subplot(4,5,i+1)
        plt.axis('off')
        plt.imshow(imgarr[i,0,:,:], cmap='gray')
#        plt.imshow(imgarr[:,:,0,i], cmap='gray')
    plt.annotate("All Slices, b=0",xy=(5,5), xycoords = "figure points")

    # Show diffusion coefficients for all slices
    plt.figure(num=2, figsize=(6,5))
    plt.subplots_adjust(left=0.01, bottom=0.07, right=0.99, top=0.99, wspace=0.02, hspace=0.02)
    for i in range(0, slices):
        plt.subplot(4,5,i+1)
        plt.axis('off')
        plt.imshow(fitval[1,i,:,:], cmap='hot', vmin = 0, vmax = 0.005)
#        plt.imshow(fitval[:,:,i,1], cmap='hot', vmin = 0, vmax = 0.005)
    cb=plt.colorbar(fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=7)
    plt.annotate("ADC",xy=(5,5), xycoords = "figure points")

    out_file = save_dir + dir_to_analyze[k] + "_Diffusion_Results_2param.pdf"
    pdf = matplotlib.backends.backend_pdf.PdfPages(out_file)
    pdf.savefig(1) # b=0 image of all slices
    pdf.savefig(2) # Diffusion coeffients of all slices

    for i in range(0, slices):
        
        plt.figure(num=i+3, figsize=(7,2))
        plt.subplots_adjust(left=0.03, bottom=0.07, right=0.96, top=0.99, wspace=0.3, hspace=0.01)
        plt.subplot(1,4,1)
        plt.axis('off')
        plt.imshow(fitval[0,i,:,:], cmap='hot', vmin = 0, vmax = 100)
#        plt.imshow(fitval[:,:,i,0], cmap='hot', vmin = 0, vmax = 100)
        plt.title("Coeff of Exponential", fontsize=8)
        cb=plt.colorbar(fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=7)

        plt.subplot(1,4,2)
        plt.axis('off')
        plt.imshow(fitval[1,i,:,:], cmap='hot', vmin = 0, vmax = 0.005)
        plt.title("ADC Map", fontsize=8)
        cb=plt.colorbar(fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=7)

        plt.subplot(1,4,3)
        plt.axis('off')
        plt.imshow(fitval[2,i,:,:], cmap='hot', vmin = 0, vmax = 15)
        plt.title("Kurtosis", fontsize=8)
        cb=plt.colorbar(fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=7)

        plt.subplot(1,4,4)
        plt.axis('off')
        plt.imshow(fitval[3,i,:,:], cmap='seismic', vmin = 0, vmax = 1.0)
        plt.title("Red=LowSNR White=Error", fontsize=8)

        textout = "Slice = " + str(i+1)
        plt.annotate(textout,xy=(5,5), xycoords = "figure points")
        pdf.savefig(i+3) # Save results for each slice
        plt.close()
        
    pdf.close()

plt.show()

print("Finish =", datetime.now())
print("DONE!")