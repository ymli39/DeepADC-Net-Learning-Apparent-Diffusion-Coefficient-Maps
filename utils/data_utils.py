import numpy as np
import os
import matplotlib.pyplot as plt
import pdb
from functools import partial
from multiprocessing import Pool
import scipy.misc
import tempfile
import SimpleITK as sitk
import random
from skimage import io
from scipy import signal
import matplotlib.backends.backend_pdf
  
  
def read_raw(binary_file_name, image_size = [96,96], image_spacing=None,
              image_origin=None, big_endian=False):
     """
     Read a raw binary scalar image.
  
     Parameters
     ----------
     binary_file_name (str): Raw, binary image file content.
     image_size (tuple like): Size of image (e.g. [2048,2048])
     sitk_pixel_type (SimpleITK pixel type: Pixel type of data (e.g.
         sitk.sitkUInt16).
     image_spacing (tuple like): Optional image spacing, if none given assumed
         to be [1]*dim.
     image_origin (tuple like): Optional image origin, if none given assumed to
         be [0]*dim.
     big_endian (bool): Optional byte order indicator, if True big endian, else
         little endian.
  
     Returns
     -------
     SimpleITK image or None if fails.
     """
     
     string_to_pixelType = {"sitkUInt8": sitk.sitkUInt8,
                            "sitkInt8": sitk.sitkInt8,
                            "sitkUInt16": sitk.sitkUInt16,
                            "sitkInt16": sitk.sitkInt16,
                            "sitkUInt32": sitk.sitkUInt32,
                            "sitkInt32": sitk.sitkInt32,
                            "sitkUInt64": sitk.sitkUInt64,
                            "sitkInt64": sitk.sitkInt64,
                            "sitkFloat32": sitk.sitkFloat32,
                            "sitkFloat64": sitk.sitkFloat64}
     
     pixel_dict = {sitk.sitkUInt8: 'MET_UCHAR',
                   sitk.sitkInt8: 'MET_CHAR',
                   sitk.sitkUInt16: 'MET_USHORT',
                   sitk.sitkInt16: 'MET_SHORT',
                   sitk.sitkUInt32: 'MET_UINT',
                   sitk.sitkInt32: 'MET_INT',
                   sitk.sitkUInt64: 'MET_ULONG_LONG',
                   sitk.sitkInt64: 'MET_LONG_LONG',
                   sitk.sitkFloat32: 'MET_FLOAT',
                   sitk.sitkFloat64: 'MET_DOUBLE'}
     
     direction_cosine = ['1 0 0 1', '1 0 0 0 1 0 0 0 1',
                         '1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1']
     
     sitk_pixel_type=string_to_pixelType['sitkUInt8']
     
     dim = len(image_size)
     header = ['ObjectType = Image\n'.encode(),
               ('NDims = {0}\n'.format(dim)).encode(),
               ('DimSize = ' + ' '.join([str(v) for v in image_size]) + '\n')
               .encode(),
               ('ElementSpacing = ' + (' '.join([str(v) for v in image_spacing])
                                       if image_spacing else ' '.join(
                   ['1'] * dim)) + '\n').encode(),
               ('Offset = ' + (
                   ' '.join([str(v) for v in image_origin]) if image_origin
                   else ' '.join(['0'] * dim) + '\n')).encode(),
               ('TransformMatrix = ' + direction_cosine[dim - 2] + '\n')
               .encode(),
               ('ElementType = ' + pixel_dict[sitk_pixel_type] + '\n').encode(),
               'BinaryData = True\n'.encode(),
               ('BinaryDataByteOrderMSB = ' + str(big_endian) + '\n').encode(),
               # ElementDataFile must be the last entry in the header
               ('ElementDataFile = ' + os.path.abspath(
                   binary_file_name) + '\n').encode()]
     fp = tempfile.NamedTemporaryFile(suffix='.mhd', delete=False)
  
     # Not using the tempfile with a context manager and auto-delete
     # because on windows we can't open the file a second time for ReadImage.
     fp.writelines(header)
     fp.close()
     img = sitk.ReadImage(fp.name)
     os.remove(fp.name)
     
     image = sitk.GetArrayFromImage(img)
     return image
 

def make_data_list(folder_dir, savefile):
    data_list = [f.split('-')[0].split('_')[0] for f in os.listdir(folder_dir)]
    
    random.shuffle(data_list)

    datadir = '/home/ym/Desktop/research/titanserver/home/yli/DWI_project/DWI_new_process/datalists/image_list'
    file = open(os.path.join(savefile, 'image_list'), 'w')
    for item in data_list:
        file.writelines(item + '\n')
    file.close()

    namefile = open(datadir, 'r')
    data_list = namefile.readlines()
    data_list = [f[:-1] for f in data_list]

    
    total_num = len(data_list)
    vali_num = int(total_num*0.1)
    test_num = int(total_num*0.2)
    train_num = total_num - vali_num - test_num
    
    train_list = data_list[:train_num]
    file = open(os.path.join(savefile, 'train_list'), 'w')
    for item in train_list:
        file.writelines(item + '\n')
    file.close()
    
    vali_list = data_list[train_num:train_num+vali_num]
    file = open(os.path.join(savefile, 'vali_list'), 'w')
    for item in vali_list:
        file.writelines(item + '\n')
    file.close()
    
    test_list = data_list[train_num+vali_num:]
    file = open(os.path.join(savefile, 'test_list'), 'w')
    for item in test_list:
        file.writelines(item + '\n')
    file.close()
    

def fftshift_demo():
    '''
    code could be explained at:
        https://medium.com/@hicraigchen/digital-image-processing-using-fourier-transform-in-python-bcb49424fd82
    '''
    img_c1 = img
    img_c2 = np.fft.fft2(img_c1)
    img_c3 = np.fft.fftshift(img_c2)
    img_c4 = np.fft.ifftshift(img_c3)
    img_c5 = np.fft.ifft2(img_c4)
    
    img_c2[10,10]
    
    plt.subplot(151), plt.imshow(img_c1, "gray"), plt.title("Original Image")
    plt.subplot(152), plt.imshow(np.log(1+np.abs(img_c2)), "gray"), plt.title("Spectrum")
    plt.subplot(153), plt.imshow(np.log(1+np.abs(img_c3)), "gray"), plt.title("Centered Spectrum")
    plt.subplot(154), plt.imshow(np.log(1+np.abs(img_c4)), "gray"), plt.title("Decentralized")
    plt.subplot(155), plt.imshow(np.abs(img_c5), "gray"), plt.title("Processed Image")
    
def downsample_demo():
    
    orig_img = '/home/ym/Desktop/research/server2/home/yli/DWI_project/DataHighRes/s_2019072503/image/image0005.raw'
    down_01 = '/home/ym/Desktop/research/server2/home/yli/DWI_project/DataHighRes/s_2019080201/corruptedData01/image0005.raw'
    down_02 = '/home/ym/Desktop/research/server2/home/yli/DWI_project/DataHighRes/s_2019080201/corruptedData02/image0005.raw'
    down_03 = '/home/ym/Desktop/research/server2/home/yli/DWI_project/DataHighRes/s_2019072503/corruptedData03/image0005.raw'
    image_orig = read_raw(binary_file_name=orig_img)
    image_01 = read_raw(binary_file_name=down_01)
    image_02 = read_raw(binary_file_name=down_02)
    image_03 = read_raw(binary_file_name=down_03)
    
    stack = '/home/ym/Desktop/research/titanserver/home/yli/DWI_project/Processed DWI data/processed 2019071802 dwi data/ADCstack.raw'
    ADCstack = read_raw(binary_file_name=stack)
    plt.imshow(ADCstack[0])
    
    plt.figure(dpi=400)
    plt.subplot(141), plt.imshow(image_orig, "gray"), plt.title("Original Image")
    plt.subplot(142), plt.imshow(image_03, "gray"), plt.title("down 1/4")
    plt.subplot(143), plt.imshow(image_01, "gray"), plt.title("down 1/2")
    plt.subplot(144), plt.imshow(image_02, "gray"), plt.title("down 3/4")
    plt.savefig('dwi_image.png')
    
    
def image_visual():
    highres_dir = '/home/yli/DWI_project/DataHighRes'
    data_dir = [os.path.join(highres_dir, o) for o in os.listdir(highres_dir) if os.path.isdir(os.path.join(highres_dir,o))]
    
    for data_folder in data_dir:
        image_dir = os.path.join(data_folder, 'image')
        image_files = [os.path.join(image_dir, o) for o in os.listdir(image_dir)]
        
        visual = os.path.join(image_dir, 'visual')
        if not os.path.exists(visual):
            os.makedirs(visual)
            
        num_slice = int(len(image_files)/10)
        for i in range(num_slice):
            image_file1 = os.path.join(image_dir, 'image{0:04}.raw'.format(i+1))
            image1 = read_raw(binary_file_name=image_file1)
            
            image_file2 = os.path.join(image_dir, 'image{0:04}.raw'.format(i+1+num_slice))
            image2 = read_raw(binary_file_name=image_file2)
            
            image_file3 = os.path.join(image_dir, 'image{0:04}.raw'.format(i+1+num_slice*2))
            image3 = read_raw(binary_file_name=image_file3)
            
            image_file4 = os.path.join(image_dir, 'image{0:04}.raw'.format(i+1+num_slice*3))
            image4 = read_raw(binary_file_name=image_file4)
            
            image_file5 = os.path.join(image_dir, 'image{0:04}.raw'.format(i+1+num_slice*4))
            image5 = read_raw(binary_file_name=image_file5)
            
            image_file6 = os.path.join(image_dir, 'image{0:04}.raw'.format(i+1+num_slice*5))
            image6 = read_raw(binary_file_name=image_file6)
            
            image_file7 = os.path.join(image_dir, 'image{0:04}.raw'.format(i+1+num_slice*6))
            image7 = read_raw(binary_file_name=image_file7)
            
            image_file8 = os.path.join(image_dir, 'image{0:04}.raw'.format(i+1+num_slice*7))
            image8 = read_raw(binary_file_name=image_file8)
            
            image_file9 = os.path.join(image_dir, 'image{0:04}.raw'.format(i+1+num_slice*8))
            image9 = read_raw(binary_file_name=image_file9)
            
            image_file10 = os.path.join(image_dir, 'image{0:04}.raw'.format(i+1+num_slice*9))
            image10 = read_raw(binary_file_name=image_file10)
            
            plt.switch_backend('agg')
            plt.figure(dpi=200)
            plt.subplot(2,5,1), plt.imshow(image1, "gray"), plt.xticks([]), plt.yticks([])
            plt.subplot(2,5,2), plt.imshow(image2, "gray"), plt.xticks([]), plt.yticks([])
            plt.subplot(2,5,3), plt.imshow(image3, "gray"), plt.xticks([]), plt.yticks([])
            plt.subplot(2,5,4), plt.imshow(image4, "gray"), plt.xticks([]), plt.yticks([])
            plt.subplot(2,5,5), plt.imshow(image5, "gray"), plt.xticks([]), plt.yticks([])
            plt.subplot(2,5,6), plt.imshow(image6, "gray"), plt.xticks([]), plt.yticks([])
            plt.subplot(2,5,7), plt.imshow(image7, "gray"), plt.xticks([]), plt.yticks([])
            plt.subplot(2,5,8), plt.imshow(image8, "gray"), plt.xticks([]), plt.yticks([])
            plt.subplot(2,5,9), plt.imshow(image9, "gray"), plt.xticks([]), plt.yticks([])
            plt.subplot(2,5,10), plt.imshow(image10, "gray"), plt.xticks([]), plt.yticks([])
            plt.tight_layout()
            plt.savefig(os.path.join(visual, 'slice'+ str(i+1) + '.png'))
        
#        for image_file in image_files:
#            
#            image_orig = read_raw(binary_file_name=image_file)
#            image_id = image_file.split('/')[-1][:-4]
#            
#            plt.switch_backend('agg')
#            plt.figure(dpi=100)
#            plt.imshow(image_orig, "gray")
#            plt.savefig(os.path.join(visual, image_id + '.png'))
        print('finished folder: ' + image_dir)
        
        
def gen_multi_list():
    highres_dir = '/home/yli/DWI_project/DataHighRes'
    data_dir = [os.path.join(highres_dir, o) for o in os.listdir(highres_dir) if os.path.isdir(os.path.join(highres_dir,o))]
    
    img_lists = []
    for data_folder in data_dir:
        image_dir = os.path.join(data_folder, 'image')
        image_files = [os.path.join(image_dir, o) for o in os.listdir(image_dir)]
        
        num_slice = int(len(image_files)/10)
        
        for i in range(num_slice):
            imgind1 = i+1
            imgind2 = i+1+num_slice*1
            imgind3 = i+1+num_slice*2
            imgind4 = i+1+num_slice*3
            imgind5 = i+1+num_slice*4
            imgind6 = i+1+num_slice*5
            imgind7 = i+1+num_slice*6
            imgind8 = i+1+num_slice*7
            imgind9 = i+1+num_slice*8
            imgind10 = i+1+num_slice*9
            
            img_list1 = image_dir + '/ ' + str(imgind1) + ' ' + str(imgind6)
            img_list2 = image_dir + '/ ' + str(imgind2) + ' ' + str(imgind7)
            img_list3 = image_dir + '/ ' + str(imgind3) + ' ' + str(imgind8)
            img_list4 = image_dir + '/ ' + str(imgind4) + ' ' + str(imgind9)
            img_list5 = image_dir + '/ ' + str(imgind5) + ' ' + str(imgind10)
            
            img_lists.append(img_list1)
            img_lists.append(img_list2)
            img_lists.append(img_list3)
            img_lists.append(img_list4)
            img_lists.append(img_list5)
            
#            img_list = image_dir + '/ ' + str(imgind1) + ' ' + str(imgind2) + ' ' + str(imgind3) + ' ' + str(imgind4) + ' ' + str(imgind5) + ' ' + str(imgind6) + ' ' + str(imgind7) + ' ' + str(imgind8) + ' ' + str(imgind9) + ' ' + str(imgind10)
#            img_lists.append(img_list)
    
    random.shuffle(img_lists)
    savefile = '/home/ym/Desktop/research/titanserver/home/yli/DWI_project/DWI_code/datalists/'
    file = open(os.path.join(savefile, 'image_adc'), 'w')
    for item in img_lists:
        file.writelines(item)
    file.close()
    
    total_num = len(img_lists)
    vali_num = int(total_num*0.1)
    test_num = int(total_num*0.2)
    train_num = total_num - vali_num - test_num
    
    train_list = img_lists[:train_num]
    file = open(os.path.join(savefile, 'train_adc'), 'w')
    for item in train_list:
        file.writelines(item)# + '\n')
    file.close()
    
    vali_list = img_lists[train_num:train_num+vali_num]
    file = open(os.path.join(savefile, 'vali_adc'), 'w')
    for item in vali_list:
        file.writelines(item)# + '\n')
    file.close()
    
    test_list = img_lists[train_num+vali_num:]
    file = open(os.path.join(savefile, 'test_adc'), 'w')
    for item in test_list:
        file.writelines(item)# + '\n')
    file.close()
    
    

def readTifImage():
    from PIL import Image
    im = Image.open('/home/ym/Desktop/research/titanserver/home/yli/DWI_project/Processed DWI data/processed 2019071802 dwi data/ADCstack.tif')
    imarray = np.array(im)
    
    plt.imshow(imarray)
    
    from skimage import io
    im1 = io.imread('/home/ym/Desktop/research/titanserver/home/yli/DWI_project/Processed DWI data/processed 2019072301 dwi data/ADCstack.tif')
    
    im2 = io.imread('/home/ym/Desktop/research/titanserver/home/yli/DWI_project/Processed DWI data/processed 2019072301 dwi data/rad stack/ADC stack.tif')
    
    im2 = im2[1:]
    
    im1[np.where(np.isnan(im1))]=0
    np.max(im1)
    np.mean(im1)
    
    from tifffile import imsave
    imsave('/home/ym/Desktop/research/titanserver/home/yli/DWI_project/Processed DWI data/processed 2019072301 dwi data/ADCstack.tif', im2)
    
    np.sum(im1-im2)
    
    plt.imshow(im1[1], cmap='bone')
    
    plt.figure(dpi=200)
    plt.subplot(1,3,1), plt.imshow(im1[0], "gray"), plt.xticks([]), plt.yticks([])
    plt.subplot(1,3,2), plt.imshow(im1[1], "gray"), plt.xticks([]), plt.yticks([])
    plt.subplot(1,3,3), plt.imshow(im1[2], "gray"), plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join('/home/ym/Desktop/research/titanserver/home/yli/DWI_project/Processed DWI data/processed 2019071802 dwi data/rad stack', 'ADC.png'))
    
    file_path = '/home/ym/Desktop/research/titanserver/home/yli/DWI_project/Processed DWI data/processed 2019071802 dwi data/radsemsdw_sp_01.fid/ADCslice005.fdf'
    fdf_file = open(file_path, 'rb')
    fdf = fdf_file.read()
    
    print(fdf.decode('latin-1'))

# https://github.com/ismrmrd/ismrmrd-python-tools/blob/master/ismrmrdtools/transform.py
import numpy as np
from numpy.fft import fftshift, ifftshift, fftn, ifftn

def transform_kspace_to_image(k, dim=None, img_shape=None):
    """ Computes the Fourier transform from k-space to image space
    along a given or all dimensions
    :param k: k-space data
    :param dim: vector of dimensions to transform
    :param img_shape: desired shape of output image
    :returns: data in image space (along transformed dimensions)
    """
    if not dim:
        dim = range(k.ndim)

    img = fftshift(ifftn(ifftshift(k, axes=dim), s=img_shape, axes=dim), axes=dim)
    img *= np.sqrt(np.prod(np.take(img.shape, dim)))
    return img


def transform_image_to_kspace(img, dim=None, k_shape=None):
    """ Computes the Fourier transform from image space to k-space space
    along a given or all dimensions
    :param img: image space data
    :param dim: vector of dimensions to transform
    :param k_shape: desired shape of output k-space data
    :returns: data in k-space (along transformed dimensions)
    """
    if not dim:
        dim = range(img.ndim)

    k = fftshift(fftn(ifftshift(img, axes=dim), s=k_shape, axes=dim), axes=dim)
    k /= np.sqrt(np.prod(np.take(img.shape, dim)))
    return k



def preprocess_complex_img():
    datadir = '/mnt/4TBHDD/yli/DWI_data/image_complex/'
    datadir_4x = os.path.join(datadir, '4')
    
    data_list4x = [os.path.join(datadir_4x, o) for o in os.listdir(datadir_4x)]
    
    for data_list in data_list4x:

        print(data_list)
        
        img_data = np.load(data_list)
        data_crop, _ = crop_and_fft(img_data)
        
#        data_list_split = data_list.split('/')
#        kspace_list = data_list_split.copy()
#        kspace_list[-3] = 'kspace'
#        kspace_dir = '/'.join(kspace_list)
        
        
        np.save(data_list, data_crop)
#        np.save(kspace_dir, kspace_crop)
        print('finished 4x image list: ' + data_list)
        
        
def crop_and_fft(img_data):
    
    sl, bval, xres, yres = img_data.shape

    print(xres)
    assert xres == yres == 192
    
    data_crop = np.zeros((sl, bval, xres//2, yres//2), dtype=np.complex)
    kspace_crop = np.zeros((sl, bval, xres//2, yres//2), dtype=np.complex)
    for ii in range(0, sl):
        for jj in range(0, bval):
            cur_img = img_data[ii][jj]
            crop_cur_img = cur_img[48:144, 48:144]
            data_crop[ii][jj] = crop_cur_img
            
            cur_kspace = np.fft.fftshift(np.fft.fft2(crop_cur_img))
            kspace_crop[ii][jj] = cur_kspace
    
#            plt.imshow(np.abs(crop_cur_img), cmap='gray')
#            plt.imshow(np.log(1+np.abs(cur_kspace)), cmap='gray')
    
    return data_crop, kspace_crop
            
        
def compared_adc():
    from skimage.measure import compare_psnr, compare_ssim
    from skimage.metrics import structural_similarity as ssim

    adc_pred_dir = '/home/ym/Desktop/research/titanserver/mnt/4TBHDD/yli/DWI_data/image_complex/adc_generated/'
    adc_gt_dir = '/home/ym/Desktop/research/titanserver/mnt/4TBHDD/yli/DWI_data/image_complex/adcmap/'
    
    data_list = ['M45_post_Diffusion_Fits.npy',
                 'M20_Diffusion_Fits.npy',
                 'M43_Diffusion_Fits.npy',
                 'M25_Diffusion_Fits.npy',
                 'M08_Diffusion_Fits.npy',
            ]
    
#    data_list = ['M13_Diffusion_Fits.npy',
#                 'M09_Diffusion_Fits.npy',
#                 'M12_Diffusion_Fits.npy',
#                 'M30_Diffusion_Fits.npy',
#                 'M26_Diffusion_Fits.npy'
#            ]
    test_num = len(data_list)
    
    rec_total_nmse  = 0
    rec_total_psnr  = 0
    rec_total_ssim  = 0
    rec_total_corr = 0
    
    for cur_list in data_list:
        adc_pred = np.load(os.path.join(adc_pred_dir, cur_list))[1]
        adc_gt = np.load(os.path.join(adc_gt_dir, cur_list))[1]
        
        #%%
        adc_pred = np.load('/home/ym/Desktop/research/titanserver/home/yli/DWI_project/DWI_new_process/checkpoints/DWI_comb/visualization/test/M02_cx_image.npy')
        adc_gt = np.load('/home/ym/Desktop/research/titanserver/mnt/4TBHDD/DWI_data_new/M02-20201118-KPC-26594-1/M02_Diffusion_Fits.npy')
        adc_pred = adc_pred[13]*0.005
        adc_gt = adc_gt[1][13]
        
        bool_min = adc_gt>np.min(adc_gt)
        bool_max = adc_gt<np.max(adc_gt)
        mask = (bool_min * bool_max).astype(int)
        
        adc_pred_flat = adc_pred.flatten()
        adc_gt_flat = adc_gt.flatten()
        
        plt.figure(dpi=400)
        plt.subplot(131), plt.imshow(adc_gt, "gray"), plt.title("gt adc")
        plt.subplot(132), plt.imshow(adc_pred, "gray"), plt.title("predicted adc")
        plt.subplot(133), plt.imshow(mask, "gray"), plt.title("mask")
        plt.savefig('dwi_image.png')
        
        plt.figure(dpi=200)
        plt.scatter(adc_gt_flat, adc_pred_flat, s=1), plt.title("scatter plot"), plt.xlim(0.0001, 0.005), plt.ylim(0.0001, 0.005)
        plt.savefig('scatter_comb.png')
        
        input_file = '/home/ym/Desktop/research/titanserver/mnt/4TBHDD/DWI_data_new/M02-20201118-KPC-26594-1/M02_AllSlicesBvalues.bin'
        tmpdat = np.fromfile(input_file, dtype=np.float64, count = 96*96*5*19)
    #    imgarr = np.reshape(tmpdat,(xres,yres,bvalues,slices), order = 'F') # Fortran order (fastest changing first)
        imgarr = np.reshape(tmpdat,(19,5,96,96))
        
        np.max(imgarr[0][0])
        np.min(imgarr[0][0])
        np.mean(imgarr[0][0])
        
        _ = plt.hist(imgarr[0][0], bins='auto')  # arguments are passed to np.histogram
        plt.title("Histogram with 'auto' bins")
        plt.savefig('histogram.png')
        
        #%%
        ind = 10
        bval = np.load('/home/ym/Desktop/research/titanserver/mnt/4TBHDD/yli/DWI_data/image_complex/down_sampled_8x/M03_cx_image_data_8x.npy')
        bvals = np.abs(bval[ind])
        
        adc_map = np.load('/home/ym/Desktop/research/titanserver/mnt/4TBHDD/yli/DWI_data/image_complex/reconstructed_adcs_8x/M03_Diffusion_Fits_5b_noc.npy')
        adc = adc_map[1][ind]

        # upper_thres = 0.0032
        #
        # from scipy import ndimage
        # adc[adc >= upper_thres] = 0
        # adc = ndimage.median_filter(adc, size=3)

        plt.figure(dpi=800)
        plt.subplot(161), plt.imshow((bvals[0]), "gray"), plt.xticks([]), plt.yticks([]), plt.title("b1")
        plt.subplot(162), plt.imshow((bvals[1]), "gray"), plt.xticks([]), plt.yticks([]), plt.title("b2")
        plt.subplot(163), plt.imshow((bvals[2]), "gray"), plt.xticks([]), plt.yticks([]), plt.title("b3")
        plt.subplot(164), plt.imshow((bvals[3]), "gray"), plt.xticks([]), plt.yticks([]), plt.title("b4")
        plt.subplot(165), plt.imshow((bvals[4]), "gray"), plt.xticks([]), plt.yticks([]), plt.title("b5")
        # plt.subplot(166), plt.imshow((adc), "hot", vmin=0, vmax=0.0035), plt.xticks([]), plt.yticks([]), plt.title("adc map")
        plt.subplot(166), plt.imshow((adc), "hot"), plt.xticks([]), plt.yticks([]), plt.title("S0")

        plt.savefig('/home/ym/Desktop/research/titanserver/home/yli/DWI_project/DWI_new_process/M03_8x_' + str(ind) + '.png')
        
        #%%
        adcmap = np.load('/home/ym/Desktop/research/titanserver/mnt/4TBHDD/DWI_data_new/M25-20201127-nude109-689/M25_Diffusion_Fits.npy')[1][1]
        fig, ax = plt.subplots(dpi=400)
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)

        im = ax.imshow(adcmap, "hot") 
        ax.set_xticks([]), ax.set_yticks([])
#        cb = plt.colorbar(im, cax=cax, orientation='vertical')
        fig.colorbar(im, cax=cax, orientation='vertical')
        plt.savefig('/home/ym/Desktop/research/titanserver/home/yli/DWI_project/DWI_new_process/adc_demo.png')
        
        
        #%%
def adc_clip_mediam_filter():
    datadir = '/home/ym/Desktop/research/titanserver/mnt/4TBHDD/yli/DWI_data/image_complex/adcmap_4x/'
    datanames = [o for o in os.listdir(datadir)]
    
    newadcdir = '/home/ym/Desktop/research/titanserver/mnt/4TBHDD/yli/DWI_data/image_complex/adcmap_4x_mediam_k3/'
    if not os.path.exists(newadcdir):
        os.makedirs(newadcdir)
    
    from scipy import ndimage
    for data_list in datanames:
        read_all = np.load(os.path.join(datadir, data_list))
        adc_map = read_all[1]
        
        new_adc = np.clip(adc_map, 0, 0.0032)
        
        adc_stack = []
        for ind in range(len(adc_map)):
            cur_adc = new_adc[ind]
            mask = cur_adc.copy()
            mask[mask>0]=1
            
            adc_mediam = ndimage.median_filter(cur_adc, size=3)
#            plt.imshow(adc_mediam*mask, cmap='hot')
#            plt.imshow(adc_mediam, cmap='hot')
            adc_stack.append(adc_mediam)
            if np.max(adc_mediam)-np.min(adc_mediam) == 0:
                print(data_list + ' and ind num ' + str(ind))
        
        adc_array = np.asarray(adc_stack)
        np.save(os.path.join(newadcdir, data_list), adc_array)


#%%
def func(x, a, b, c):
    return a * np.exp(-b * x) + c


def calc_residule(data_name, adc_name, save_name):
    
    #%%
    folder_dir = '/home/ym/Desktop/research/titanserver/mnt/4TBHDD/DWI_data_new/'
    
    folder_list = ['M45-20201210-KPC-26778-dce_post',
                 'M36-20201204-KPC-9019-dce_post',
                 'M37-20201207-CSX-789-3',
                 'M10-20201121-nude1-191',
                 'M45-20201210-KPC-26778-dce_pre',
                 'M42-20201208-CSX-778-19',
                 'M08-20201120-KPC-8817',
                 'M18-20201125-KPC-26303-dce',
                 'M07-20201120-KPC-26412-dce_pre',
                 'M06-2020-KPC-26594',
                 'M29-20201201nude375-689',
                 'M40-20201208-CSX-789-2'
                 ]
    
    data_dir = [os.path.join(folder_dir, o) for o in folder_list]
    
    residues1_all = []
    residues2_all = []
    
    residues1_all_std = []
    residues2_all_std = []
    for cur_dir in data_dir:
        
        scan_num = cur_dir.split('/')[-1].split('-')[0]
        
        data_name = scan_num + '_AllSlicesBvalues.bin'
        adc_name = scan_num + '_Diffusion_Fits_3b_noc.npy'
        
        if 'post' in cur_dir.split('/')[-1].split('-')[-1]:
            scan_num = scan_num + '_post'
        elif 'pre' in cur_dir.split('/')[-1].split('-')[-1]:
            scan_num = scan_num + '_pre'
            
        save_name = scan_num +  "pixel_compare.pdf"
         
        bvalues = 5
        yres = 96
        xres = 96
        slices = 19
        data_dir = cur_dir +'/'+ data_name
        adc_dir = cur_dir +'/'+ adc_name
        dl_data_dir = '/home/ym/Desktop/research/titanserver/home/yli/DWI_project/DWI_new_process/checkpoints/DWI_adc_unpool_sample12/visualization/vali/' + scan_num +  '_curve_fit_img.npy'
        dl_adc_dir = '/home/ym/Desktop/research/titanserver/home/yli/DWI_project/DWI_new_process/checkpoints/DWI_adc_unpool_sample12/visualization/vali/' + scan_num +  '_curve_fit_param.npy'
        
        
        #load data
        tmpdat = np.fromfile(data_dir, dtype=np.float64, count = xres*yres*bvalues*slices)
        imgarr = np.reshape(tmpdat,(slices,bvalues,yres,xres))
        adcarr = np.load(adc_dir)
        dl_data = np.load(dl_data_dir)
        dl_adc = np.load(dl_adc_dir)
        
    #    b_array_5 = np.array([24.25, 536.86, 1072.62, 1482.07, 2144.69]) #5bvalues
        b_array_3 = np.array([24.25, 536.86, 1072.62]) #3bvalues
        b_array = b_array_3
        
        
        All_residues1 = []
        All_residues2 = []
        
        All_residues1_std = []
        All_residues2_std = []
        
        adc_stack =[]
        for ind in range(len(imgarr)):
        
            S0 = adcarr[0][ind]
            ADC = adcarr[1][ind]
            noise = adcarr[2][ind]
            img_stack = imgarr[ind]
            dl_stack = dl_data[ind]
            cur_adc = dl_adc[ind,1,:,:]
            
            bool_b12 = img_stack[0,:,:] > img_stack[1,:,:]
            bool_b23 = img_stack[1,:,:] > img_stack[2,:,:]
            mask1 = (bool_b12 * bool_b23).astype(int)
            
            bool_min = ADC>np.min(ADC)
            bool_max = ADC<np.max(ADC)
            mask2 = (bool_min * bool_max).astype(int)
            mask = mask1 * mask2
            
            masked_adc = cur_adc*mask
            adc_stack.append(masked_adc)
            
            residues1 = []
            residues2 = []
            for i in range(len(b_array)):
                cur_img = img_stack[i,:,:]
                cur_recon = func(ADC, S0, b_array[i], noise)
                img_scale = (cur_img-np.min(cur_img))/(np.max(cur_img) - np.min(cur_img)) * 100
                
                cur_dl = dl_stack[i,:,:]
                
                residue1 = np.abs(img_scale * mask - cur_recon * mask)
                residue1 = (residue1-np.min(residue1))/(np.max(residue1) - np.min(residue1))
                residues1.append(residue1)
                
                residue2 = np.abs(cur_img * mask - cur_dl * mask)
                residue2 = (residue2-np.min(residue2))/(np.max(residue2) - np.min(residue2))
                residues2.append(residue2)
            residues1 = np.asarray(residues1)
            residues1_mean = np.mean(residues1, axis=0)
            residues1_std = np.std(residues1, axis=0)
            
            residues2 = np.asarray(residues2)
            residues2_mean = np.mean(residues2, axis=0)
            residues2_std = np.std(residues2, axis=0)
            
            All_residues1.append(residues1_mean)
            All_residues2.append(residues2_mean)
            
            All_residues1_std.append(residues1_std)
            All_residues2_std.append(residues2_std)
        All_residues1 = np.asarray(All_residues1)
        mean_residues1 = np.mean(All_residues1)
        
        All_residues1_std = np.asarray(All_residues1_std)
        
        residues1_all.append(All_residues1)
        residues1_all_std.append(All_residues1_std)
        
        All_residues2 = np.asarray(All_residues2)
        mean_residues2 = np.mean(All_residues2)
        
        All_residues2_std = np.asarray(All_residues2_std)
        residues2_all_std.append(All_residues2_std)
        
        residues2_all.append(All_residues2)
        
        adc_stack = np.asarray(adc_stack)
        
        
        # Show diffusion coefficients for all slices
        plt.figure(num=1, figsize=(6,5))
        plt.subplots_adjust(left=0.01, bottom=0.07, right=0.99, top=0.99, wspace=0.02, hspace=0.02)
        for i in range(0, slices):
            plt.subplot(4,5,i+1)
            plt.axis('off')
            plt.imshow(adcarr[1,i,:,:], cmap='hot', vmin = 0, vmax = 0.005)
        cb=plt.colorbar(fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=7)
        plt.annotate("ADC from HeeKown",xy=(5,5), xycoords = "figure points")
        
        # Show residue map of all slice
        plt.figure(num=2, figsize=(6,5))
        plt.subplots_adjust(left=0.01, bottom=0.07, right=0.99, top=0.99, wspace=0.02, hspace=0.02)
        for i in range(0, slices):
            plt.subplot(4,5,i+1)
            plt.axis('off')
            plt.imshow(All_residues1[i,:,:], cmap='hot', vmin = 0, vmax = 1)
        cb=plt.colorbar(fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=7)
        plt.annotate("Residue map from HeeKown: {}".format(mean_residues1),xy=(5,5), xycoords = "figure points")
        
        plt.figure(num=3, figsize=(6,5))
        plt.subplots_adjust(left=0.01, bottom=0.07, right=0.99, top=0.99, wspace=0.02, hspace=0.02)
        for i in range(0, slices):
            plt.subplot(4,5,i+1)
            plt.axis('off')
            plt.imshow(adc_stack[i,:,:], cmap='hot', vmin = 0, vmax = 0.005)
        cb=plt.colorbar(fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=7)
        plt.annotate("ADC from dl",xy=(5,5), xycoords = "figure points")
        
        plt.figure(num=4, figsize=(6,5))
        plt.subplots_adjust(left=0.01, bottom=0.07, right=0.99, top=0.99, wspace=0.02, hspace=0.02)
        for i in range(0, slices):
            plt.subplot(4,5,i+1)
            plt.axis('off')
            plt.imshow(All_residues2[i,:,:], cmap='hot', vmin = 0, vmax = 1)
        cb=plt.colorbar(fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=7)
        plt.annotate("Residue map from dl: {}".format(mean_residues2),xy=(5,5), xycoords = "figure points")
        
        out_file = '/home/ym/Desktop/research/titanserver/home/yli/DWI_project/DWI_new_process/' + data_name[0:3] + save_name
        
        pdf = matplotlib.backends.backend_pdf.PdfPages(out_file)
        pdf.savefig(1) # diffusion coefficients for all slices
        pdf.savefig(2) # residue map of all slice
        pdf.savefig(3) 
        pdf.savefig(4) 
        
        plt.close()
        pdf.close()
        
    residues1_all = np.asarray(residues1_all)
    print(np.mean(residues1_all))
    
    residues1_all_mean = np.mean(residues1_all, axis=(1,2,3))
    residues2_all_mean = np.mean(residues2_all, axis=(1,2,3))
    
    from scipy.stats import wilcoxon
    w, p = wilcoxon(np.abs(residues1_all_mean- residues2_all_mean))
    
    np.mean(residues1_all_mean)
    np.mean(residues2_all_mean)
    
    residues1_all_std = np.asarray(residues1_all_std)
    residues2_all_std = np.asarray(residues2_all_std)
    
    residues2_all = np.asarray(residues2_all)
    print(np.mean(residues2_all))
    
    print(np.mean(residues1_all_std))
    print(np.mean(residues2_all_std))
    

#%%
def getSNR():

    #%%
    datadir = '/home/ym/Desktop/research/titanserver/mnt/4TBHDD/yli/DWI_data/image_complex/1x/'
    adcdir = '/home/ym/Desktop/research/titanserver/mnt/4TBHDD/yli/DWI_data/image_complex/adc_3b_noc/'
    datalist = '/home/ym/Desktop/research/titanserver/home/yli/DWI_project/DWI_new_process/datalists/image_list'
    
    namefile = open(datalist, 'r')
    datanames = namefile.readlines()
    datanames = [f[:-1] for f in datanames]
    
#    valid_id = ['M07_post',
#             'M51',
#             'M48',
#             'M05',
#             'M47',
#             'M32',
#             'M28',
#             'M09',
#             'M46',
#             'M10',
#             'M29',
#             'M33',
#             'M07_pre',
#             'M08',
#             'M36_pre',
#             'M39',
#             'M38',
#             'M41',
#             'M03',
#             'M42',
#             'M40',
#             'M06',
#             'M37',
#             'M36_post',
#             'M18',
#             'M45_pre',
#             'M45_post']
    
#    valid_id = ['M02',]
#             'M61',
#             'M62']
    
    out_file = '/home/ym/Desktop/research/titanserver/home/yli/DWI_project/DWI_new_process/SNR_analysis.pdf'
    pdf = matplotlib.backends.backend_pdf.PdfPages(out_file)
    
    count = 1
    SNR_list = []
    bind = 1
    
    for name in datanames:
        cur_datas = np.load(os.path.join(datadir, name + '_cx_image_data.npy'))
        cur_data = cur_datas[:,bind,:,:]
        cur_adc = np.load(os.path.join(adcdir, name + '_Diffusion_Fits_3b_noc.npy'))[1]
        
        cur_data = np.absolute(cur_data)
        
        bool_min = cur_adc>np.min(cur_adc)
        bool_max = cur_adc<np.max(cur_adc)
        mask = (bool_min * bool_max).astype(int)
        
        dimen, x_ind, y_ind = np.where(mask==1)
        signal = np.mean(cur_data[dimen, x_ind, y_ind])

        dimen, x_ind, y_ind = np.where(mask==0)
        noise = np.mean(cur_data[dimen, x_ind, y_ind])
        
        SNR = signal/noise
        SNR_list.append(SNR)
    
    sortedind = sorted(range(len(SNR_list)), key=lambda k: SNR_list[k])
    sorted_names = [datanames[f] for f in sortedind]

    file = open(datalist, 'w')
    for item in sorted_names:
        file.writelines(item + '\n')
    file.close()
    
    for name in sorted_names:
        cur_datas = np.load(os.path.join(datadir, name + '_cx_image_data.npy'))
        cur_data = cur_datas[:,2,:,:]
        cur_adc = np.load(os.path.join(adcdir, name + '_Diffusion_Fits_3b_noc.npy'))[1]
        
        cur_data = np.absolute(cur_data)

        dim, xres, yres = cur_data.shape
        interm = int(dim/2)
        
        bool_min = cur_adc>np.min(cur_adc)
        bool_max = cur_adc<np.max(cur_adc)
        mask = (bool_min * bool_max).astype(int)
        
        dimen, x_ind, y_ind = np.where(mask==1)
        signal = np.mean(cur_data[dimen, x_ind, y_ind])

        dimen, x_ind, y_ind = np.where(mask==0)
        noise = np.mean(cur_data[dimen, x_ind, y_ind])
        
        SNR = signal/noise

        first_slices_all = np.absolute(cur_datas[interm])
        first_slices = np.absolute(cur_datas[interm,2,:,:])
        first_mask = mask[interm]

        x_ind, y_ind = np.where(first_mask==1)
        f_tissue = np.mean(first_slices[x_ind, y_ind])

        x_ind, y_ind = np.where(first_mask==0)
        f_noise = np.mean(first_slices[x_ind, y_ind])
        
        f_signal = np.mean(f_tissue)
        f_noise = np.mean(f_noise)
        f_SNR = f_signal/f_noise

        #display the normalized images
        orig_b1 = first_slices_all[0, :, :]
        orig_b1 = np.reshape(orig_b1, (96 * 96, 1))
        temp_simu = np.unique(orig_b1)
        orig_b1_max = temp_simu[int(len(temp_simu) * 0.99)]

        first_slices_all = np.clip(first_slices_all, 0, orig_b1_max)
        first_slices_all = first_slices_all / orig_b1_max
        
        
        plt.figure(count, dpi=200)
        plt.suptitle(name + '\n 10th Slice b3 SNR: {}. \n All slices only b3 SNR: {}'.format(f_SNR, SNR))
        plt.subplots_adjust(left=0.01, bottom=0.07, right=0.9, top=0.99, wspace=0.5, hspace=0.02)
        
        plt.subplot(1,6,1)
        plt.axis('off')
        plt.imshow(first_slices_all[0], cmap='gray')
        plt.title("image b1", fontsize=8)
        cb=plt.colorbar(fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=5)
        
        plt.subplot(1,6,2)
        plt.axis('off')
        plt.imshow(first_slices_all[1], cmap='gray')
        plt.title("image b2", fontsize=8)
        cb=plt.colorbar(fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=5)
        
        plt.subplot(1,6,3)
        plt.axis('off')
        plt.imshow(first_slices_all[2], cmap='gray')
        plt.title("image b3", fontsize=8)
        cb=plt.colorbar(fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=5)
        
        plt.subplot(1,6,4)
        plt.axis('off')
        plt.imshow(first_slices_all[3], cmap='gray')
        plt.title("image b4", fontsize=8)
        cb=plt.colorbar(fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=5)
        
        plt.subplot(1,6,5)
        plt.axis('off')
        plt.imshow(first_slices_all[4], cmap='gray')
        plt.title("image b5", fontsize=8)
        cb=plt.colorbar(fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=5)

        plt.subplot(1, 6, 6)
        plt.axis('off')
        plt.imshow(cur_adc[interm], cmap='hot')
        plt.title("adc map", fontsize=8)
        cb = plt.colorbar(fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=5)
        
#        line = name + ' : SNR = ' + str(SNR)
#        print(line)
        
        pdf.savefig(count)
        count += 1
    
    pdf.close()
    
#%%
def getSNR_ALL():
    #%%
    from scipy import ndimage
    
    name = 'M02'
    
    datadir = '/home/ym/Desktop/research/titanserver/mnt/4TBHDD/yli/DWI_data/image_complex/1x/'
    adcdir = '/home/ym/Desktop/research/titanserver/mnt/4TBHDD/yli/DWI_data/image_complex/adc_3b_noc/'
    datalist = '/home/ym/Desktop/research/titanserver/home/yli/DWI_project/DWI_new_process/datalists/image_list'
    
    datadir_4x = '/home/ym/Desktop/research/titanserver/mnt/4TBHDD/yli/DWI_data/image_complex/4x/'
    adcdir_4x = '/home/ym/Desktop/research/titanserver/mnt/4TBHDD/yli/DWI_data/image_complex/adc_3b_noc_4x/'
    
    pred_ADC = '/home/ym/Desktop/research/titanserver/home/yli/DWI_project/DWI_new_process/checkpoints/DWI_adc_4x_v3_DU_both_w0.05_Apr30/visualization/test/pred_ADC_'+name+'.npy'
    orig_ADC = '/home/ym/Desktop/research/titanserver/home/yli/DWI_project/DWI_new_process/checkpoints/DWI_adc_4x_v3_DU_both_w0.05_Apr30/visualization/test/orig_ADC_'+name+'.npy'
    
    namefile = open(datalist, 'r')
    datanames = namefile.readlines()
    datanames = [f[:-1] for f in datanames]
    
    
    out_file = '/home/ym/Desktop/research/titanserver/home/yli/DWI_project/DWI_new_process/checkpoints/DWI_adc_4x_v3_DU_both_w0.05_Apr30/visualization/test/'+name+'_SNR_analysis.pdf'
    pdf = matplotlib.backends.backend_pdf.PdfPages(out_file)
    
    upper_thres=0.0032
    lower_thres = 0.000032
    
    pred_ADC_npy = np.load(pred_ADC)
    
    for num in range(len(pred_ADC_npy)):
        cur_datas = np.load(os.path.join(datadir, name + '_cx_image_data.npy'))
        cur_data = cur_datas[num,:,:,:]
        cur_adc = np.load(orig_ADC)[num]
        cur_data = np.absolute(cur_data)
        
        cur_datas_4x = np.load(os.path.join(datadir_4x, name + '_cx_image_data_4x.npy'))
        cur_data_4x = cur_datas_4x[num,:,:,:]
        cur_adc_4x = np.load(os.path.join(adcdir_4x, name + '_Diffusion_Fits_3b_noc_4x.npy'))[1][num]
        cur_data_4x = np.absolute(cur_data_4x)
        
#        cur_adc[cur_adc>=upper_thres] = 0
#        cur_adc = ndimage.median_filter(cur_adc, size=3)
#        cur_adc[cur_adc<lower_thres] = 0
#        
#        cur_adc_4x[cur_adc_4x>=upper_thres] = 0
#        cur_adc_4x = ndimage.median_filter(cur_adc_4x, size=3)
#        cur_adc_4x[cur_adc_4x<lower_thres] = 0
        
        bool_min = cur_adc>np.min(cur_adc)
        bool_max = cur_adc<np.max(cur_adc)
        mask = (bool_min * bool_max).astype(int)
            
        cur_pred_ADC = pred_ADC_npy[num]
        
        adc_diff = (np.abs(cur_adc - cur_pred_ADC) / (cur_adc+0.0000001))
        adc_diff[adc_diff == adc_diff[0, 0]] = 0
        
        x_ind, y_ind = np.where(mask == 1)
        pred_ADC_ind = cur_pred_ADC[x_ind, y_ind]
        orig_ADC_ind = cur_adc[x_ind, y_ind]
        
        adc_diff_ind = (np.abs(orig_ADC_ind - pred_ADC_ind) / (orig_ADC_ind))
        adc_diff_ind = np.clip(adc_diff_ind, 0, 1)
        
        for ind in range(5):
            cur_slice =cur_data[ind]
            
            x_ind, y_ind = np.where(mask==1)
            signal = np.mean(cur_slice[x_ind, y_ind])
    
            x_ind, y_ind = np.where(mask==0)
            noise = np.mean(cur_slice[x_ind, y_ind])
        
            SNR = signal/noise
        
            plt.figure(num, dpi=200)
            plt.subplot(3,5,ind+1)
            plt.axis('off')
            plt.imshow(cur_slice, cmap='gray')
            plt.title('b{} SNR: {:.2f}'.format(ind, SNR), fontsize=6)
            
            #####################this is for downsampled####################
            cur_slice_4x =cur_data_4x[ind]
            
            bool_min = cur_adc>np.min(cur_adc)
            bool_max = cur_adc<np.max(cur_adc)
            mask = (bool_min * bool_max).astype(int)
            
            x_ind, y_ind = np.where(mask==1)
            signal = np.mean(cur_slice_4x[x_ind, y_ind])
    
            x_ind, y_ind = np.where(mask==0)
            noise = np.mean(cur_slice_4x[x_ind, y_ind])
        
            SNR = signal/noise
        
            plt.figure(num, dpi=200)
            plt.subplot(3,5,6+ind)
            plt.axis('off')
            plt.imshow(cur_slice_4x, cmap='gray')
            plt.title('b{} SNR: {:.2f}'.format(ind, SNR), fontsize=6)
            
        plt.figure(num, dpi=200)
        plt.subplot(3,5,11)
        plt.axis('off')
        plt.imshow(cur_adc*0.0032, cmap='hot', vmin=0, vmax=0.0032)
        plt.title('full sampled ADC', fontsize=6)
#        cb=plt.colorbar(fraction=0.046, pad=0.04)
#        cb.ax.tick_params(labelsize=5)
        
        plt.subplot(3,5,12)
        plt.axis('off')
        plt.imshow(cur_adc_4x, cmap='hot', vmin=0, vmax=0.0032)
        plt.title('under sampled ADC', fontsize=6)
#        cb=plt.colorbar(fraction=0.046, pad=0.04)
#        cb.ax.tick_params(labelsize=5)
        
        plt.subplot(3,5,13)
        plt.axis('off')
        plt.imshow(cur_pred_ADC*0.0032, cmap='hot', vmin=0, vmax=0.0032)
        plt.title('predict ADC', fontsize=6)
#        cb=plt.colorbar(fraction=0.046, pad=0.04)
#        cb.ax.tick_params(labelsize=5)
        
        plt.subplot(3,5,14)
        plt.axis('off')
        plt.imshow(adc_diff*mask, cmap='hot', vmin=0, vmax=0.5)
        plt.title('percent diff: {}'.format(round(np.mean(adc_diff_ind), 3)), fontsize=6)
        cb=plt.colorbar(fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=5)
        
        pdf.savefig(num)
    
    pdf.close()
    
#%%
    
        

def getSNR_matrix():
    datadir = '/home/ym/Desktop/research/titanserver/mnt/4TBHDD/yli/DWI_data/image_complex/1x/'
    adcdir = '/home/ym/Desktop/research/titanserver/mnt/4TBHDD/yli/DWI_data/image_complex/adc_3b_noc/'
    datalist = '/home/ym/Desktop/research/titanserver/home/yli/DWI_project/DWI_new_process/datalists/image_list'
    
    namefile = open(datalist, 'r')
    datanames = namefile.readlines()
    datanames = [f[:-1] for f in datanames]
    
    binds = [0,1,2,3,4]
    
    all_SNR = []
    for bind in binds:
        
        SNR_list = []
        for name in datanames:
            cur_datas = np.load(os.path.join(datadir, name + '_cx_image_data.npy'))
            cur_adcs = np.load(os.path.join(adcdir, name + '_Diffusion_Fits_3b_noc.npy'))[1]
            
            SNR_slice = []
            for ii in range(len(cur_datas)):
                cur_data = cur_datas[ii,bind,:,:]
                cur_adc = cur_adcs[ii]
                
                cur_data = np.absolute(cur_data)
                
                bool_min = cur_adc>np.min(cur_adc)
                bool_max = cur_adc<np.max(cur_adc)
                mask = (bool_min * bool_max).astype(int)
                
                x_ind, y_ind = np.where(mask==1)
                signal = np.mean(cur_data[x_ind, y_ind])
        
                x_ind, y_ind = np.where(mask==0)
                noise = np.mean(cur_data[x_ind, y_ind])
                
                SNR = signal/noise
                SNR_slice.append(SNR)
            SNR_list.append(SNR_slice)
        all_SNR.append(SNR_list)
    all_SNR = np.asarray(all_SNR)
    
    cases = np.arange(54)*2.25
    SNR_mean = np.mean(all_SNR, axis=2)
    SNR_std = np.std(all_SNR, axis=2)
    
    SNR_list = SNR_mean[0]
    sortedind = sorted(range(len(SNR_list)), key=lambda k: SNR_list[k], reverse=True)
    
    sort_SNR_mean = SNR_mean[:,sortedind]
    sort_SNR_std = SNR_std[:,sortedind]
    
    
    fig, ax = plt.subplots(figsize=[25, 10], dpi=200)
    width = 0.25
    
    rects1 = ax.bar(cases-2.25, sort_SNR_mean[0], yerr=sort_SNR_std[0], label='b1', width = 0.4)
    rects2 = ax.bar(cases-1.85, sort_SNR_mean[1], yerr=sort_SNR_std[1], label='b2', width = 0.4)
    rects3 = ax.bar(cases-1.45, sort_SNR_mean[2], yerr=sort_SNR_std[2], label='b3', width = 0.4)
    rects4 = ax.bar(cases-1.05, sort_SNR_mean[3], yerr=sort_SNR_std[3], label='b4', width = 0.4)
    rects5 = ax.bar(cases-0.65, sort_SNR_mean[4], yerr=sort_SNR_std[4], label='b5', width = 0.4)
    
    
#    plt.plot(cases, np.ones(54)*3.4, 'r', label="3.4line")
#    plt.errorbar(cases, sort_SNR_mean[0], sort_SNR_std[0], linestyle='None', marker='o', label="b1")
#    plt.errorbar(cases, sort_SNR_mean[1], sort_SNR_std[1], linestyle='None', marker='o', label="b2")
#    plt.errorbar(cases, sort_SNR_mean[2], sort_SNR_std[2], linestyle='None', marker='o', label="b3")
#    plt.errorbar(cases, sort_SNR_mean[3], sort_SNR_std[3], linestyle='None', marker='o', label="b4")
#    plt.errorbar(cases, sort_SNR_mean[4], sort_SNR_std[4], linestyle='None', marker='o', label="b5")
    
    plt.title('SNR for all bvalues')
    plt.legend()
    plt.savefig('/home/ym/Desktop/research/titanserver/home/yli/DWI_project/DWI_new_process/snr_bval.png')
    
    





def add_SNR():
    #%%
    image_dir = '/home/ym/Desktop/research/titanserver/home/yli/DWI_project/DWI_new_process/checkpoints/DWI_adc_unpool_adc0.0035/visualization/vali/M07_pre_curve_fit_img.npy'
    image_slices = np.load(image_dir)
    image_slice = image_slices[11]
    
    b_val = 2
    
#    SNR = [5, 6, 7, 9, 11, 13, 15]
#    SNR = [3, 4, 5, 6, 7, 8, 10]
    SNR = [1.63, 2, 3, 4, 5, 6, 7]
    
    plt.figure(dpi=400)
    plt.subplot(4, 4, 1)
    plt.imshow(image_slice[b_val], cmap='gray')
    plt.xticks([]), plt.yticks([]), plt.title('image', fontsize=5)
    
    count =2
    for snr in SNR:
        rg = np.random.RandomState(123)
        noise = rg.normal(0, 1 / snr, (96, 96))
        
        bool_b12 = image_slice[0,:,:] > image_slice[1,:,:]
        bool_b23 = image_slice[1,:,:] > image_slice[2,:,:]
        mask = (bool_b12 * bool_b23).astype(int)
        
        x_ind, y_ind = np.where(mask==1)
        
        cur_img = image_slice[b_val]
        
        noise_mean = np.mean(cur_img[x_ind, y_ind]) / snr
        image_map = image_slice[b_val] + noise * noise_mean
        
        plt.subplot(4, 4, count)
        plt.imshow(image_map, cmap='gray')
        plt.xticks([]), plt.yticks([]), plt.title(snr, fontsize=5)
        
        plt.subplot(4, 4, count + 8)
        plt.imshow(noise * noise_mean, cmap='gray', vmin = 0, vmax = np.mean(cur_img))
        plt.xticks([]), plt.yticks([]), plt.title(round(noise_mean), fontsize=5)
        
        count += 1
    
    
    plt.savefig('/home/ym/Desktop/research/titanserver/home/yli/DWI_project/DWI_new_process/simu_snrs.jpg')


def diffusion_fit():
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
    
    #from PIL import Image
    # R-G-B-A (A = alpha or opacity: 1=opaque, 0=transparent)
    
    # Define the diffusion function to fit
    #def func(x, a, b, c):
    #    return a * np.exp(-b * x) + c
    
    def func(x, a, b):
        return a * np.exp(-b * x) 
        
    print("Start =", datetime.now())
    
    xres = 96
    yres = 96
    slices = 19
    b_array = np.array([24.25, 536.86, 1072.62]) # 3 b-values
    
    data_dir = '/mnt/4TBHDD/yli/DWI_data/image_complex/adc_params_k3/'
    save_dir = '/mnt/4TBHDD/yli/DWI_data/image_complex/heekown_adc_max1/'
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    dir_to_analyze = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if 'img' in f]
    
    for k in range(0, len(dir_to_analyze)):
        current_dir = dir_to_analyze[k]
        imgarr = np.load(current_dir)
        
        adcarr = np.load(current_dir[:-7] + 'param.npy')
        
        fitval = np.zeros((4,slices,yres,xres)) # Fit all slices
    
        for sl in range(0, slices):
    
            print("Slice = ", sl)
            img = imgarr[sl,:,:,:]
            adc = adcarr[sl,1,:,:]
            
#            ###########################add noise###############################################
#            rg = np.random.RandomState(123)
#            b1_noise = 4.47
#            b2_noise = 2.24
#            b3_noise = 1.63
#            
#            noise_b1 = rg.normal(0, 1 / b1_noise, (96, 96))
#            noise_b2 = rg.normal(0, 1 / b2_noise, (96, 96))
#            noise_b3 = rg.normal(0, 1 / b3_noise, (96, 96))
#            
#            mask = (adc > 0).astype(int)
#            x_ind, y_ind = np.where(adc>0)
#            
#            cur_b1 = img[0]
#            noise_mean = np.mean(cur_b1[x_ind, y_ind]) / b1_noise
#            img[0] = cur_b1 + noise_b1 * noise_mean
#            img[0] = img[0] * mask
#            
#            cur_b2 = img[1]
#            noise_mean = np.mean(cur_b2[x_ind, y_ind]) / b2_noise
#            img[1] = cur_b2 + noise_b2 * noise_mean
#            
#            cur_b3 = img[2]
#            noise_mean = np.mean(cur_b3[x_ind, y_ind]) / b3_noise
#            img[2] = cur_b3 + noise_b3 * noise_mean
#            ###########################add noise###############################################
            
            maxval = np.amax(np.absolute(img))
            img = img/maxval * 100 # Normalize to 100 since absolute numbers are not meaningful

            noise_mean = np.average(img[0,0:10,0:10])
            fit_threshold = 5*noise_mean
    
            error_count = 0
            for j in range(0,yres):
                for i in range(0,xres):
                    yn = img[:,j,i]
    #                yn = img[i,j,:]
                    if (yn[0] > fit_threshold):
                        init_bval = -np.log(yn[1]/yn[0])/(b_array[1]-b_array[0])
                        try:
                            popt, pcov = curve_fit(func, b_array, yn, p0 = [(yn[0]-yn[2]), init_bval]) #this is for 3 b-values
                            fitval[0:2,sl,j,i] = popt
                        except RuntimeError:
                            print("Error - curve_fit failed. i,j = ", i, j)
                            error_count += 1
                            fitval[:,sl,j,i] = [0,0,0,0.5] # Assign to certain color if fit doesn't converge
                    else: fitval[:,sl,j,i] = [0,0,0,0.8] # Assign to certain color if signal is too low to fit
    #                        fitval[i,j,sl,:] = [0,0,0,0.5] # Assign to certain color if fit doesn't converge
    #                else: fitval[i,j,sl,:] = [0,0,0,0.8] # Assign to certain color if signal is too low to fit
                     
        # Set min/max to reasonable values
        fitval[0,:,:,:] = np.clip(fitval[0,:,:,:],0,100) # Max of 100% of first b-value signal for coefficient
        fitval[1,:,:,:] = np.clip(fitval[1,:,:,:],0,0.005) # Max diffusion value of 0.005
        fitval[2,:,:,:] = np.clip(fitval[2,:,:,:],0,100) # Max of 100% of first b-value for baseline
        
        # Save results to file
        split_name = current_dir.split('/')[-1][:-7] + 'HK_max1.npy'
        np.save(os.path.join(save_dir, split_name), fitval)
        
        
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
        
        #################################################################################################################
        plt.figure(num=3, figsize=(6,5))
        plt.subplots_adjust(left=0.01, bottom=0.07, right=0.99, top=0.99, wspace=0.02, hspace=0.02)
        for i in range(0, slices):
            plt.subplot(4,5,i+1)
            plt.axis('off')
            
            bool_min = fitval[1,i,:,:]>0
            bool_max = fitval[1,i,:,:]<=0.0032
            mask = (bool_min * bool_max).astype(int)
            
            plt.imshow(mask * fitval[1,i,:,:], cmap='hot', vmin = 0, vmax = 0.005)
    #        plt.imshow(fitval[:,:,i,1], cmap='hot', vmin = 0, vmax = 0.005)
        cb=plt.colorbar(fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=7)
        plt.annotate("Masked adc",xy=(5,5), xycoords = "figure points")
        #################################################################################################################
        
        
        out_file = current_dir.split('/')[-1][:-7] + 'Diffusion_Results_max1.pdf'
        pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(save_dir, out_file))
        pdf.savefig(1) # b=0 image of all slices
        pdf.savefig(2) # Diffusion coeffients of all slices
        pdf.savefig(3) # Mask of adc map
    
        for i in range(0, slices):
            
            plt.figure(num=i+4, figsize=(7,2))
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
    

def cal_adc_diff():
    
    data_dir = '/home/ym/Desktop/research/titanserver/mnt/4TBHDD/yli/DWI_data/image_complex/adc_params_k3/'
    pred_dir = '/home/ym/Desktop/research/titanserver/mnt/4TBHDD/yli/DWI_data/image_complex/heekown_adc_SNR3/'
    
#    data_names = [f[:-18] for f in os.listdir(data_dir) if 'img' in f]
    data_names = ['M45_pre', 'M45_post']
    b_array = np.array([24.25, 536.86, 1072.62]) # 3 b-values
    
    total_diff_adc = 0
    total_diff_S0 = 0
    total_diff_b1 = 0
    total_diff_b2 = 0
    total_diff_b3 = 0
    count = 0
    for name in data_names:
        
        cur_params = np.load(data_dir + name + '_curve_fit_param.npy')
        pred_params = np.load(pred_dir + name + '_curve_fit_HK.npy')
        
        cur_adcs = cur_params[:,1,:,:]
        pred_adcs = pred_params[1]
        
        cur_S0s = cur_params[:,0,:,:]
        pred_S0s = pred_params[0]
        
        for ii in range(len(cur_adcs)):
            cur_adc = cur_adcs[ii]
            pred_adc = pred_adcs[ii]
            
            cur_S0 = cur_S0s[ii]
            pred_S0 = pred_S0s[ii]/100
            
            gt_bs = cur_S0[np.newaxis,:,:] * np.exp(b_array[:,np.newaxis,np.newaxis] * cur_adc[np.newaxis,:,:])
            pred_bs = pred_S0[np.newaxis,:,:] * np.exp(b_array[:,np.newaxis,np.newaxis] * pred_adc[np.newaxis,:,:])
            
            nanmask = (1-np.isnan(pred_adc)).astype(int)
            adc_mask = (cur_adc>0).astype(int)
            
            x_ind, y_ind = np.where(nanmask*adc_mask)
#            x_ind, y_ind = np.where(adc_mask)
            
            diff_adc = np.sum(np.abs(cur_adc[x_ind, y_ind] - pred_adc[x_ind, y_ind]))
            diff_S0 = np.sum(np.abs(cur_S0[x_ind, y_ind] - pred_S0[x_ind, y_ind]))
            diff_b1 = np.sum(np.abs(gt_bs[0, x_ind, y_ind] - pred_bs[0, x_ind, y_ind]))
            diff_b2 = np.sum(np.abs(gt_bs[1, x_ind, y_ind] - pred_bs[1, x_ind, y_ind]))
            diff_b3 = np.sum(np.abs(gt_bs[2, x_ind, y_ind] - pred_bs[2, x_ind, y_ind]))
            
            total_diff_adc += diff_adc
            total_diff_S0 += diff_S0
            total_diff_b1 += diff_b1
            total_diff_b2 += diff_b2
            total_diff_b3 += diff_b3
            
            count +=len(cur_adc[x_ind, y_ind])
    
    total_diff_b1/count
    total_diff_b2/count
    total_diff_b3/count
    total_diff_adc/count
    total_diff_S0/count
    
    #%%
#     adc_diff[np.isnan(adc_diff)]=0
#     
#     adc_mask = adc_diff>0.1
#     fig, (ax1) = plt.subplots(1, 1, figsize=(15,15))
#     im_in = ax1.imshow(adc_mask, "hot")#, vmin = 0, vmax = 1)
#     ax1.set_xticks([]), ax1.set_yticks([]), ax1.set_title("pred_ADC", size=10)
#     divider = make_axes_locatable(ax1)
#     cax = divider.append_axes("right", size="5%", pad=2)
#     cbar = fig.colorbar(im_in, cax=cax)
#     cbar.ax.tick_params(labelsize=10)
#     
#     orig_adc = np.load('/home/ym/Desktop/research/titanserver/home/yli/DWI_project/DWI_new_process/checkpoints/DWI_adc_4x_v3_perent_L1_res_bl5_fl128_twosample_Apr6/visualization/test/orig_adc.npy')
#     pred_adc = np.load('/home/ym/Desktop/research/titanserver/home/yli/DWI_project/DWI_new_process/checkpoints/DWI_adc_4x_v3_perent_L1_res_bl5_fl128_twosample_Apr6/visualization/test/pred_adc.npy')
#     
#     mask = orig_adc>0
#     pred_adc = pred_adc * mask
#     
#     
#     M62_adc = np.load('/home/ym/Desktop/research/titanserver/mnt/4TBHDD/yli/DWI_data/image_complex/adc_3b_noc_4x/M62_Diffusion_Fits_noC_4x.npy')
#     M62_adc = np.load('/home/ym/Desktop/research/titanserver/mnt/4TBHDD/yli/DWI_data/image_complex/adc_3b_noc/M62_Diffusion_Fits_3b_noc.npy')
#     M62_adc = M62_adc[1]
#     
#     M62_adc1 = M62_adc[0]
#     M62_adc2 = M62_adc1.copy()
#     M62_adc2[M62_adc2>0.003199] = 0
#     
#     adc2_mediam = ndimage.median_filter(M62_adc2, size=3)
#     
#     np.unique(M62_adc1)
#     np.mean(M62_adc2)
#     
#     from scipy.ndimage.measurements import label
#     labeled, ncomponents = label(adc2_mediam)
#     
#     plt.figure(dpi=400)
#     plt.subplot(131), plt.imshow(M62_adc1, "hot", vmin = 0, vmax = 0.0032), plt.title("Original ADC")
#     plt.subplot(132), plt.imshow(M62_adc2, "hot", vmin = 0, vmax = 0.0032), plt.title("filtered ADC")
#     plt.subplot(133), plt.imshow(adc2_mediam, "hot", vmin = 0, vmax = 0.0032), plt.title("Median filter")
#     plt.savefig('adc_demo.png')
    
def rotate_imgs():
    img_dir_4x = '/home/ym/Desktop/research/titanserver/mnt/4TBHDD/yli/DWI_data/image_complex/4x'
    img_dir_1x = '/home/ym/Desktop/research/titanserver/mnt/4TBHDD/yli/DWI_data/image_complex/1x'
    adc_dir_4x = '/home/ym/Desktop/research/titanserver/mnt/4TBHDD/yli/DWI_data/image_complex/adc_3b_noc_4x'
    adc_dir_1x = '/home/ym/Desktop/research/titanserver/mnt/4TBHDD/yli/DWI_data/image_complex/adc_3b_noc'

    # rot_id = [#'M89',
    #           'M86',
    #           'M88',
    #           'M84',
    #           'M66',
    #           'M69',
    #           'M70',
    #           'M68',
    #           'M76',
    #           'M63',]

    rot_id = ['M75',
              'M65',
              'M74',
              'M95',
              'M80',
              'M83',
              'M79',
              'M87',
              'M78',
              'M81',
              'M90',
              'M93',
              ]

    for name in rot_id:
        img_1x = np.load(os.path.join(img_dir_1x, name + '_cx_image_data.npy'))
        img_4x = np.load(os.path.join(img_dir_4x, name + '_cx_image_data_4x.npy'))

        adc_1x = np.load(os.path.join(adc_dir_1x, name + '_Diffusion_Fits_3b_noc.npy'))
        adc_4x = np.load(os.path.join(adc_dir_4x, name + '_Diffusion_Fits_3b_noc_4x.npy'))

        rot_img_1x = np.rot90(img_1x, k=3, axes=(2, 3))
        rot_img_4x = np.rot90(img_4x, k=3, axes=(2, 3))
        rot_adc_1x = np.rot90(adc_1x, k=3, axes=(2, 3))
        rot_adc_4x = np.rot90(adc_4x, k=3, axes=(2, 3))

        # plt.imshow(np.abs(rot_img_1x[0][0]), 'gray'), plt.show()
        # plt.imshow(np.abs(rot_adc_1x[1][0]), 'hot'), plt.show()
        np.save(os.path.join(img_dir_1x, name + '_cx_image_data.npy'), rot_img_1x)
        np.save(os.path.join(img_dir_1x, name + '_cx_image_data_4x.npy'), rot_img_4x)
        np.save(os.path.join(img_dir_1x, name + '_Diffusion_Fits_3b_noc.npy'), rot_adc_1x)
        np.save(os.path.join(img_dir_1x, name + '_Diffusion_Fits_3b_noc_4x.npy'), rot_adc_4x)


def stack_images():
    '''
    This is only for generated DWI outputs
    Returns:

    '''
    import re

    img_dir = '/home/ym/Desktop/research/titanserver/home/yli/DWI_project/DWI_new_process/checkpoints/DU_ADC_outdwi_Jun28/visualization/test'
    image_files = [o for o in os.listdir(img_dir)] #os.path.join(img_dir, o)

    image_files = [f for f in image_files if '.npy' in f]

    sorted_img = [f[:-20] for f in image_files]
    sorted_img = list(set(sorted_img))

    for ii in range(len(sorted_img)):
        cur_name = sorted_img[ii]
        check_ind = [f for f in image_files if cur_name in f]

        dwi_img = np.zeros((len(check_ind), 3, 96, 96))

        for jj in range(len(check_ind)):
            cur_str = check_ind[jj]
            cur_ind = int(re.findall(r"\d+", cur_str)[1])

            cur_img = np.load(os.path.join(img_dir, cur_str))
            dwi_img[cur_ind] = cur_img
            os.remove(os.path.join(img_dir, cur_str))

        out_name = cur_str[:-6]
        np.save(os.path.join(img_dir, out_name), dwi_img)
        print('finished ' + out_name)

def check_masks():
    dwi_mask_dir = '/mnt/4TBHDD/yli/DWI_data/all_dwimasks_upto262'
    dwi_adc_dir = '/mnt/4TBHDD/yli/DWI_data/reconstructed_adcs_3b_noc/'
    image_files = [o for o in os.listdir(dwi_mask_dir)]

    datadir = '/mnt/4TBHDD/yli/DWI_data/codes_preprocess/all_lists'
    namefile = open(datadir, 'r')
    data_list = namefile.readlines()
    data_list = [f[:-1] for f in data_list]

    dwi_mask_list = []
    for name in data_list:
        print('start name {}'.format(name))
        adc_img = np.load(os.path.join(dwi_adc_dir, name + '_Diffusion_Fits_3b_noc.npy'))
        _, dim, _, _ = adc_img.shape

        dwi_mask = os.path.join(dwi_mask_dir, name + '_mask_dwi.raw')

        if os.path.exists(dwi_mask):
            mask_npy = read_raw(binary_file_name=dwi_mask, image_size=[96, 96, dim])

            save_name = os.path.join(dwi_mask_dir, name + '_mask_dwi.npy')
            np.save(save_name, mask_npy)

            dwi_mask_list.append(name)

            print('finished name {}'.format(name))
        else:
            print('skipped slice with name {}'.format(name))

    savefile = '/mnt/4TBHDD/yli/DWI_data/codes_preprocess/mask_lists'
    file = open(savefile, 'w')
    for item in dwi_mask_list:
        file.writelines(item + '\n')
    file.close()



def remove_phamtom():
    imagedir_1x = '/home/ym/Desktop/research/titanserver/mnt/4TBHDD/yli/DWI_data/image_complex/1x/M149_cx_image_data.npy'
    imagedir_4x = '/home/ym/Desktop/research/titanserver/mnt/4TBHDD/yli/DWI_data/image_complex/4x/M149_cx_image_data_down.npy'
    adcdir_1x = '/home/ym/Desktop/research/titanserver/mnt/4TBHDD/yli/DWI_data/image_complex/adc_3b_noc/M149_Diffusion_Fits_3b_noc.npy'
    adcdir_4x = '/home/ym/Desktop/research/titanserver/mnt/4TBHDD/yli/DWI_data/image_complex/adc_3b_noc_4x/M149_Diffusion_Fits_3b_noc_down.npy'

    image_4x = np.load(imagedir_4x)
    image_1x = np.load(imagedir_1x)
    adc_1x = np.load(adcdir_1x)
    adc_4x = np.load(adcdir_4x)

    plt.imshow(adc_1x[1, 0, :, :]), plt.show()

    mask = np.zeros((96, 96), dtype=np.int32)
    mask[33:, :] = 1
    plt.imshow(mask), plt.show()

    mask = mask[np.newaxis, np.newaxis]

    image_4x = image_4x * mask
    image_1x = image_1x * mask
    adc_1x = adc_1x * mask
    adc_4x = adc_4x * mask

    plt.imshow(adc_1x[1, 0, :, :]), plt.show()

    np.save(imagedir_1x, image_1x)
    np.save(imagedir_4x, image_4x)
    np.save(adcdir_1x, adc_1x)
    np.save(adcdir_4x, adc_4x)


if __name__ == '__main__':
     
     
     # Read the image using both big and little endian
#     img_dir = '/home/ym/Desktop/research/titanserver/home/yli/DWI_project/DataHighRes/s_2019080801/image/image0018.raw'
#     image_orig = read_raw(binary_file_name=img_dir)
#     orig_real = (image_orig - np.min(image_orig))/(np.max(image_orig) - np.min(image_orig))
#     
#     plt.imshow(orig_real, cmap='gray')
     
     
     # datadir = '/mnt/4TBHDD/yli/DWI_data/image_complex/1x'
     # savefile = '/home/yli/DWI_project/DWI_new_process/datalists'
     # make_data_list(datadir, savefile)
#     image_visual()
#     gen_multi_list()
#     preprocess_complex_img()
#     adc_clip_mediam_filter()
#     data_name = 'M01_AllSlicesBvalues.bin'
#     adc_name = 'M01_Diffusion_Fits_3b_noc.npy'
#     save_name = "Residue_map_3b_noc.pdf"
#     calc_residule(data_name, adc_name, save_name)
#      getSNR()
     
#     diffusion_fit()
#      preprocess_complex_img()
#      rotate_imgs()

     # stack_images()
     check_masks()

     print('done')
     
     
     
     
     
     