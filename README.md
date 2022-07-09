# Learning Apparent Diffusion Coefficient Maps from Undersampled Radial k-Space Diffusion-Weighted MRI in Mice using a Deep CNN-Transformer Model in Conjunction with a Monoexponential Model

----------------------------------------------------------------------------------------------------------------------------------------
## Preparation
This research provides two strategies to train the CNN for generating high-quality ADC maps:
1. With masks generated from fully-sampled ADC scans
2. Without masks generated from fully-sampled ADC scans

For strategy 1, we will apply the generated masks to predicted ADC maps prior feeding the ADC prediction to loss functions. 
In this way the loss function will only focus on the tissue region while ignoring the rest of backgrounds/noises. Our paper is implemented based on this strategy.

For strategy 2, we will use the predicted ADC maps, including background and noises, to calculate the losses. 
In this way the CNN will yield less accurate performance compared to strategy 1. However, this strategy would be useful to test on the DW images without fully sampled ADC maps as ground truth.

----------------------------------------------------------------------------------------------------------------------------------------
#### Files associated with Strategy 1:
This is for training and validating, need ground truth
{config_adc_mask.py, train_adc_mask.py, validate_adc_mask.py}

#### Files associated with Strategy 2:
This is for training and validating, need ground truth
{config_adc_train.py, train_adc.py, validate_adc.py}

This is for testing only, no ground truth needed
{config_adc_test.py, test_adc.py}

----------------------------------------------------------------------------------------------------------------------------------------
All file directories need to be provided in the config_adc_XXX.py file by replacing "DIRECTORY" with the correct dataset folders. The directory needs to be changed in config file:
data_dir: fully sampled DW images directory
data_4x_dir: undersampled DW images directory
adc_dir: fully sampled ADC maps directory
adc_4x_dir : undersampled ADC maps directory

----------------------------------------------------------------------------------------------------------------------------------------
Using M01 dataset as an example, Our DW images follows the format "M01_cx_image_data.npy" for fully sampled data, and "M01_cx_image_data_downsampled.npy" for undersampled data. 
Our ADC maps follows the format "M01_Diffusion_Fits_2param.npy" for both fully sampled and undersampled images. Those names could be changed in the "./Data/DWI_loader_XXX.py" to apply the codes on your own dataset.

The image lists are saved under directory "../datalists/" with the format:
M01/n M02/n ...

----------------------------------------------------------------------------------------------------------------------------------------
## Training:
All parameters, learning rates, batch size can be tuned in config file.
Change load_saved_model=None in config file before start training.

#### To start a new training with strategy 1, use:
CUDA_VISIBLE_DEVICES=0 python train_adc_mask.py

#### To start a new training with strategy 2, use:
CUDA_VISIBLE_DEVICES=0 python train_adc.py

----------------------------------------------------------------------------------------------------------------------------------------
## Validation:
Change load_saved_model='4x_checkpoint.pth.tar' (the saved model name) in config file before start Validation and testing.

#### For running the validation with strategy 1, use:
CUDA_VISIBLE_DEVICES=0 python validate_adc_mask.py

#### For running the validation with strategy 2, use:
CUDA_VISIBLE_DEVICES=0 python validate_adc.py

----------------------------------------------------------------------------------------------------------------------------------------
## Testing:
#### For runnning the testing, with strategy 2, use:
CUDA_VISIBLE_DEVICES=0 python test_adc.py


----------------------------------------------------------------------------------------------------------------------------------------
## Reference:
[Paper](https://arxiv.org/abs/2207.02399/)
```
@article{li2021acenet,
  title={Learning Apparent Diffusion Coefficient Maps from Undersampled Radial k-Space Diffusion-Weighted MRI in Mice using a Deep CNN-Transformer Model in Conjunction with a Monoexponential Model},
  author={Li, Yuemeng and Song, Hee Kwon and Joaquim, Miguel Romanello and Pickup, Stephen and Zhou, Rong and Fan, Yong},
  doi = {10.48550/ARXIV.2207.02399},
  year={2022},
  publisher={arXiv}
}
```


