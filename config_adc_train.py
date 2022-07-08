# coding:utf8
import warnings
import torch as t
import numpy as np

class DefaultConfig(object):

    env = 'Model_4x'  # visdom environment
    list_dir = './datalists/'
    data_dir = 'DIRECTORY/fullsampled_images/' #directory of fullsampled_images
    data_4x_dir = 'DIRECTORY/downsampled_images/' #directory of downsampled_images

    adc_dir = 'DIRECTORY/fullsampled_adcs/' #directory of fullsampled_adcs
    adc_4x_dir = 'DIRECTORY/downsampled_adcs/' #directory of downsampled_adcs

    load_model_path = './checkpoints/' + env + '/' # path of pretrain model
    load_saved_model = None #'4x_checkpoint.pth.tar'


    batch_size = 32 # batch size
    val_batch_size = 10
    use_gpu = True  # user GPU or not
    num_workers = 10  # how many workers for loading data

    max_epoch = 1001
    lr = 0.00001
    lr_list = np.logspace(-2, -4, 800)
    lr_decay = 0.5  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 1e-4  # 
    b_array = t.tensor([24.25, 536.86, 1072.62, 1482.07, 2144.69])

    def _parse(self, kwargs):
        """
        update parameter in config file
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        opt.device = t.device('cuda') if opt.use_gpu else t.device('cpu')

opt = DefaultConfig()
