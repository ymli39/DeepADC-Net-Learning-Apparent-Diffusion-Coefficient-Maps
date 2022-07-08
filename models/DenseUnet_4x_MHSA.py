#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 13:20:49 2020

@author: ym
"""

import numpy as np
import torch.nn as nn
import models.Blockmodule_MHSA as bm
import models.SEmodule as se
import torch.nn.functional as F
import pdb
import torch

params = {'num_channels': 6,
          'num_filters': 64,
          'kernel_h': 3,
          'kernel_w': 3,
          'kernel_c': 1,
          'stride_conv': 1,
          'pool': 2,
          'stride_pool': 2,
          # Valid options : NONE, CSE, SSE, CSSE
          'se_block': "NONE",
          'drop_out': 0.1}


class Backbone(nn.Module):

    def __init__(self):
        super(Backbone, self).__init__()

        self.encode1 = bm.EncoderBlock(params, se_block_type='NONE')
        params['num_channels'] = 64
        self.encode2 = bm.EncoderBlock(params, se_block_type='NONE')
        self.encode3 = bm.EncoderBlock(params, se_block_type='NONE')

        self.encode4 = bm.EncoderBlock(params, se_block_type='NONE')
        self.encode5 = bm.EncoderBlock(params, se_block_type='NONE')

        self.bottleneck = bm.DenseBlock_MHSA(params, se_block_type='NONE')

        params['num_channels'] = 128
        self.decode5 = bm.DecoderBlock(params, se_block_type='NONE')
        self.decode4 = bm.DecoderBlock(params, se_block_type='NONE')

        self.decode3 = bm.DecoderBlock(params, se_block_type='NONE')
        self.decode2 = bm.DecoderBlock(params, se_block_type='NONE')
        self.decode1 = bm.DecoderBlock(params, se_block_type='NONE')

        self.decode0 = bm.DecoderBlock(params, se_block_type='NONE')
        self.conv0 = nn.Conv2d(64, 1, 1)

        self.conv1 = nn.Conv2d(64, 32, 1)  # for regression outputs
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 1, 1)

    def forward(self, input):
        b1 = torch.unsqueeze(input[:, 0, :, :], 1)

        e1, out1, ind1 = self.encode1.forward(input)
        e2, out2, ind2 = self.encode2.forward(e1)
        e3, out3, ind3 = self.encode3.forward(e2)

        e4, out4, ind4 = self.encode4.forward(e3)
        e5, out5, ind5 = self.encode5.forward(e4)
        bn = self.bottleneck.forward(e5)

        d5 = self.decode5.forward(bn, out5, ind5)
        d4 = self.decode4.forward(d5, out4, ind4)

        d3 = self.decode3.forward(d4, out3, ind3)
        d2 = self.decode2.forward(d3, out2, ind2)
        d1 = self.decode1.forward(d2, out1, ind1)

        d0 = self.decode0.forward(d2, out1, ind1)
        pred_S0 = torch.mul((1 + self.relu(self.conv0(d0))), b1)  # output bx1x96x96
        # pred_S0 = self.relu(self.conv0(d0)) #output bx1x96x96

        out = self.relu(self.bn(self.conv1(d1)))
        pred_ADC = torch.sigmoid(self.conv2(out))  # * 32 #output bx1x96x96

        return pred_ADC, pred_S0
