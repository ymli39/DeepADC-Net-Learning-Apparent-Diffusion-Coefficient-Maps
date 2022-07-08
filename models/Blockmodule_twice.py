#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 18:08:38 2019

@author: ym
"""

import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
import models.SEmodule as se
from torch.autograd import Function
import pdb


class MHSA(nn.Module):
    '''
    References: https://github.com/leaderj1001/BottleneckTransformers/blob/main/model.py
    '''

    def __init__(self, n_dims, width=14, height=14, heads=4):
        super(MHSA, self).__init__()
        self.heads = heads

        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)

        self.rel_h = nn.Parameter(torch.randn([1, heads, n_dims // heads, 1, height]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, heads, n_dims // heads, width, 1]), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):

        n_batch, C, width, height = x.size()
        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1)
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1)
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)

        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)

        content_position = (self.rel_h + self.rel_w).view(1, self.heads, C // self.heads, -1).permute(0, 1, 3, 2)
        content_position = torch.matmul(content_position, q)

        energy = content_content + content_position
        attention = self.softmax(energy)

        out = torch.matmul(v, attention.permute(0, 1, 3, 2))
        out = out.view(n_batch, C, width, height)

        return out


class DenseBlock(nn.Module):
    """Block with dense connections
    :param params
    :type params: dict
    :param se_block_type: Squeeze and Excite block type to be included, defaults to None
    :type se_block_type: str, valid options are {'NONE', 'CSE', 'SSE', 'CSSE'}, optional
    :return: forward passed tensor
    :rtype: torch.tonsor [FloatTensor]
    """

    def __init__(self, params, se_block_type=None):
        super(DenseBlock, self).__init__()

        if se_block_type == se.SELayer.CSE.value:
            self.SELayer = se.ChannelSELayer(params['num_filters'])

        elif se_block_type == se.SELayer.SSE.value:
            self.SELayer = se.SpatialSELayer(params['num_filters'])

        elif se_block_type == se.SELayer.CSSE.value:
            self.SELayer = se.ChannelSpatialSELayer(params['num_filters'])
        else:
            self.SELayer = None

        padding_h = int((params['kernel_h'] - 1) / 2)
        padding_w = int((params['kernel_w'] - 1) / 2)

        conv1_out_size = int(params['num_channels'] + params['num_filters'])
        conv2_out_size = int(
            params['num_channels'] + params['num_filters'] + params['num_filters'])
        conv3_out_size = int(
            params['num_channels'] + params['num_filters'] + params['num_filters'] + params['num_filters'])
        conv4_out_size = int(
            params['num_channels'] + params['num_filters'] + params['num_filters'] + params['num_filters']+ params['num_filters'])

        self.conv1 = nn.Conv2d(in_channels=params['num_channels'], out_channels=params['num_filters'],
                               kernel_size=(
                                   params['kernel_h'], params['kernel_w']),
                               padding=(padding_h, padding_w),
                               stride=params['stride_conv'])
        self.conv2 = nn.Conv2d(in_channels=conv1_out_size, out_channels=params['num_filters'],
                               kernel_size=(
                                   params['kernel_h'], params['kernel_w']),
                               padding=(padding_h, padding_w),
                               stride=params['stride_conv'])
        self.conv3 = nn.Conv2d(in_channels=conv2_out_size, out_channels=params['num_filters'],
                               kernel_size=(1, 1),
                               padding=(0, 0),
                               stride=params['stride_conv'])
        self.conv4 = nn.Conv2d(in_channels=conv3_out_size, out_channels=params['num_filters'],
                               kernel_size=(1, 1),
                               padding=(0, 0),
                               stride=params['stride_conv'])
        self.conv5 = nn.Conv2d(in_channels=conv4_out_size, out_channels=params['num_filters'],
                               kernel_size=(1, 1),
                               padding=(0, 0),
                               stride=params['stride_conv'])
        self.batchnorm1 = nn.BatchNorm2d(num_features=params['num_channels'])
        self.batchnorm2 = nn.BatchNorm2d(num_features=conv1_out_size)
        self.batchnorm3 = nn.BatchNorm2d(num_features=conv2_out_size)
        self.batchnorm4 = nn.BatchNorm2d(num_features=conv3_out_size)
        self.batchnorm5 = nn.BatchNorm2d(num_features=conv4_out_size)
        self.relu = nn.ReLU()
        if params['drop_out'] > 0:
            self.drop_out_needed = True
            self.drop_out = nn.Dropout2d(params['drop_out'])
        else:
            self.drop_out_needed = False

    def forward(self, input):
        """Forward pass
        :param input: Input tensor, shape = (N x C x H x W)
        :type input: torch.tensor [FloatTensor]
        :return: Forward passed tensor
        :rtype: torch.tensor [FloatTensor]
        """

        o1 = self.batchnorm1(input)
        o2 = self.relu(o1)
        o3 = self.conv1(o2)
        o4 = torch.cat((input, o3), dim=1)
        o5 = self.batchnorm2(o4)
        o6 = self.relu(o5)
        o7 = self.conv2(o6)
        o8 = torch.cat((input, o3, o7), dim=1)
        o9 = self.batchnorm3(o8)
        o10 = self.relu(o9)
        o11 = self.conv3(o10)
        o12 = torch.cat((input, o3, o7, o11), dim=1)
        o13 = self.batchnorm4(o12)
        o14 = self.relu(o13)
        o15 = self.conv4(o14)
        o16 = torch.cat((input, o3, o7, o11, o15), dim=1)
        o17 = self.batchnorm5(o16)
        o18 = self.relu(o17)
        out = self.conv5(o18)
        return out

class DenseBlock_MHSA(nn.Module):
    """Block with dense connections
    :param params
    :type params: dict
    :param se_block_type: Squeeze and Excite block type to be included, defaults to None
    :type se_block_type: str, valid options are {'NONE', 'CSE', 'SSE', 'CSSE'}, optional
    :return: forward passed tensor
    :rtype: torch.tonsor [FloatTensor]
    """

    def __init__(self, params, se_block_type=None):
        super(DenseBlock_MHSA, self).__init__()

        if se_block_type == se.SELayer.CSE.value:
            self.SELayer = se.ChannelSELayer(params['num_filters'])

        elif se_block_type == se.SELayer.SSE.value:
            self.SELayer = se.SpatialSELayer(params['num_filters'])

        elif se_block_type == se.SELayer.CSSE.value:
            self.SELayer = se.ChannelSpatialSELayer(params['num_filters'])
        else:
            self.SELayer = None

        padding_h = int((params['kernel_h'] - 1) / 2)
        padding_w = int((params['kernel_w'] - 1) / 2)

        conv1_out_size = int(params['num_channels'] + params['num_filters'])
        conv2_out_size = int(
            params['num_channels'] + params['num_filters'] + params['num_filters'])

        self.conv1 = nn.Conv2d(in_channels=params['num_channels'], out_channels=params['num_filters'],
                               kernel_size=(1, 1),
                               stride=params['stride_conv'])
        self.conv2 = nn.Conv2d(in_channels=conv1_out_size, out_channels=params['num_filters'],
                               kernel_size=(1, 1),
                               stride=params['stride_conv'])
        self.conv3 = nn.Conv2d(in_channels=conv2_out_size, out_channels=params['num_filters'],
                               kernel_size=(1, 1),
                               padding=(0, 0),
                               stride=params['stride_conv'])

        self.conv1_mhsa = MHSA(params['num_filters'], width=3, height=3, heads=4)
        self.conv2_mhsa = MHSA(params['num_filters'], width=3, height=3, heads=4)

        self.batchnorm1 = nn.BatchNorm2d(num_features=params['num_channels'])
        self.batchnorm2 = nn.BatchNorm2d(num_features=conv1_out_size)
        self.batchnorm3 = nn.BatchNorm2d(num_features=conv2_out_size)
        self.relu = nn.ReLU()
        if params['drop_out'] > 0:
            self.drop_out_needed = True
            self.drop_out = nn.Dropout2d(params['drop_out'])
        else:
            self.drop_out_needed = False

    def forward(self, input):
        """Forward pass
        :param input: Input tensor, shape = (N x C x H x W)
        :type input: torch.tensor [FloatTensor]
        :return: Forward passed tensor
        :rtype: torch.tensor [FloatTensor]
        """

        o1 = self.batchnorm1(input)
        o2 = self.relu(o1)
        o3 = self.conv1(o2)
        o3_att = self.conv1_mhsa(o3)
        # o3_att = self.conv1_mhsa(o3) + o3
        o4 = torch.cat((input, o3_att), dim=1)
        o5 = self.batchnorm2(o4)
        o6 = self.relu(o5)
        o7 = self.conv2(o6)
        o7_att = self.conv1_mhsa(o7)
        # o7_att = self.conv1_mhsa(o7) + o7
        o8 = torch.cat((input, o3_att, o7_att), dim=1)
        o9 = self.batchnorm3(o8)
        o10 = self.relu(o9)
        out = self.conv3(o10)
        return out


class EncoderBlock(DenseBlock):
    """Dense encoder block with maxpool and an optional SE block
    :param params
    :type params: dict
    :param se_block_type: Squeeze and Excite block type to be included, defaults to None
    :type se_block_type: str, valid options are {'NONE', 'CSE', 'SSE', 'CSSE'}, optional
    :return: output tensor with maxpool, output tensor without maxpool, indices for unpooling
    :rtype: torch.tensor [FloatTensor], torch.tensor [FloatTensor], torch.tensor [LongTensor]
    """

    def __init__(self, params, se_block_type=None):
        super(EncoderBlock, self).__init__(params, se_block_type=se_block_type)
        self.maxpool = nn.MaxPool2d(
            kernel_size=params['pool'], stride=params['stride_pool'], return_indices=True)

    def forward(self, input):
        """Forward pass

        :param input: Input tensor, shape = (N x C x H x W)
        :type input: torch.tensor [FloatTensor]
        :return: output tensor with maxpool, output tensor without maxpool, indices for unpooling
        :rtype: torch.tensor [FloatTensor], torch.tensor [FloatTensor], torch.tensor [LongTensor]
        """

        out_block = super(EncoderBlock, self).forward(input)
        if self.SELayer:
            out_block, squeeze_tensor = self.SELayer(out_block)

        if self.drop_out_needed:
            out_block = self.drop_out(out_block)

        _, dim1, _, _ = input.shape
        _, dim2, _, _ = out_block.shape

        if dim1 == dim2:
            out_block = out_block + input

        out_encoder, indices = self.maxpool(out_block)
        return out_encoder, out_block, indices


class DecoderBlock_nocat(DenseBlock):
    """Dense decoder block with maxunpool and an optional skip connections and SE block
    :param params
    :type params: dict
    :param se_block_type: Squeeze and Excite block type to be included, defaults to None
    :type se_block_type: str, valid options are {'NONE', 'CSE', 'SSE', 'CSSE'}, optional
    :return: forward passed tensor
    :rtype: torch.tensor [FloatTensor]
    """

    def __init__(self, params, in_dim, out_dim, se_block_type=None):
        super(DecoderBlock_nocat, self).__init__(params, se_block_type=se_block_type)
        self.unpool = nn.ConvTranspose2d(in_dim, out_dim, kernel_size=2, stride=2)

    def forward(self, input):
        """Forward pass
        :param input: Input tensor, shape = (N x C x H x W)
        :type input: torch.tensor [FloatTensor]
        :param out_block: Tensor for skip connection, shape = (N x C x H x W), defaults to None
        :type out_block: torch.tensor [FloatTensor], optional
        :param indices: Indices used for unpooling operation, defaults to None
        :type indices: torch.tensor, optional
        :return: Forward passed tensor
        :rtype: torch.tensor [FloatTensor]
        """

        unpool = self.unpool(input)

        out_block = super(DecoderBlock_nocat, self).forward(unpool)

        _, dim1, _, _ = unpool.shape
        _, dim2, _, _ = out_block.shape

        if dim1 == dim2:
            out_block = out_block + unpool

        if self.SELayer:
            out_block, squeeze_tensor = self.SELayer(out_block)

        if self.drop_out_needed:
            out_block = self.drop_out(out_block)

        return out_block, squeeze_tensor


class DecoderBlock(DenseBlock):
    """Dense decoder block with maxunpool and an optional skip connections and SE block
    :param params
    :type params: dict
    :param se_block_type: Squeeze and Excite block type to be included, defaults to None
    :type se_block_type: str, valid options are {'NONE', 'CSE', 'SSE', 'CSSE'}, optional
    :return: forward passed tensor
    :rtype: torch.tensor [FloatTensor]
    """

    def __init__(self, params, se_block_type=None):
        super(DecoderBlock, self).__init__(params, se_block_type=se_block_type)
        self.unpool = nn.MaxUnpool2d(
            kernel_size=params['pool'], stride=params['stride_pool'])

    def forward(self, input, out_block=None, indices=None):
        """Forward pass
        :param input: Input tensor, shape = (N x C x H x W)
        :type input: torch.tensor [FloatTensor]
        :param out_block: Tensor for skip connection, shape = (N x C x H x W), defaults to None
        :type out_block: torch.tensor [FloatTensor], optional
        :param indices: Indices used for unpooling operation, defaults to None
        :type indices: torch.tensor, optional
        :return: Forward passed tensor
        :rtype: torch.tensor [FloatTensor]
        """

        unpool = self.unpool(input, indices)
        concat = torch.cat((out_block, unpool), dim=1)
        out_block = super(DecoderBlock, self).forward(concat)

        _, dim1, _, _ = unpool.shape
        _, dim2, _, _ = out_block.shape

        if dim1 == dim2:
            out_block = out_block + unpool

        if self.SELayer:
            out_block, squeeze_tensor = self.SELayer(out_block)

        if self.drop_out_needed:
            out_block = self.drop_out(out_block)
        return out_block


class Decoder(nn.Module):
    def __init__(self, dim_in1, dim_in2, dim_out, kernel_size=5, padding=2):
        super().__init__()
        self.deconv = nn.Sequential(nn.ConvTranspose2d(dim_in1, dim_out, kernel_size=2, stride=2))
        self.conv1 = nn.Sequential(nn.Conv2d(dim_out + dim_in2, dim_out, kernel_size=kernel_size, padding=padding))
        self.NormRelu = nn.Sequential(nn.BatchNorm2d(dim_out),
                                      nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, padding=padding))

    def forward(self, input, out_block, indices=None):
        up = self.deconv(input)
        concat = torch.cat((up, out_block), dim=1)
        out = self.conv1(concat)
        out = self.NormRelu(out)
        out = self.conv2(out)
        out = self.NormRelu(out)
        return out


class Encoder(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=5, padding=2, stride=2, se_block_type=None):
        super().__init__()

        if se_block_type == se.SELayer.CSE.value:
            self.SELayer = se.ChannelSELayer(dim_out)

        elif se_block_type == se.SELayer.SSE.value:
            self.SELayer = se.SpatialSELayer(dim_out)

        elif se_block_type == se.SELayer.CSSE.value:
            self.SELayer = se.ChannelSpatialSELayer(dim_out)
        else:
            self.SELayer = None

        self.code = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True))

        self.maxpool = nn.MaxPool2d(
            kernel_size=stride, stride=stride, return_indices=True)

    def forward(self, input):
        out_block = self.code(input)
        if self.SELayer:
            out_block = self.SELayer(out_block)

        out_encoder, indices = self.maxpool(out_block)

        return out_encoder, indices


class Mean(nn.Module):
    def __init__(self, dim, keep_dim=False):
        super(Mean, self).__init__()
        self.dim = dim
        self.keep_dim = keep_dim

    def forward(self, input):
        return input.mean(self.dim, self.keep_dim)


class ClassifierBlock(nn.Module):
    """
    Last layer
    :type params: dict
    :return: forward passed tensor
    :rtype: torch.tensor [FloatTensor]
    """

    def __init__(self, params):
        super(ClassifierBlock, self).__init__()
        self.conv = nn.Conv2d(
            params['num_channels'], params['num_class'], params['kernel_c'], params['stride_conv'])

    def forward(self, input, weights=None):
        """Forward pass
        :param input: Input tensor, shape = (N x C x H x W)
        :type input: torch.tensor [FloatTensor]
        :param weights: Weights for classifier regression, defaults to None
        :type weights: torch.tensor (N), optional
        :return: logits
        :rtype: torch.tensor
        """
        batch_size, channel, a, b = input.size()
        if weights is not None:
            weights, _ = torch.max(weights, dim=0)
            weights = weights.view(1, channel, 1, 1)
            out_conv = F.conv2d(input, weights)
        else:
            out_conv = self.conv(input)
        return out_conv


