#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Custom operators."""

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorchvideo.layers.swish import Swish

from slowfast.models.utils import get_gkern


class _SeparableConv(nn.Module):
    def __init__(self, *args, **kwargs):
        super(_SeparableConv, self).__init__()

        self.dwconv = None
        self.dwconv_normalization = None
        self.dwconv_activation = None

        self.pwconv = None
        self.pwconv_normalization = None
        self.pwconv_activation = None

    def forward(self, x):
        assert self.dwconv is not None and self.pwconv is not None, (
            "Depthwise Convolution and/or Pointwise Convolution is/are not implemented"
            " yet."
        )

        x = self.dwconv(x)

        if self.dwconv_normalization is not None:
            x = self.dwconv_normalization(x)

        if self.dwconv_activation is not None:
            x = self.dwconv_activation(x)

        x = self.pwconv(x)

        if self.pwconv_normalization is not None:
            x = self.pwconv_normalization(x)

        if self.dwconv_activation is not None:
            x = self.pwconv_activation(x)

        return x


class SeparableConv3d(_SeparableConv):
    r"""Applies a 3D depthwise separable convolution over an input signal composed of several input
    planes as described in the paper
    `Xception: Deep Learning with Depthwise Separable Convolutions <https://arxiv.org/abs/1610.02357>`__ .

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all four sides of
            the input. Default: 0
        padding_mode (string, optional): ``'zeros'``, ``'reflect'``,
            ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        depth_multiplier (int, optional): The number of depthwise convolution output channels
            for each input channel.
            The total number of depthwise convolution output
            channels will be equal to `in_channels * depth_multiplier`. Default: 1
        normalization_dw (str, optional): depthwise convolution normalization. Default: 'bn'
        normalization_pw (str): pointwise convolution normalization. Default: 'bn'
        activation_dw (Callable[..., torch.nn.Module], optional): depthwise convolution activation. Default: ``torch.nn.ReLU``
        activation_pw (Callable[..., torch.nn.Module], optional): pointwise convolution activation. Default: ``torch.nn.ReLU``
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        padding_mode: str = "zeros",
        dilation=1,
        depth_multiplier: int = 1,
        normalization_dw: str = "bn",
        normalization_pw: str = "bn",
        activation_dw=nn.ReLU,
        activation_pw=nn.ReLU,
        bias=True,
        device=None,
        dtype=None,
    ) -> None:
        super(SeparableConv3d, self).__init__()

        expansion_channels = max(in_channels * int(depth_multiplier), in_channels)

        if in_channels * depth_multiplier != expansion_channels:
            raise ValueError("depth_multiplier must be integer>=1")

        self.dwconv = nn.Conv3d(
            in_channels,
            expansion_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

        self.dwconv_normalization = (
            nn.BatchNorm3d(expansion_channels)
            if normalization_dw == "bn"
            else nn.InstanceNorm3d(expansion_channels)
            if normalization_dw == "in"
            else None
        )

        if self.dwconv_normalization is None:
            warnings.warn(
                "normalization_dw is invalid. Default to ``None``. "
                "Please consider using valid normalization: "
                "'bn' for ``nn.BatchNorm3d`` or 'in' for ``nn.InstanceNorm3d``."
            )

        self.dwconv_activation = activation_dw()

        self.pwconv = nn.Conv3d(
            expansion_channels,
            out_channels,
            1,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

        self.pwconv_normalization = (
            nn.BatchNorm3d(out_channels)
            if normalization_pw == "bn"
            else nn.InstanceNorm3d(out_channels)
            if normalization_pw == "in"
            else None
        )

        if self.pwconv_normalization is None:
            warnings.warn(
                "normalization_pw is invalid. Default to ``None``. "
                "Please consider using valid normalization: "
                "'bn' for ``nn.BatchNorm3d`` or 'in' for ``nn.InstanceNorm3d``."
            )

        self.pwconv_activation = activation_pw()


class SE(nn.Module):
    """Squeeze-and-Excitation (SE) block w/ Swish: AvgPool, FC, Swish, FC, Sigmoid."""

    def _round_width(self, width, multiplier, min_width=8, divisor=8):
        """
        Round width of filters based on width multiplier
        Args:
            width (int): the channel dimensions of the input.
            multiplier (float): the multiplication factor.
            min_width (int): the minimum width after multiplication.
            divisor (int): the new width should be dividable by divisor.
        """
        if not multiplier:
            return width

        width *= multiplier
        min_width = min_width or divisor
        width_out = max(min_width, int(width + divisor / 2) // divisor * divisor)
        if width_out < 0.9 * width:
            width_out += divisor
        return int(width_out)

    def __init__(self, dim_in, ratio, relu_act=True):
        """
        Args:
            dim_in (int): the channel dimensions of the input.
            ratio (float): the channel reduction ratio for squeeze.
            relu_act (bool): whether to use ReLU activation instead
                of Swish (default).
            divisor (int): the new width should be dividable by divisor.
        """
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        dim_fc = self._round_width(dim_in, ratio)
        self.fc1 = nn.Conv3d(dim_in, dim_fc, 1, bias=True)
        self.fc1_act = nn.ReLU() if relu_act else Swish()
        self.fc2 = nn.Conv3d(dim_fc, dim_in, 1, bias=True)

        self.fc2_sig = nn.Sigmoid()

    def forward(self, x):
        x_in = x
        for module in self.children():
            x = module(x)
        return x_in * x


class HOGLayerC(nn.Module):
    def __init__(self, nbins=9, pool=7, gaussian_window=0):
        super(HOGLayerC, self).__init__()
        self.nbins = nbins
        self.pool = pool
        self.pi = math.pi
        weight_x = torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        weight_x = weight_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        weight_y = weight_x.transpose(2, 3)
        self.register_buffer("weight_x", weight_x)
        self.register_buffer("weight_y", weight_y)

        self.gaussian_window = gaussian_window
        if gaussian_window:
            gkern = get_gkern(gaussian_window, gaussian_window // 2)
            self.register_buffer("gkern", gkern)

    @torch.no_grad()
    def forward(self, x):
        # input is RGB image with shape [B 3 H W]
        x = F.pad(x, pad=(1, 1, 1, 1), mode="reflect")
        gx_rgb = F.conv2d(x, self.weight_x, bias=None, stride=1, padding=0, groups=3)
        gy_rgb = F.conv2d(x, self.weight_y, bias=None, stride=1, padding=0, groups=3)
        norm_rgb = torch.stack([gx_rgb, gy_rgb], dim=-1).norm(dim=-1)
        phase = torch.atan2(gx_rgb, gy_rgb)
        phase = phase / self.pi * self.nbins  # [-9, 9]

        b, c, h, w = norm_rgb.shape
        out = torch.zeros((b, c, self.nbins, h, w), dtype=torch.float, device=x.device)
        phase = phase.view(b, c, 1, h, w)
        norm_rgb = norm_rgb.view(b, c, 1, h, w)
        if self.gaussian_window:
            if h != self.gaussian_window:
                assert h % self.gaussian_window == 0, "h {} gw {}".format(
                    h, self.gaussian_window
                )
                repeat_rate = h // self.gaussian_window
                temp_gkern = self.gkern.repeat([repeat_rate, repeat_rate])
            else:
                temp_gkern = self.gkern
            norm_rgb *= temp_gkern

        out.scatter_add_(2, phase.floor().long() % self.nbins, norm_rgb)

        out = out.unfold(3, self.pool, self.pool)
        out = out.unfold(4, self.pool, self.pool)
        out = out.sum(dim=[-1, -2])

        out = torch.nn.functional.normalize(out, p=2, dim=2)

        return out  # B 3 nbins H W
