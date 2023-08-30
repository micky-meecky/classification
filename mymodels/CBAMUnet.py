#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: 杨亚峰<yang_armageddon@163.com>
@version: 1.0.0
@license:  Apache Licence
@editor: Pycharm yyf
@file: CBAMUnet.py
@datatime: 8/17/2023 3:25 PM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mymodels.unet.unet_utils import getModelSize
from mymodels.generatorGAN import PixelwiseViT as PixViT
from mymodels.unet.unet_utils import ChannelAttention, SpatialAttention, SideSEConv2d, SideConv2d, \
    MultiDilatedConv


class DoubleConvWithCBAM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DoubleConvWithCBAM, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.double_conv(x)
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DownwithCBAM(nn.Module):
    """
    Downscaling with CBAM and maxpool or Conv2d to downsampling then double conv
    extract feature map for next layer, CBAM is used to another route for skip connection
    """

    def __init__(self, in_channels, out_channels, method='maxpool'):
        super().__init__()
        self.method = method
        if method == 'maxpool':
            # 方法一：
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                nn.Conv2d(in_channels, in_channels, kernel_size=1),
                nn.BatchNorm2d(in_channels),
                nn.SiLU(inplace=True),
                DoubleConv(in_channels, out_channels),
                nn.Dropout(0.1)
            )
        elif method == 'convpool':
            # 方法二：
            self.convpool_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True),  # SiLu 是一个新的激活函数，其实就是Sigmoid和ReLU的结合
                DoubleConv(out_channels, out_channels),
                nn.Dropout(0.1)
            )
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        if self.method == 'maxpool':
            out = self.maxpool_conv(x)
            skip_out = self.ca(out) * out
            skip_out = self.sa(skip_out) * skip_out
            return out, skip_out
        else:
            out = self.convpool_conv(x)
            skip_out = self.ca(out) * out
            skip_out = self.sa(skip_out) * skip_out
            return out, skip_out


class DilatedDownwithCBAM(nn.Module):
    """
    Downscaling with CBAM and maxpool or Conv2d to downsampling then double conv
    extract feature map for next layer, CBAM is used to another route for skip connection
    """

    def __init__(self, in_channels, out_channels, method='maxpool'):
        super().__init__()
        self.method = method
        if method == 'maxpool':
            # 方法一：
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                nn.Conv2d(in_channels, in_channels, kernel_size=1),
                nn.BatchNorm2d(in_channels),
                nn.SiLU(inplace=True),
                DoubleConv(in_channels, out_channels),
                MultiDilatedConv(out_channels, out_channels),
                nn.Dropout(0.1)
            )
        elif method == 'convpool':
            # 方法二：
            self.convpool_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True),  # SiLu 是一个新的激活函数，其实就是Sigmoid和ReLU的结合
                DoubleConv(out_channels, out_channels),
                MultiDilatedConv(out_channels, out_channels),
                nn.Dropout(0.1)
            )
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        if self.method == 'maxpool':
            out = self.maxpool_conv(x)
            skip_out = self.ca(out) * out
            skip_out = self.sa(skip_out) * skip_out
            return out, skip_out
        else:
            out = self.convpool_conv(x)
            skip_out = self.ca(out) * out
            skip_out = self.sa(skip_out) * skip_out
            return out, skip_out


class SideDownwithCBAM(nn.Module):
    """
    Downscaling with CBAM and maxpool or Conv2d to downsampling then double conv
    extract feature map for next layer, CBAM is used to another route for skip connection
    layernum：当前层数，用于判断是否为第一层， 如果是第一层，侧边模块的输入通道数为3，否则为in_channels
    """

    def __init__(self, in_channels, out_channels, sidemode='NOSE',
                 method='maxpool', layernum=222,
                 Islastlayer=False, IsPixViT=False):
        super().__init__()
        self.method = method
        if layernum == 1:
            side_in_channels = 3
        else:
            side_in_channels = in_channels
        if sidemode == 'SE':
            self.sideconv = SideSEConv2d(side_in_channels, out_channels)
        else:
            self.sideconv = SideConv2d(side_in_channels, out_channels)
        self.Islastlayer = Islastlayer
        self.IsPixViT = IsPixViT
        if method == 'maxpool':
            # 方法一：
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                nn.Conv2d(in_channels, in_channels, kernel_size=1),
                nn.BatchNorm2d(in_channels),
                nn.SiLU(inplace=True),
                DoubleConv(in_channels, out_channels),
                nn.Dropout(0.1)
            )
        elif method == 'convpool':
            # 方法二：
            self.convpool_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True),  # SiLu 是一个新的激活函数，其实就是Sigmoid和ReLU的结合
                DoubleConv(out_channels, out_channels),
                nn.Dropout(0.1)
            )
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x, side):
        if self.method == 'maxpool':
            out = self.maxpool_conv(x)
            side = self.sideconv(out, side)
            if self.Islastlayer is True:
                if self.IsPixViT is False:
                    skip_out = self.ca(side) * side
                    skip_out = self.sa(skip_out) * skip_out
                else:
                    skip_out = side  # bottleneck为pixvit的最后一层不需要cbam, 直接返回side即可
            else:
                skip_out = self.ca(out) * out
                skip_out = self.sa(skip_out) * skip_out
            return out, skip_out, side
        else:
            out = self.convpool_conv(x)
            side = self.sideconv(out, side)
            if self.Islastlayer is True:
                if self.IsPixViT is False:
                    skip_out = self.ca(side) * side
                    skip_out = self.sa(skip_out) * skip_out
                else:
                    skip_out = side  # bottleneck为pixvit的最后一层不需要cbam, 直接返回side即可
            else:
                skip_out = self.ca(out) * out
                skip_out = self.sa(skip_out) * skip_out
            return out, skip_out, side


class SideDiDownwithCBAM(nn.Module):
    """
    Downscaling with CBAM and maxpool or Conv2d to downsampling then double conv
    extract feature map for next layer, CBAM is used to another route for skip connection
    layernum：当前层数，用于判断是否为第一层， 如果是第一层，侧边模块的输入通道数为3，否则为in_channels
    """

    def __init__(self, in_channels, out_channels, sidemode='NOSE',
                 method='maxpool', layernum=222,
                 Islastlayer=False, IsPixViT=False):
        super().__init__()
        self.method = method
        if layernum == 1:
            side_in_channels = 3
        else:
            side_in_channels = in_channels
        if sidemode == 'SE':
            self.sideconv = SideSEConv2d(side_in_channels, out_channels)
        else:
            self.sideconv = SideConv2d(side_in_channels, out_channels)
        self.Islastlayer = Islastlayer
        self.IsPixViT = IsPixViT
        if method == 'maxpool':
            # 方法一：
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                nn.Conv2d(in_channels, in_channels, kernel_size=1),
                nn.BatchNorm2d(in_channels),
                nn.SiLU(inplace=True),
                DoubleConv(in_channels, out_channels),
                MultiDilatedConv(out_channels, out_channels),
                nn.Dropout(0.1)
            )
        elif method == 'convpool':
            # 方法二：
            self.convpool_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True),  # SiLu 是一个新的激活函数，其实就是Sigmoid和ReLU的结合
                DoubleConv(out_channels, out_channels),
                MultiDilatedConv(out_channels, out_channels),
                nn.Dropout(0.1)
            )
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x, side):
        if self.method == 'maxpool':
            out = self.maxpool_conv(x)
            side = self.sideconv(out, side)
            if self.Islastlayer is True:
                if self.IsPixViT is False:
                    skip_out = self.ca(side) * side
                    skip_out = self.sa(skip_out) * skip_out
                else:
                    skip_out = side  # bottleneck为pixvit的最后一层不需要cbam, 直接返回side即可
            else:
                skip_out = self.ca(out) * out
                skip_out = self.sa(skip_out) * skip_out
            return out, skip_out, side
        else:
            out = self.convpool_conv(x)
            side = self.sideconv(out, side)
            if self.Islastlayer is True:
                if self.IsPixViT is False:
                    skip_out = self.ca(side) * side
                    skip_out = self.sa(skip_out) * skip_out
                else:
                    skip_out = side  # bottleneck为pixvit的最后一层不需要cbam, 直接返回side即可
            else:
                skip_out = self.ca(out) * out
                skip_out = self.sa(skip_out) * skip_out
            return out, skip_out, side


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, method='maxpool'):
        super().__init__()
        self.method = method
        if method == 'maxpool':
            # 方法一：
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                nn.Conv2d(in_channels, in_channels, kernel_size=1),
                nn.BatchNorm2d(in_channels),
                nn.SiLU(inplace=True),
                DoubleConv(in_channels, out_channels),
                nn.Dropout(0.1)
            )
        elif method == 'convpool':
            # 方法二：
            self.convpool_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True),
                DoubleConv(out_channels, out_channels),
                nn.Dropout(0.1)
            )

    def forward(self, x):
        if self.method == 'maxpool':
            return self.maxpool_conv(x)
        else:
            return self.convpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class AGUp(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.out_channels = out_channels

        # Attention gate
        self.att = AttentionGate(F_g=in_channels // 2, F_l=out_channels, F_int=in_channels // 4)

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels // 2 + self.out_channels, self.out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # Attention gate
        x2 = self.att(g=x1, x=x2)

        x = torch.cat([x2, x1], dim=1)
        # self.conv = DoubleConv(inch + outch, self.out_channels)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()

        # W_g用于对g进行特征转换，包含一个卷积层和一个批量归一化层
        # 卷积层的作用是对输入特征进行线性变换，批量归一化层的作用是对特征进行归一化，使得网络更容易训练
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        # W_x用于对x进行特征转换，包含一个卷积层和一个批量归一化层
        # 卷积层和批量归一化层的作用同上
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        # psi用于计算注意力系数，包含一个卷积层，一个批量归一化层和一个Sigmoid激活函数
        # 卷积层和批量归一化层的作用同上，Sigmoid激活函数的作用是将特征转换到(0, 1)的范围内，使得它们可以作为注意力系数
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        # ReLU激活函数用于增加非线性
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # 对g进行特征转换
        g1 = self.W_g(g)
        # 对x进行特征转换
        x1 = self.W_x(x)
        # 将转换后的g和x相加，并通过ReLU激活函数进行非线性变换
        psi = self.relu(g1 + x1)
        # 计算注意力系数
        psi = self.psi(psi)

        # 将注意力系数应用到x上，得到最终的输出
        return x * psi


class AgCBAMUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(AgCBAMUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear  # bilinear表示是否使用双线性插值

        self.inc = (DoubleConv(n_channels, 64))
        self.inca = ChannelAttention(64)
        self.insa = SpatialAttention()
        self.down1 = (DownwithCBAM(64, 128, method='maxpool'))
        self.down2 = (DownwithCBAM(128, 256, method='maxpool'))
        self.down3 = (DownwithCBAM(256, 512, method='maxpool'))
        factor = 2 if bilinear else 1
        self.down4 = (DownwithCBAM(512, 1024 // factor, method='maxpool'))

        # self.upsample = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)
        self.up1 = (AGUp(1024, 512 // factor, bilinear))
        self.up2 = (AGUp(512, 256 // factor, bilinear))
        self.up3 = (AGUp(256, 128 // factor, bilinear))
        self.up4 = (AGUp(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))
        self.activation = nn.Sigmoid()

        # classification head
        self.linear = nn.Linear(1024, 1)

    def forward(self, x):
        # encoder
        x1 = self.inc(x)
        x1_skip = self.inca(x1) * x1
        x1_skip = self.insa(x1_skip) * x1_skip
        x2, x2_skip = self.down1(x1)
        x3, x3_skip = self.down2(x2)
        x4, x4_skip = self.down3(x3)
        x5, x5_skip = self.down4(x4)

        # decoder
        # decoder with attention gates
        x = self.up1(x5_skip, x4_skip)

        x = self.up2(x, x3_skip)

        x = self.up3(x, x2_skip)

        x = self.up4(x, x1_skip)

        # segmentation head
        logits = self.outc(x)
        # logits = self.activation(logits)

        # classification head
        clsx = F.adaptive_avg_pool2d(x5_skip, (1, 1))  # 维度变化为[batch_size, 1024, 1, 1]
        clsx = clsx.view(-1, 1024)  # [batch_size, 1024]
        label = self.linear(clsx)
        return label, logits


class CBAMUNet(nn.Module):
    def __init__(self, n_channels, n_classes, Method='maxpool', bilinear=False):
        super(CBAMUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear  # bilinear表示是否使用双线性插值

        self.inc = (DoubleConv(n_channels, 64))
        self.inca = ChannelAttention(64)
        self.insa = SpatialAttention()
        self.down1 = (DownwithCBAM(64, 128, method=Method))
        self.down2 = (DownwithCBAM(128, 256, method=Method))
        self.down3 = (DownwithCBAM(256, 512, method=Method))
        factor = 2 if bilinear else 1
        self.down4 = (DownwithCBAM(512, 1024 // factor, method=Method))

        # self.upsample = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))
        self.activation = nn.Sigmoid()

        # classification head
        self.linear = nn.Linear(1024, 1)

    def forward(self, x):
        # encoder
        x1 = self.inc(x)
        x1_skip = self.inca(x1) * x1
        x1_skip = self.insa(x1_skip) * x1_skip
        x2, x2_skip = self.down1(x1)
        x3, x3_skip = self.down2(x2)
        x4, x4_skip = self.down3(x3)
        x5, x5_skip = self.down4(x4)

        # decoder
        # decoder with attention gates
        x = self.up1(x5_skip, x4_skip)

        x = self.up2(x, x3_skip)

        x = self.up3(x, x2_skip)

        x = self.up4(x, x1_skip)

        # segmentation head
        logits = self.outc(x)
        # logits = self.activation(logits)

        # classification head
        clsx = F.adaptive_avg_pool2d(x5_skip, (1, 1))  # 维度变化为[batch_size, 1024, 1, 1]
        clsx = clsx.view(-1, 1024)  # [batch_size, 1024]
        label = self.linear(clsx)
        return label, logits


class DialatedCBAMUNet(nn.Module):
    def __init__(self, n_channels, n_classes, Method='maxpool', bilinear=False):
        super(DialatedCBAMUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear  # bilinear表示是否使用双线性插值

        self.inc = (DoubleConv(n_channels, 64))
        self.inca = ChannelAttention(64)
        self.insa = SpatialAttention()
        self.down1 = (DilatedDownwithCBAM(64, 128, method=Method))
        self.down2 = (DilatedDownwithCBAM(128, 256, method=Method))
        self.down3 = (DilatedDownwithCBAM(256, 512, method=Method))
        factor = 2 if bilinear else 1
        self.down4 = (DownwithCBAM(512, 1024 // factor, method=Method))

        # self.upsample = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))
        self.activation = nn.Sigmoid()

        # classification head
        self.linear = nn.Linear(1024, 1)

    def forward(self, x):
        # encoder
        x1 = self.inc(x)
        x1_skip = self.inca(x1) * x1
        x1_skip = self.insa(x1_skip) * x1_skip
        x2, x2_skip = self.down1(x1)
        x3, x3_skip = self.down2(x2)
        x4, x4_skip = self.down3(x3)
        x5, x5_skip = self.down4(x4)

        # decoder
        # decoder with attention gates
        x = self.up1(x5_skip, x4_skip)

        x = self.up2(x, x3_skip)

        x = self.up3(x, x2_skip)

        x = self.up4(x, x1_skip)

        # segmentation head
        logits = self.outc(x)
        # logits = self.activation(logits)

        # classification head
        clsx = F.adaptive_avg_pool2d(x5_skip, (1, 1))  # 维度变化为[batch_size, 1024, 1, 1]
        clsx = clsx.view(-1, 1024)  # [batch_size, 1024]
        label = self.linear(clsx)
        return label, logits


class NewCBAMUNet(nn.Module):
    def __init__(self, n_channels, n_classes, Method='maxpool', bilinear=False):
        super(NewCBAMUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear  # bilinear表示是否使用双线性插值

        self.inc = (DoubleConv(n_channels, 64))
        self.inca = ChannelAttention(64)
        self.insa = SpatialAttention()
        self.down1 = (DownwithCBAM(64, 128, method=Method))
        self.down2 = (DownwithCBAM(128, 256, method=Method))
        self.down3 = (DownwithCBAM(256, 512, method=Method))
        factor = 2 if bilinear else 1
        self.down4 = (DownwithCBAM(512, 1024 // factor, method=Method))

        # self.upsample = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))
        self.activation = nn.Sigmoid()

        # classification head
        self.linear = nn.Linear(1024, 1)

    def forward(self, x):
        # encoder
        x1 = self.inc(x)
        x1_skip = self.inca(x1) * x1
        x1_skip = self.insa(x1_skip) * x1_skip
        _, x2_skip = self.down1(x1_skip)
        _, x3_skip = self.down2(x2_skip)
        _, x4_skip = self.down3(x3_skip)
        _, x5_skip = self.down4(x4_skip)

        # decoder
        # decoder with attention gates
        x = self.up1(x5_skip, x4_skip)

        x = self.up2(x, x3_skip)

        x = self.up3(x, x2_skip)

        x = self.up4(x, x1_skip)

        # segmentation head
        logits = self.outc(x)
        # logits = self.activation(logits)

        # classification head
        clsx = F.adaptive_avg_pool2d(x5_skip, (1, 1))  # 维度变化为[batch_size, 1024, 1, 1]
        clsx = clsx.view(-1, 1024)  # [batch_size, 1024]
        label = self.linear(clsx)
        return label, logits


class DiNewCBAMUNet(nn.Module):
    def __init__(self, n_channels, n_classes, Method='maxpool', bilinear=False):
        super(DiNewCBAMUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear  # bilinear表示是否使用双线性插值

        self.inc = (DoubleConv(n_channels, 64))
        self.inca = ChannelAttention(64)
        self.insa = SpatialAttention()
        self.down1 = (DilatedDownwithCBAM(64, 128, method=Method))
        self.down2 = (DilatedDownwithCBAM(128, 256, method=Method))
        self.down3 = (DilatedDownwithCBAM(256, 512, method=Method))
        factor = 2 if bilinear else 1
        self.down4 = (DilatedDownwithCBAM(512, 1024 // factor, method=Method))

        # self.upsample = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))
        self.activation = nn.Sigmoid()

        # classification head
        self.linear = nn.Linear(1024, 1)

    def forward(self, x):
        # encoder
        x1 = self.inc(x)
        x1_skip = self.inca(x1) * x1
        x1_skip = self.insa(x1_skip) * x1_skip
        _, x2_skip = self.down1(x1_skip)
        _, x3_skip = self.down2(x2_skip)
        _, x4_skip = self.down3(x3_skip)
        _, x5_skip = self.down4(x4_skip)

        # decoder
        # decoder with attention gates
        x = self.up1(x5_skip, x4_skip)

        x = self.up2(x, x3_skip)

        x = self.up3(x, x2_skip)

        x = self.up4(x, x1_skip)

        # segmentation head
        logits = self.outc(x)
        # logits = self.activation(logits)

        # classification head
        clsx = F.adaptive_avg_pool2d(x5_skip, (1, 1))  # 维度变化为[batch_size, 1024, 1, 1]
        clsx = clsx.view(-1, 1024)  # [batch_size, 1024]
        label = self.linear(clsx)
        return label, logits



class SideDiCBAMUNet(nn.Module):
    def __init__(self, n_channels, n_classes, Method='maxpool', bilinear=False):
        super(SideDiCBAMUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear  # bilinear表示是否使用双线性插值

        self.inc = (DoubleConv(n_channels, 64))
        self.inca = ChannelAttention(64)
        self.insa = SpatialAttention()
        self.down1 = SideDiDownwithCBAM(64, 128, sidemode='SE3', method=Method, layernum=1)
        self.down2 = SideDiDownwithCBAM(128, 256, sidemode='SE3', method=Method)
        self.down3 = SideDiDownwithCBAM(256, 512, sidemode='SE3', method=Method)
        factor = 2 if bilinear else 1
        self.down4 = SideDiDownwithCBAM(512, 1024 // factor, sidemode='SE3', method=Method, Islastlayer=True)

        # self.upsample = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))
        self.activation = nn.Sigmoid()

        # classification head
        self.linear = nn.Linear(1024, 1)

    def forward(self, x):
        # encoder
        x1 = self.inc(x)
        x1_skip = self.inca(x1) * x1
        x1_skip = self.insa(x1_skip) * x1_skip
        x2, x2_skip, side_x2 = self.down1(x1, x)
        x3, x3_skip, side_x3 = self.down2(x2, side_x2)
        x4, x4_skip, side_x4 = self.down3(x3, side_x3)
        _, x5_skip, side_x5 = self.down4(x4, side_x4)

        # decoder
        # decoder with attention gates
        x = self.up1(x5_skip, x4_skip)

        x = self.up2(x, x3_skip)

        x = self.up3(x, x2_skip)

        x = self.up4(x, x1_skip)

        # segmentation head
        logits = self.outc(x)
        # logits = self.activation(logits)

        # classification head
        clsx = F.adaptive_avg_pool2d(x5_skip, (1, 1))  # 维度变化为[batch_size, 1024, 1, 1]
        clsx = clsx.view(-1, 1024)  # [batch_size, 1024]
        label = self.linear(clsx)
        return label, logits


class SideCBAMUNet(nn.Module):
    def __init__(self, n_channels, n_classes, Method='maxpool', bilinear=False):
        super(SideCBAMUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear  # bilinear表示是否使用双线性插值

        self.inc = (DoubleConv(n_channels, 64))
        self.inca = ChannelAttention(64)
        self.insa = SpatialAttention()
        self.down1 = SideDownwithCBAM(64, 128, sidemode='SE3', method=Method, layernum=1)
        self.down2 = SideDownwithCBAM(128, 256, sidemode='SE3', method=Method)
        self.down3 = SideDownwithCBAM(256, 512, sidemode='SE3', method=Method)
        factor = 2 if bilinear else 1
        self.down4 = SideDownwithCBAM(512, 1024 // factor, sidemode='SE3', method=Method, Islastlayer=True)

        # self.upsample = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))
        self.activation = nn.Sigmoid()

        # classification head
        self.linear = nn.Linear(1024, 1)

    def forward(self, x):
        # encoder
        x1 = self.inc(x)
        x1_skip = self.inca(x1) * x1
        x1_skip = self.insa(x1_skip) * x1_skip
        x2, x2_skip, side_x2 = self.down1(x1, x)
        x3, x3_skip, side_x3 = self.down2(x2, side_x2)
        x4, x4_skip, side_x4 = self.down3(x3, side_x3)
        _, x5_skip, side_x5 = self.down4(x4, side_x4)

        # decoder
        # decoder with attention gates
        x = self.up1(x5_skip, x4_skip)

        x = self.up2(x, x3_skip)

        x = self.up3(x, x2_skip)

        x = self.up4(x, x1_skip)

        # segmentation head
        logits = self.outc(x)
        # logits = self.activation(logits)

        # classification head
        clsx = F.adaptive_avg_pool2d(x5_skip, (1, 1))  # 维度变化为[batch_size, 1024, 1, 1]
        clsx = clsx.view(-1, 1024)  # [batch_size, 1024]
        label = self.linear(clsx)
        return label, logits


class SideAgCBAMUNet(nn.Module):
    def __init__(self, n_channels, n_classes, Method='maxpool', bilinear=False):
        super(SideAgCBAMUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear  # bilinear表示是否使用双线性插值

        self.inc = (DoubleConv(n_channels, 64))
        self.inca = ChannelAttention(64)
        self.insa = SpatialAttention()
        self.down1 = (SideDownwithCBAM(64, 128, sidemode='SE', method=Method, layernum=1))
        self.down2 = (SideDownwithCBAM(128, 256, sidemode='SE', method=Method))
        self.down3 = (SideDownwithCBAM(256, 512, sidemode='SE', method=Method))
        factor = 2 if bilinear else 1
        self.down4 = (SideDownwithCBAM(512, 1024 // factor, sidemode='SE', method=Method, Islastlayer=True))

        # self.upsample = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)
        self.up1 = (AGUp(1024, 512 // factor, bilinear))
        self.up2 = (AGUp(512, 256 // factor, bilinear))
        self.up3 = (AGUp(256, 128 // factor, bilinear))
        self.up4 = (AGUp(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))
        self.activation = nn.Sigmoid()

        # classification head
        self.linear = nn.Linear(1024, 1)

    def forward(self, x):
        # encoder
        x1 = self.inc(x)
        x1_skip = self.inca(x1) * x1
        x1_skip = self.insa(x1_skip) * x1_skip
        x2, x2_skip, side_x2 = self.down1(x1, x)
        x3, x3_skip, side_x3 = self.down2(x2, side_x2)
        x4, x4_skip, side_x4 = self.down3(x3, side_x3)
        _, x5_skip, side_x5 = self.down4(x4, side_x4)

        # decoder
        # decoder with attention gates
        x = self.up1(x5_skip, x4_skip)

        x = self.up2(x, x3_skip)

        x = self.up3(x, x2_skip)

        x = self.up4(x, x1_skip)

        # segmentation head
        logits = self.outc(x)
        # logits = self.activation(logits)

        # classification head
        clsx = F.adaptive_avg_pool2d(x5_skip, (1, 1))  # 维度变化为[batch_size, 1024, 1, 1]
        clsx = clsx.view(-1, 1024)  # [batch_size, 1024]
        label = self.linear(clsx)
        return label, logits


class AgCBAMPixViTUNet(nn.Module):
    def __init__(self, img_size, n_channels, n_classes, Method='maxpool', bilinear=False):
        super(AgCBAMPixViTUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear  # bilinear表示是否使用双线性插值

        self.inc = (DoubleConv(n_channels, 64))
        self.inca = ChannelAttention(64)
        self.insa = SpatialAttention()
        self.down1 = (DownwithCBAM(64, 128, method=Method))
        self.down2 = (DownwithCBAM(128, 256, method=Method))
        self.down3 = (DownwithCBAM(256, 512, method=Method))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor, method=Method))

        self.PixViT = PixViT(
            features=1024, n_heads=8, n_blocks=2, ffn_features=4096,
            embed_features=1024, activ='gelu', norm=None,
            image_shape=(1024, img_size // 16, img_size // 16), rezero=True
        )

        # self.upsample = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)
        self.up1 = (AGUp(1024, 512 // factor, bilinear))
        self.up2 = (AGUp(512, 256 // factor, bilinear))
        self.up3 = (AGUp(256, 128 // factor, bilinear))
        self.up4 = (AGUp(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))
        self.activation = nn.Sigmoid()

        # classification head
        self.linear = nn.Linear(1024, 1)

    def forward(self, x):
        # encoder
        x1 = self.inc(x)
        x1_skip = self.inca(x1) * x1
        x1_skip = self.insa(x1_skip) * x1_skip
        x2, x2_skip = self.down1(x1)
        x3, x3_skip = self.down2(x2)
        x4, x4_skip = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.PixViT(x5)
        # decoder
        # decoder with attention gates
        x = self.up1(x5, x4_skip)

        x = self.up2(x, x3_skip)

        x = self.up3(x, x2_skip)

        x = self.up4(x, x1_skip)

        # segmentation head
        logits = self.outc(x)
        # logits = self.activation(logits)

        # classification head
        clsx = F.adaptive_avg_pool2d(x5, (1, 1))  # 维度变化为[batch_size, 1024, 1, 1]
        clsx = clsx.view(-1, 1024)  # [batch_size, 1024]
        label = self.linear(clsx)
        return label, logits


class CBAMPixViTUNet(nn.Module):
    def __init__(self, img_size, n_channels, n_classes, Method='maxpool', bilinear=False):
        super(CBAMPixViTUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear  # bilinear表示是否使用双线性插值

        self.inc = (DoubleConv(n_channels, 64))
        self.inca = ChannelAttention(64)
        self.insa = SpatialAttention()
        self.down1 = (DownwithCBAM(64, 128, method=Method))
        self.down2 = (DownwithCBAM(128, 256, method=Method))
        self.down3 = (DownwithCBAM(256, 512, method=Method))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor, method=Method))

        self.PixViT = PixViT(
            features=1024, n_heads=8, n_blocks=2, ffn_features=4096,
            embed_features=1024, activ='gelu', norm=None,
            image_shape=(1024, img_size // 16, img_size // 16), rezero=True
        )

        # self.upsample = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))
        self.activation = nn.Sigmoid()

        # classification head
        self.linear = nn.Linear(1024, 1)

    def forward(self, x):
        # encoder
        x1 = self.inc(x)
        x1_skip = self.inca(x1) * x1
        x1_skip = self.insa(x1_skip) * x1_skip
        x2, x2_skip = self.down1(x1)
        x3, x3_skip = self.down2(x2)
        x4, x4_skip = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.PixViT(x5)
        # decoder
        # decoder with attention gates
        x = self.up1(x5, x4_skip)

        x = self.up2(x, x3_skip)

        x = self.up3(x, x2_skip)

        x = self.up4(x, x1_skip)

        # segmentation head
        logits = self.outc(x)
        # logits = self.activation(logits)

        # classification head
        clsx = F.adaptive_avg_pool2d(x5, (1, 1))  # 维度变化为[batch_size, 1024, 1, 1]
        clsx = clsx.view(-1, 1024)  # [batch_size, 1024]
        label = self.linear(clsx)
        return label, logits


class SideCBAMPixViTUNet(nn.Module):
    def __init__(self, img_size, n_channels, n_classes, Method='maxpool', bilinear=False):
        super(SideCBAMPixViTUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear  # bilinear表示是否使用双线性插值

        self.inc = (DoubleConv(n_channels, 64))
        self.inca = ChannelAttention(64)
        self.insa = SpatialAttention()
        self.down1 = SideDownwithCBAM(64, 128, sidemode='SE3', method=Method, layernum=1)
        self.down2 = SideDownwithCBAM(128, 256, sidemode='SE3', method=Method)
        self.down3 = SideDownwithCBAM(256, 512, sidemode='SE3', method=Method)
        factor = 2 if bilinear else 1
        self.down4 = SideDownwithCBAM(512, 1024 // factor, sidemode='SE3', method=Method,
                                      Islastlayer=True, IsPixViT=True)

        self.PixViT = PixViT(
            features=1024, n_heads=8, n_blocks=12, ffn_features=4096,
            embed_features=1024, activ='gelu', norm=None,
            image_shape=(1024, img_size // 16, img_size // 16), rezero=True
        )

        # self.upsample = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))
        self.activation = nn.Sigmoid()

        # classification head
        self.linear = nn.Linear(1024, 1)

    def forward(self, x):
        # encoder
        x1 = self.inc(x)
        x1_skip = self.inca(x1) * x1
        x1_skip = self.insa(x1_skip) * x1_skip
        x2, x2_skip, side_x2 = self.down1(x1, x)
        x3, x3_skip, side_x3 = self.down2(x2, side_x2)
        x4, x4_skip, side_x4 = self.down3(x3, side_x3)
        _, x5_skip, side_x5 = self.down4(x4, side_x4)
        x5 = self.PixViT(x5_skip)
        # decoder
        # decoder with attention gates
        x = self.up1(x5, x4_skip)

        x = self.up2(x, x3_skip)

        x = self.up3(x, x2_skip)

        x = self.up4(x, x1_skip)

        # segmentation head
        logits = self.outc(x)
        # logits = self.activation(logits)

        # classification head
        clsx = F.adaptive_avg_pool2d(x5, (1, 1))  # 维度变化为[batch_size, 1024, 1, 1]
        clsx = clsx.view(-1, 1024)  # [batch_size, 1024]
        label = self.linear(clsx)
        return label, logits


class SideAgCBAMPixViTUNet(nn.Module):
    def __init__(self, img_size, n_channels, n_classes, Method='maxpool', bilinear=False):
        super(SideAgCBAMPixViTUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear  # bilinear表示是否使用双线性插值

        self.inc = (DoubleConv(n_channels, 64))
        self.inca = ChannelAttention(64)
        self.insa = SpatialAttention()
        self.down1 = SideDownwithCBAM(64, 128, sidemode='SE', method=Method, layernum=1)
        self.down2 = SideDownwithCBAM(128, 256, sidemode='SE', method=Method)
        self.down3 = SideDownwithCBAM(256, 512, sidemode='SE', method=Method)
        factor = 2 if bilinear else 1
        self.down4 = SideDownwithCBAM(512, 1024 // factor, sidemode='SE', method=Method,
                                      Islastlayer=True, IsPixViT=True)

        self.PixViT = PixViT(
            features=1024, n_heads=8, n_blocks=2, ffn_features=4096,
            embed_features=1024, activ='gelu', norm=None,
            image_shape=(1024, img_size // 16, img_size // 16), rezero=True
        )

        # self.upsample = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)
        self.up1 = (AGUp(1024, 512 // factor, bilinear))
        self.up2 = (AGUp(512, 256 // factor, bilinear))
        self.up3 = (AGUp(256, 128 // factor, bilinear))
        self.up4 = (AGUp(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

        # classification head
        self.linear = nn.Linear(1024, 1)

    def forward(self, x):
        # encoder
        x1 = self.inc(x)
        x1_skip = self.inca(x1) * x1
        x1_skip = self.insa(x1_skip) * x1_skip
        x2, x2_skip, side_x2 = self.down1(x1, x)
        x3, x3_skip, side_x3 = self.down2(x2, side_x2)
        x4, x4_skip, side_x4 = self.down3(x3, side_x3)
        _, x5_skip, side_x5 = self.down4(x4, side_x4)
        x5 = self.PixViT(x5_skip)
        # decoder
        # decoder with attention gates
        x = self.up1(x5, x4_skip)

        x = self.up2(x, x3_skip)

        x = self.up3(x, x2_skip)

        x = self.up4(x, x1_skip)

        # segmentation head
        logits = self.outc(x)
        # logits = self.activation(logits)

        # classification head
        clsx = F.adaptive_avg_pool2d(x5, (1, 1))  # 维度变化为[batch_size, 1024, 1, 1]
        clsx = clsx.view(-1, 1024)  # [batch_size, 1024]
        label = self.linear(clsx)
        return label, logits


class DSSideAgCBAMPixViTUNet(nn.Module):
    def __init__(self, img_size, n_channels, n_classes, Method='maxpool', bilinear=False):
        super(DSSideAgCBAMPixViTUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear  # bilinear表示是否使用双线性插值

        self.inc = (DoubleConv(n_channels, 64))
        self.inca = ChannelAttention(64)
        self.insa = SpatialAttention()
        self.down1 = SideDownwithCBAM(64, 128, sidemode='SE', method=Method, layernum=1)
        self.down2 = SideDownwithCBAM(128, 256, sidemode='SE', method=Method)
        self.down3 = SideDownwithCBAM(256, 512, sidemode='SE', method=Method)
        factor = 2 if bilinear else 1
        self.down4 = SideDownwithCBAM(512, 1024 // factor, sidemode='SE', method=Method,
                                      Islastlayer=True, IsPixViT=True)

        self.PixViT = PixViT(
            features=1024, n_heads=8, n_blocks=2, ffn_features=4096,
            embed_features=1024, activ='gelu', norm=None,
            image_shape=(1024, img_size // 16, img_size // 16), rezero=True
        )

        # self.upsample = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)
        self.up1 = (AGUp(1024, 512 // factor, bilinear))
        self.up2 = (AGUp(512, 256 // factor, bilinear))
        self.up3 = (AGUp(256, 128 // factor, bilinear))
        self.up4 = (AGUp(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

        self.sup3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.outconv3 = OutConv(512, n_classes)
        self.sup2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.outconv2 = OutConv(256, n_classes)
        self.sup1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.outconv1 = OutConv(128, n_classes)

        # classification head
        self.linear = nn.Linear(1024, 1)

    def forward(self, x):
        # encoder
        x1 = self.inc(x)
        x1_skip = self.inca(x1) * x1
        x1_skip = self.insa(x1_skip) * x1_skip
        x2, x2_skip, side_x2 = self.down1(x1, x)
        x3, x3_skip, side_x3 = self.down2(x2, side_x2)
        x4, x4_skip, side_x4 = self.down3(x3, side_x3)
        _, x5_skip, side_x5 = self.down4(x4, side_x4)
        x5 = self.PixViT(x5_skip)
        # decoder
        # decoder with attention gates
        x = self.up1(x5, x4_skip)
        sup1 = self.sup3(x)
        sup1 = self.outconv3(sup1)

        x = self.up2(x, x3_skip)
        sup2 = self.sup2(x)
        sup2 = self.outconv2(sup2)

        x = self.up3(x, x2_skip)
        sup3 = self.sup1(x)
        sup3 = self.outconv1(sup3)

        x = self.up4(x, x1_skip)

        # segmentation head
        logits = self.outc(x)
        # logits = self.activation(logits)

        # classification head
        clsx = F.adaptive_avg_pool2d(x5, (1, 1))  # 维度变化为[batch_size, 1024, 1, 1]
        clsx = clsx.view(-1, 1024)  # [batch_size, 1024]
        label = self.linear(clsx)
        return label, sup1, sup2, sup3, logits


class PixViTUNet(nn.Module):
    def __init__(self, img_size, n_channels, n_classes, Method='maxpool', bilinear=False):
        super(PixViTUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear  # bilinear表示是否使用双线性插值

        self.inc = (DoubleConv(n_channels, 64))
        self.inca = ChannelAttention(64)
        self.insa = SpatialAttention()
        self.down1 = (Down(64, 128, method=Method))
        self.down2 = (Down(128, 256, method=Method))
        self.down3 = (Down(256, 512, method=Method))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor, method=Method))

        self.PixViT = PixViT(
            features=1024, n_heads=8, n_blocks=2, ffn_features=4096,
            embed_features=1024, activ='gelu', norm=None,
            image_shape=(1024, img_size // 16, img_size // 16), rezero=True
        )

        # self.upsample = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)
        self.up1 = (AGUp(1024, 512 // factor, bilinear))
        self.up2 = (AGUp(512, 256 // factor, bilinear))
        self.up3 = (AGUp(256, 128 // factor, bilinear))
        self.up4 = (AGUp(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))
        self.activation = nn.Sigmoid()

        # classification head
        self.linear = nn.Linear(1024, 1)

    def forward(self, x):
        # encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.PixViT(x5)
        # decoder
        # decoder with attention gates
        x = self.up1(x5, x4)

        x = self.up2(x, x3)

        x = self.up3(x, x2)

        x = self.up4(x, x1)

        # segmentation head
        logits = self.outc(x)
        # logits = self.activation(logits)

        # classification head
        clsx = F.adaptive_avg_pool2d(x5, (1, 1))  # 维度变化为[batch_size, 1024, 1, 1]
        clsx = clsx.view(-1, 1024)  # [batch_size, 1024]
        label = self.linear(clsx)
        return label, logits


if __name__ == '__main__':
    img_size = 256
    model = SideAgCBAMPixViTUNet(img_size, 3, 1)
    getModelSize(model)
    # model = SideCBAMUNet(3, 1)
    model.eval()
    input = torch.randn(2, 3, img_size, img_size)
    label = torch.randn(2, 1, img_size, img_size)
    labels, logits = model(input)

    # 定义loss
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(logits, label)
    print('loss:', loss)

    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss.backward()

    print('labels:', labels)
    print('logits:', logits)

    print(labels.shape)
    print(logits.shape)
