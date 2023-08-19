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
import torchvision.models as models
from segmentation_models_pytorch.encoders import get_encoder
from mymodels.generatorGAN import PixelwiseViT as PixViT


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


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
                nn.SiLU(inplace=True),
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
        self.bilinear = bilinear    # bilinear表示是否使用双线性插值

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
        clsx = F.adaptive_avg_pool2d(x5_skip, (1, 1))   # 维度变化为[batch_size, 1024, 1, 1]
        clsx = clsx.view(-1, 1024)  # [batch_size, 1024]
        label = self.linear(clsx)
        return label, logits


class AgCBAMPixViTUNet(nn.Module):
    def __init__(self, img_size, n_channels, n_classes, bilinear=False):
        super(AgCBAMPixViTUNet, self).__init__()
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
        self.down4 = (Down(512, 1024 // factor, method='maxpool'))

        self.PixViT = PixViT(
            features=1024, n_heads=8, n_blocks=12, ffn_features=4096,
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


if __name__ == '__main__':
    img_size = 512
    model = AgCBAMPixViTUNet(img_size, 3, 1)
    # model = UNet(3, 1)
    model.eval()
    input = torch.randn(10, 3, img_size, img_size)
    label = torch.randn(10, 3, img_size, img_size)
    labels, logits = model(input)
    print('labels:', labels)
    print('logits:', logits)

    print(labels.shape)
    print(logits.shape)









