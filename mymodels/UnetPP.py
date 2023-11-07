#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: 杨亚峰<yang_armageddon@163.com>
@version: 1.0.0
@license:  Apache Licence
@editor: Pycharm yyf
@file: UnetPP.py
@datatime: 8/28/2023 2:22 PM
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mymodels.unet.unet_utils import AttentionGate, getModelSize



# Convolutional block for single layer of the decoder and encoder
class Vgg_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(Vgg_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.pool(x)
        return x


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = Vgg_block(in_channels, out_channels)
        self.down = Downsample(in_channels, out_channels)

    def forward(self, x):
        x = self.down(x)
        for_skip = self.conv(x)
        return for_skip


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.upsample(x)
        return x


class Up(nn.Module):
    def __init__(self, in_channels, s_channels, out_channels):
        super().__init__()
        self.conv = Vgg_block(in_channels + s_channels, out_channels)
        self.upsample = Upsample(in_channels, out_channels)
        self.att = AttentionGate(F_g=out_channels * 2, F_w=out_channels, F_int=out_channels)

    def forward(self, x, for_skip, s=None):
        x = self.upsample(x)
        for_skip = self.att(x, for_skip)
        if s is not None:
            x = torch.cat([x, for_skip, s], dim=1)
        else:
            x = torch.cat([x, for_skip], dim=1)
        x = self.conv(x)
        return x, for_skip


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNetPlusPlusSeg(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = None
        self.gradients = None
        self.f_ch = [16, 32, 64, 128, 256]  # , 512, 1024]  # feature channels

        self.inc = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.down0_0 = Down(self.f_ch[0], self.f_ch[1])
        self.down1_0 = Down(self.f_ch[1], self.f_ch[2])
        self.down2_0 = Down(self.f_ch[2], self.f_ch[3])
        self.down3_0 = Down(self.f_ch[3], self.f_ch[4])

        # classificator
        self.classificator = nn.Linear(self.f_ch[4], 1)

        self.up4_0 = Up(self.f_ch[4] + self.f_ch[3] * 0, self.f_ch[3], self.f_ch[3])
        self.up3_1 = Up(self.f_ch[3] + self.f_ch[2] * 1, self.f_ch[2], self.f_ch[2])
        self.up2_2 = Up(self.f_ch[2] + self.f_ch[1] * 2, self.f_ch[1], self.f_ch[1])
        self.up1_3 = Up(self.f_ch[1] + self.f_ch[0] * 3, self.f_ch[0], self.f_ch[0])

        self.up3_0 = Up(self.f_ch[3] + self.f_ch[2] * 0, self.f_ch[2], self.f_ch[2])
        self.up2_1 = Up(self.f_ch[2] + self.f_ch[1] * 1, self.f_ch[1], self.f_ch[1])
        self.up1_2 = Up(self.f_ch[1] + self.f_ch[0] * 2, self.f_ch[0], self.f_ch[0])

        self.up2_0 = Up(self.f_ch[2] + self.f_ch[1] * 0, self.f_ch[1], self.f_ch[1])
        self.up1_1 = Up(self.f_ch[1] + self.f_ch[0] * 1, self.f_ch[0], self.f_ch[0])

        self.up1_0 = Up(self.f_ch[1], self.f_ch[0], self.f_ch[0])

        self.outc = OutConv(self.f_ch[0], out_channels)

    # 注册hook来获取特征和梯度
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        # encoder
        x0_0 = self.inc(x)  # [bs, 64, 256, 256]
        x1_0 = self.down0_0(x0_0)  # [bs, 128, 128, 128]
        x2_0 = self.down1_0(x1_0)  # [bs, 256, 64, 64]
        x3_0 = self.down2_0(x2_0)  # [bs, 512, 32, 32]
        x4_0 = self.down3_0(x3_0)  # [bs, 1024, 16, 16]

        # classifier
        clsout = F.adaptive_avg_pool2d(x4_0, (1, 1))
        clsout = clsout.view(clsout.size(0), -1)
        clsout = self.classificator(clsout)

        # decoder
        x3_1, _ = self.up4_0(x4_0, x3_0)
        x2_1, forskip2_0 = self.up3_0(x3_0, x2_0)
        x1_1, forskip1_0 = self.up2_0(x2_0, x1_0)
        x0_1, forskip0_0 = self.up1_0(x1_0, x0_0)

        x2_2, _ = self.up3_1(x3_1, x2_1, forskip2_0)
        x1_2, forskip1_1 = self.up2_1(x2_1, x1_1, forskip1_0)
        x0_2, forskip0_1 = self.up1_1(x1_1, x0_1, forskip0_0)

        x1_3, _ = self.up2_2(x2_2, x1_2, torch.cat([forskip1_0, forskip1_1], dim=1))
        x0_3, forskip0_2 = self.up1_2(x1_2, x0_2, torch.cat([forskip0_0, forskip0_1], dim=1))

        x0_4, _ = self.up1_3(x1_3, x0_3, torch.cat([forskip0_0, forskip0_1, forskip0_2], dim=1))

        self.features = x0_4  # 保存特征图
        if x0_4.requires_grad:
            h = x0_4.register_hook(self.activations_hook)

        x = self.outc(x0_4)

        return clsout, x

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self):
        return self.features


# deep supervision
class DSUNetPlusPlusSeg(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.f_ch = [64, 128, 256, 512, 1024]  # feature channels

        self.inc = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.down0_0 = Down(self.f_ch[0], self.f_ch[1])
        self.down1_0 = Down(self.f_ch[1], self.f_ch[2])
        self.down2_0 = Down(self.f_ch[2], self.f_ch[3])
        self.down3_0 = Down(self.f_ch[3], self.f_ch[4])

        self.up4_0 = Up(self.f_ch[4] + self.f_ch[3] * 0, self.f_ch[3], self.f_ch[3])
        self.up3_1 = Up(self.f_ch[3] + self.f_ch[2] * 1, self.f_ch[2], self.f_ch[2])
        self.up2_2 = Up(self.f_ch[2] + self.f_ch[1] * 2, self.f_ch[1], self.f_ch[1])
        self.up1_3 = Up(self.f_ch[1] + self.f_ch[0] * 3, self.f_ch[0], self.f_ch[0])

        self.up3_0 = Up(self.f_ch[3] + self.f_ch[2] * 0, self.f_ch[2], self.f_ch[2])
        self.up2_1 = Up(self.f_ch[2] + self.f_ch[1] * 1, self.f_ch[1], self.f_ch[1])
        self.up1_2 = Up(self.f_ch[1] + self.f_ch[0] * 2, self.f_ch[0], self.f_ch[0])

        self.up2_0 = Up(self.f_ch[2] + self.f_ch[1] * 0, self.f_ch[1], self.f_ch[1])
        self.up1_1 = Up(self.f_ch[1] + self.f_ch[0] * 1, self.f_ch[0], self.f_ch[0])

        self.up1_0 = Up(self.f_ch[1], self.f_ch[0], self.f_ch[0])

        self.outc0_1 = OutConv(self.f_ch[0], out_channels)
        self.outc1_1 = OutConv(self.f_ch[0], out_channels)
        self.outc2_1 = OutConv(self.f_ch[0], out_channels)
        self.outc3_1 = OutConv(self.f_ch[0], out_channels)

    def forward(self, x):
        # encoder
        x0_0 = self.inc(x)  # [bs, 64, 256, 256]
        x1_0 = self.down0_0(x0_0)  # [bs, 128, 128, 128]
        x2_0 = self.down1_0(x1_0)  # [bs, 256, 64, 64]
        x3_0 = self.down2_0(x2_0)  # [bs, 512, 32, 32]
        x4_0 = self.down3_0(x3_0)  # [bs, 1024, 16, 16]

        # decoder
        x3_1, _ = self.up4_0(x4_0, x3_0)
        x2_1, forskip2_0 = self.up3_0(x3_0, x2_0)
        x1_1, forskip1_0 = self.up2_0(x2_0, x1_0)
        x0_1, forskip0_0 = self.up1_0(x1_0, x0_0)

        x2_2, _ = self.up3_1(x3_1, x2_1, forskip2_0)
        x1_2, forskip1_1 = self.up2_1(x2_1, x1_1, forskip1_0)
        x0_2, forskip0_1 = self.up1_1(x1_1, x0_1, forskip0_0)

        x1_3, _ = self.up2_2(x2_2, x1_2, torch.cat([forskip1_0, forskip1_1], dim=1))
        x0_3, forskip0_2 = self.up1_2(x1_2, x0_2, torch.cat([forskip0_0, forskip0_1], dim=1))

        x0_4, _ = self.up1_3(x1_3, x0_3, torch.cat([forskip0_0, forskip0_1, forskip0_2], dim=1))

        out0_1 = self.outc0_1(x0_1)
        out0_2 = self.outc1_1(x0_2)
        out0_3= self.outc2_1(x0_3)
        out0_4 = self.outc3_1(x0_4)

        return out0_1, out0_2, out0_3, out0_4


class DSUNetPlusPlus(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.f_ch = [64, 128, 256, 512, 1024]  # feature channels

        self.inc = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.down0_0 = Down(self.f_ch[0], self.f_ch[1])
        self.down1_0 = Down(self.f_ch[1], self.f_ch[2])
        self.down2_0 = Down(self.f_ch[2], self.f_ch[3])
        self.down3_0 = Down(self.f_ch[3], self.f_ch[4])

        # classificator
        self.classificator = nn.Linear(self.f_ch[4], 1)

        self.up4_0 = Up(self.f_ch[4] + self.f_ch[3] * 0, self.f_ch[3], self.f_ch[3])
        self.up3_1 = Up(self.f_ch[3] + self.f_ch[2] * 1, self.f_ch[2], self.f_ch[2])
        self.up2_2 = Up(self.f_ch[2] + self.f_ch[1] * 2, self.f_ch[1], self.f_ch[1])
        self.up1_3 = Up(self.f_ch[1] + self.f_ch[0] * 3, self.f_ch[0], self.f_ch[0])

        self.up3_0 = Up(self.f_ch[3] + self.f_ch[2] * 0, self.f_ch[2], self.f_ch[2])
        self.up2_1 = Up(self.f_ch[2] + self.f_ch[1] * 1, self.f_ch[1], self.f_ch[1])
        self.up1_2 = Up(self.f_ch[1] + self.f_ch[0] * 2, self.f_ch[0], self.f_ch[0])

        self.up2_0 = Up(self.f_ch[2] + self.f_ch[1] * 0, self.f_ch[1], self.f_ch[1])
        self.up1_1 = Up(self.f_ch[1] + self.f_ch[0] * 1, self.f_ch[0], self.f_ch[0])

        self.up1_0 = Up(self.f_ch[1], self.f_ch[0], self.f_ch[0])

        self.outc0_1 = OutConv(self.f_ch[0], out_channels)
        self.outc1_1 = OutConv(self.f_ch[0], out_channels)
        self.outc2_1 = OutConv(self.f_ch[0], out_channels)
        self.outc3_1 = OutConv(self.f_ch[0], out_channels)

    def forward(self, x):
        # encoder
        x0_0 = self.inc(x)  # [bs, 64, 256, 256]
        x1_0 = self.down0_0(x0_0)  # [bs, 128, 128, 128]
        x2_0 = self.down1_0(x1_0)  # [bs, 256, 64, 64]
        x3_0 = self.down2_0(x2_0)  # [bs, 512, 32, 32]
        x4_0 = self.down3_0(x3_0)  # [bs, 1024, 16, 16]

        # classifier
        clsout = F.adaptive_avg_pool2d(x4_0, (1, 1))
        clsout = clsout.view(clsout.size(0), -1)
        clsout = self.classificator(clsout)

        # decoder
        x3_1, _ = self.up4_0(x4_0, x3_0)
        x2_1, forskip2_0 = self.up3_0(x3_0, x2_0)
        x1_1, forskip1_0 = self.up2_0(x2_0, x1_0)
        x0_1, forskip0_0 = self.up1_0(x1_0, x0_0)

        x2_2, _ = self.up3_1(x3_1, x2_1, forskip2_0)
        x1_2, forskip1_1 = self.up2_1(x2_1, x1_1, forskip1_0)
        x0_2, forskip0_1 = self.up1_1(x1_1, x0_1, forskip0_0)

        x1_3, _ = self.up2_2(x2_2, x1_2, torch.cat([forskip1_0, forskip1_1], dim=1))
        x0_3, forskip0_2 = self.up1_2(x1_2, x0_2, torch.cat([forskip0_0, forskip0_1], dim=1))

        x0_4, _ = self.up1_3(x1_3, x0_3, torch.cat([forskip0_0, forskip0_1, forskip0_2], dim=1))

        out0_1 = self.outc0_1(x0_1)
        out0_2 = self.outc1_1(x0_2)
        out0_3= self.outc2_1(x0_3)
        out0_4 = self.outc3_1(x0_4)

        return clsout, out0_1, out0_2, out0_3, out0_4


if __name__ == '__main__':
    method = 'convpool'
    model = UNetPlusPlusSeg(3, 1)
    print(model)
    getModelSize(model)
    # model = UNet(3, 1)
    model.eval()
    input = torch.randn(1, 3, 256, 256)
    label = torch.randn(1, 3, 256, 256)
    logits = model(input)
    # print(labels.shape)
    print(logits.shape)

