""" Full assembly of the parts to form the complete network """

""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from segmentation_models_pytorch.encoders import get_encoder
from mymodels.CBAMUnet import ChannelAttention, SpatialAttention
from mymodels.unet.unet_utils import MultiDilatedConv, CascadedDilatedConv, ResidualBlock, AttentionGate, Res101Encoder


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


class ResidualDown(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels, poolmethod='maxpool', num_res_blocks=1):
        super().__init__()
        self.poolmethod = poolmethod
        res_blocks = [ResidualBlock(out_channels if i == 0 else out_channels, out_channels)
                      for i in range(num_res_blocks)]
        if poolmethod == 'maxpool':
            # 方法一：
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                nn.Conv2d(in_channels, in_channels, kernel_size=1),
                nn.BatchNorm2d(in_channels),
                nn.SiLU(inplace=True),
                *res_blocks,
                nn.Dropout(0.1)
            )
        elif poolmethod == 'convpool':
            # 方法二：
            self.convpool_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True),
                *res_blocks,
                nn.Dropout(0.1)
            )

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        if self.poolmethod == 'maxpool':
            return self.maxpool_conv(x)
        else:
            return self.convpool_conv(x)


class ReplaceDilatedDown(nn.Module):
    """ 包含空洞多尺度空洞卷积的下采样模块"""
    def __init__(self, in_channels, out_channels, poolmethod='maxpool', num_dilated_convs=1):
        super().__init__()
        self.poolmethod = poolmethod
        dilated_convs = [MultiDilatedConv(out_channels if i == 0 else out_channels, out_channels)
                         for i in range(num_dilated_convs)]
        if poolmethod == 'maxpool':
            # 方法一：
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True),
                *dilated_convs,
                nn.Dropout(0.1)
            )
        elif poolmethod == 'convpool':
            # 方法二：
            self.convpool_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True),
                *dilated_convs,
                nn.Dropout(0.1)
            )

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        if self.poolmethod == 'maxpool':
            return self.maxpool_conv(x)
        else:
            return self.convpool_conv(x)


class InsertDilatedDown(nn.Module):
    """在DoubleConv后插入MultiDilatedConv的下采样模块"""
    def __init__(self, in_channels, out_channels, poolmethod='maxpool'):
        super().__init__()
        self.poolmethod = poolmethod

        if poolmethod == 'maxpool':
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                nn.Conv2d(in_channels, in_channels, kernel_size=1),
                nn.BatchNorm2d(in_channels),
                nn.SiLU(inplace=True),
                DoubleConv(in_channels, out_channels),
                MultiDilatedConv(out_channels, out_channels),
                nn.Dropout(0.1)
            )
        elif poolmethod == 'convpool':
            self.convpool_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True),
                DoubleConv(out_channels, out_channels),
                MultiDilatedConv(out_channels, out_channels),
                nn.Dropout(0.1)
            )

    def forward(self, x):
        if self.poolmethod == 'maxpool':
            return self.maxpool_conv(x)
        else:
            return self.convpool_conv(x)


class SideConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 使用深度可分离卷积，5x5卷积核
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=2, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # self.sideconv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=2, padding=2)
        # BN + ReLU
        self.bn_relu_1 = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 将侧边特征图下采样后与下采样后的特征图拼接后再进行一次卷积
        self.sideconv2 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)
        # BN + ReLU
        self.bn_relu_2 = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, side):
        side = self.depthwise(side)
        side = self.pointwise(side)
        # side = self.sideconv1(side)
        side = self.bn_relu_1(side)
        side = torch.cat([x, side], dim=1)
        side = self.sideconv2(side)
        side = self.bn_relu_2(side)
        return side


class SideDown(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, method='maxpool', layernum=222):
        super().__init__()
        self.method = method
        if layernum == 1:
            side_in_channels = 3
        else:
            side_in_channels = in_channels
        self.sideconv = SideConv2d(side_in_channels, out_channels)
        if method == 'maxpool':
            # 方法一：
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(in_channels, out_channels),
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

    def forward(self, x, side):
        if self.method == 'maxpool':
            downsample = self.maxpool_conv(x)
        else:
            downsample = self.convpool_conv(x)

        side = self.sideconv(downsample, side)
        return downsample, side


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, method='maxpool'):
        super().__init__()
        self.method = method
        if method == 'maxpool':
            # 方法一：
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(in_channels, out_channels),
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


class UNetseg(nn.Module):
    def __init__(self, n_channels, n_classes, method='maxpool', bilinear=False):
        super(UNetseg, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear    # bilinear表示是否使用双线性插值

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128, method))
        self.down2 = (Down(128, 256, method))
        self.down3 = (Down(256, 512, method))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor, method))
        # self.upsample = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))
        self.activation = nn.Sigmoid()

    def forward(self, x):
        # encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)

        x = self.up2(x, x3)

        x = self.up3(x, x2)

        x = self.up4(x, x1)

        # segmentation head
        logits = self.outc(x)
        # logits = self.activation(logits)
        return logits


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, method='maxpool', bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear    # bilinear表示是否使用双线性插值

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128, method))
        self.down2 = (Down(128, 256, method))
        self.down3 = (Down(256, 512, method))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor, method))
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
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

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
        clsx = F.adaptive_avg_pool2d(x5, (1, 1))   # 维度变化为[batch_size, 1024, 1, 1]
        clsx = clsx.view(-1, 1024)  # [batch_size, 1024]
        label = self.linear(clsx)
        return label, logits


class SideUNet(nn.Module):
    def __init__(self, n_channels, n_classes, method='maxpool', bilinear=False):
        super(SideUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear    # bilinear表示是否使用双线性插值

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (SideDown(64, 128, method, layernum=1))
        self.down2 = (SideDown(128, 256, method))
        self.down3 = (SideDown(256, 512, method))
        factor = 2 if bilinear else 1
        self.down4 = (SideDown(512, 1024 // factor, method))
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
        x2, side_x2 = self.down1(x1, x)
        x3, side_x3 = self.down2(x2, side_x2)
        x4, side_x4 = self.down3(x3, side_x3)
        x5, side_x5 = self.down4(x4, side_x4)
        # x5 就没用了，清除掉
        del x5

        # decoder
        x = self.up1(side_x5, x4)

        x = self.up2(x, x3)

        x = self.up3(x, x2)

        x = self.up4(x, x1)

        # segmentation head
        logits = self.outc(x)
        # logits = self.activation(logits)

        # classification head
        clsx = F.adaptive_avg_pool2d(side_x5, (1, 1))   # 维度变化为[batch_size, 1024, 1, 1]
        clsx = clsx.view(-1, 1024)  # [batch_size, 1024]
        label = self.linear(clsx)
        return label, logits


class InDilatedUNet(nn.Module):
    def __init__(self, n_channels, n_classes, method='maxpool', bilinear=False):
        super(InDilatedUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear    # bilinear表示是否使用双线性插值
        num_dilated_convs = [1, 2, 3, 3, 3]
        self.inc = MultiDilatedConv(n_channels, 64)
        self.down1 = ReplaceDilatedDown(64, 128, poolmethod='maxpool', num_dilated_convs=num_dilated_convs[1])
        self.down2 = ReplaceDilatedDown(128, 256, poolmethod='maxpool', num_dilated_convs=num_dilated_convs[2])
        self.down3 = ReplaceDilatedDown(256, 512, poolmethod='maxpool', num_dilated_convs=num_dilated_convs[3])
        factor = 2 if bilinear else 1
        self.down4 = ReplaceDilatedDown(512, 1024 // factor, poolmethod='maxpool', num_dilated_convs=num_dilated_convs[4])
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
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

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
        clsx = F.adaptive_avg_pool2d(x5, (1, 1))   # 维度变化为[batch_size, 1024, 1, 1]
        clsx = clsx.view(-1, 1024)  # [batch_size, 1024]
        label = self.linear(clsx)
        return label, logits


class ResUNet(nn.Module):
    def __init__(self, n_channels, n_classes, method='maxpool', bilinear=False):
        super(ResUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear    # bilinear表示是否使用双线性插值
        self.num_res_blocks = [1, 3, 6, 12]
        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (ResidualDown(64, 128, method, self.num_res_blocks[0]))
        self.down2 = (ResidualDown(128, 256, method, self.num_res_blocks[1]))
        self.down3 = (ResidualDown(256, 512, method, self.num_res_blocks[2]))
        factor = 2 if bilinear else 1
        self.down4 = (ResidualDown(512, 1024 // factor, method, self.num_res_blocks[3]))
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
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

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
        clsx = F.adaptive_avg_pool2d(x5, (1, 1))   # 维度变化为[batch_size, 1024, 1, 1]
        clsx = clsx.view(-1, 1024)  # [batch_size, 1024]
        label = self.linear(clsx)
        return label, logits


class AgUNet(nn.Module):
    def __init__(self, n_channels, n_classes, method='maxpool', bilinear=False):
        super(AgUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear    # bilinear表示是否使用双线性插值

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128, method))
        self.down2 = (Down(128, 256, method))
        self.down3 = (Down(256, 512, method))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor, method))
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
        clsx = F.adaptive_avg_pool2d(x5, (1, 1))   # 维度变化为[batch_size, 1024, 1, 1]
        clsx = clsx.view(-1, 1024)  # [batch_size, 1024]
        label = self.linear(clsx)
        return label, logits


class AgUNetseg(nn.Module):
    def __init__(self, n_channels, n_classes, method='maxpool', bilinear=False):
        super(AgUNetseg, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear    # bilinear表示是否使用双线性插值

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128, method))
        self.down2 = (Down(128, 256, method))
        self.down3 = (Down(256, 512, method))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor, method))
        # self.upsample = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)
        self.up1 = (AGUp(1024, 512 // factor, bilinear))
        self.up2 = (AGUp(512, 256 // factor, bilinear))
        self.up3 = (AGUp(256, 128 // factor, bilinear))
        self.up4 = (AGUp(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))
        self.activation = nn.Sigmoid()


    def forward(self, x):
        # encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # decoder
        # decoder with attention gates
        x = self.up1(x5, x4)

        x = self.up2(x, x3)

        x = self.up3(x, x2)

        x = self.up4(x, x1)

        # segmentation head
        logits = self.outc(x)
        # logits = self.activation(logits)

        return logits


class UNetcls(nn.Module):
    def __init__(self, n_channels, n_classes, method='maxpool', bilinear=False):
        super(UNetcls, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear    # bilinear表示是否使用双线性插值

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128, method))
        self.down2 = (Down(128, 256, method))
        self.down3 = (Down(256, 512, method))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor, method))
        # classification head
        self.linear = nn.Linear(1024, 1)

    def forward(self, x):
        # encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # classification head
        clsx = F.adaptive_avg_pool2d(x5, (1, 1))   # 维度变化为[batch_size, 1024, 1, 1]
        clsx = clsx.view(-1, 1024)  # [batch_size, 1024]
        label = self.linear(clsx)
        return label

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


class Res101UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(Res101UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear  # bilinear表示是否使用双线性插值

        self.encoder = Res101Encoder()
        factor = 2 if bilinear else 1
        self.up1 = (AGUp(2048, 1024 // factor, bilinear))
        self.up2 = (AGUp(1024, 512 // factor, bilinear))
        self.up3 = (AGUp(512, 256 // factor, bilinear))
        self.up4 = (AGUp(256, 64, bilinear))
        self.outc = (OutConv(64, n_classes))
        self.up5 = nn.ConvTranspose2d(64, 64, 2, stride=2)  # ConvTranspose2d是为了将特征图的尺寸放大一倍，
        self.ca5 = ChannelAttention(2048)
        # 参数分别为输入通道数，输出通道数，卷积核大小，步长，padding
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.activation = nn.Sigmoid()

        # classification head
        self.linear = nn.Linear(2048, 1)

    def forward(self, x):
        # encoder
        x1, x2, x3, x4, x5 = self.encoder(x)
        x5 = self.ca5(x5) * x5
        # features = self.encoder(x)
        # features = features[1:]  # remove first skip with same spatial resolution
        # x1, x2, x3, x4, x5 = features

        # decoder with attention gates
        x = self.up1(x5, x4)

        x = self.up2(x, x3)

        x = self.up3(x, x2)

        x = self.up4(x, x1)
        x = self.up5(x)
        # segmentation head
        x = self.outc(x)
        # logits = self.activation(logits)

        logits = self.upsample(x)


        # classification head
        clsx = F.adaptive_avg_pool2d(x5, (1, 1))   # 维度变化为[batch_size, 2048, 1, 1]
        clsx = clsx.view(-1, 2048)
        label = self.linear(clsx)
        return label, logits


if __name__ == '__main__':
    method = 'convpool'
    model = SideUNet(3, 1)
    # model = UNet(3, 1)
    model.eval()
    input = torch.randn(10, 3, 256, 256)
    label = torch.randn(10, 3, 256, 256)
    labels, logits = model(input)
    print(labels.shape)
    print(logits.shape)








