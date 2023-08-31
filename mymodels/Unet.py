
import torch
import torch.nn as nn
import torch.nn.functional as F
from mymodels.unet.unet_utils import getModelSize
from mymodels.CBAMUnet import ChannelAttention, SpatialAttention
from mymodels.unet.unet_utils import MultiDilatedConv, CascadedDilatedConv, ResidualBlock, AttentionGate, \
    Res101Encoder, SEModule, SideSEConv2d, SideConv2d


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
    """ 包含多尺度空洞卷积的下采样模块"""

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
        self.se = SEModule(out_channels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        if self.poolmethod == 'maxpool':
            return self.se(self.maxpool_conv(x))
        else:
            return self.se(self.convpool_conv(x))


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


class SideDown(nn.Module):
    """Downscaling with maxpool then double conv
       Param：
           in_channels：输入通道数
           out_channels：输出通道数
           method：下采样方法，有两种：'maxpool'和'convpool'
           sidemode：侧边模块，有两种：'SE'和'Conv'
           layernum：当前层数，用于判断是否为第一层， 如果是第一层，侧边模块的输入通道数为3，否则为in_channels
    """

    def __init__(self, in_channels, out_channels, sidemode='SE', method='maxpool', layernum=222):
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


class SideDiDown(nn.Module):
    """Downscaling with maxpool then double conv
       Param：
           in_channels：输入通道数
           out_channels：输出通道数
           method：下采样方法，有两种：'maxpool'和'convpool'
           sidemode：侧边模块，有两种：'SE'和'Conv'
           layernum：当前层数，用于判断是否为第一层， 如果是第一层，侧边模块的输入通道数为3，否则为in_channels
    """

    def __init__(self, in_channels, out_channels, sidemode='SE', method='maxpool', layernum=222):
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
                nn.SiLU(inplace=True),
                DoubleConv(out_channels, out_channels),
                MultiDilatedConv(out_channels, out_channels),
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
        self.att = AttentionGate(F_g=in_channels // 2, F_w=out_channels, F_int=in_channels // 4)

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
        self.bilinear = bilinear  # bilinear表示是否使用双线性插值

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
        self.bilinear = bilinear  # bilinear表示是否使用双线性插值

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
        clsx = F.adaptive_avg_pool2d(x5, (1, 1))  # 维度变化为[batch_size, 1024, 1, 1]
        clsx = clsx.view(-1, 1024)  # [batch_size, 1024]
        label = self.linear(clsx)
        return label, logits


class SideUNet(nn.Module):
    def __init__(self, n_channels, n_classes, sidemode='SE', Method='maxpool', bilinear=False):
        super(SideUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear  # bilinear表示是否使用双线性插值

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (SideDown(64, 128, sidemode, Method, layernum=1))
        self.down2 = (SideDown(128, 256, sidemode, Method))
        self.down3 = (SideDown(256, 512, sidemode, Method))
        factor = 2 if bilinear else 1
        self.down4 = (SideDown(512, 1024 // factor, sidemode, Method))
        # self.upsample = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

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
        clsx = F.adaptive_avg_pool2d(side_x5, (1, 1))  # 维度变化为[batch_size, 1024, 1, 1]
        clsx = clsx.view(-1, 1024)  # [batch_size, 1024]
        label = self.linear(clsx)
        return label, logits


class SideDiUNet(nn.Module):
    def __init__(self, n_channels, n_classes, sidemode='SE', Method='maxpool', bilinear=False):
        super(SideDiUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear  # bilinear表示是否使用双线性插值

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (SideDiDown(64, 128, sidemode, Method, layernum=1))
        self.down2 = (SideDiDown(128, 256, sidemode, Method))
        self.down3 = (SideDiDown(256, 512, sidemode, Method))
        factor = 2 if bilinear else 1
        self.down4 = (SideDiDown(512, 1024 // factor, sidemode, Method))
        # self.upsample = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

        # classification head
        self.linear = nn.Linear(1024, 1)

    def forward(self, x):
        # encoder
        x1 = self.inc(x)
        x2, side_x2 = self.down1(x1, x)
        x3, side_x3 = self.down2(x2, side_x2)
        x4, side_x4 = self.down3(x3, side_x3)
        _, side_x5 = self.down4(x4, side_x4)

        # decoder
        x = self.up1(side_x5, x4)

        x = self.up2(x, x3)

        x = self.up3(x, x2)

        x = self.up4(x, x1)

        # segmentation head
        logits = self.outc(x)
        # logits = self.activation(logits)

        # classification head
        clsx = F.adaptive_avg_pool2d(side_x5, (1, 1))  # 维度变化为[batch_size, 1024, 1, 1]
        clsx = clsx.view(-1, 1024)  # [batch_size, 1024]
        label = self.linear(clsx)
        return label, logits


class InDilatedUNet(nn.Module):
    def __init__(self, n_channels, n_classes, Method='maxpool', bilinear=False):
        super(InDilatedUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear  # bilinear表示是否使用双线性插值
        num_dilated_convs = [2, 2, 2, 2]
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = ReplaceDilatedDown(64, 128, poolmethod=Method, num_dilated_convs=num_dilated_convs[0])
        self.down2 = ReplaceDilatedDown(128, 256, poolmethod=Method, num_dilated_convs=num_dilated_convs[1])
        self.down3 = ReplaceDilatedDown(256, 512, poolmethod=Method, num_dilated_convs=num_dilated_convs[2])
        factor = 2 if bilinear else 1
        self.down4 = ReplaceDilatedDown(512, 1024 // factor, poolmethod=Method, num_dilated_convs=num_dilated_convs[3])
        # self.upsample = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

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
        clsx = F.adaptive_avg_pool2d(x5, (1, 1))  # 维度变化为[batch_size, 1024, 1, 1]
        clsx = clsx.view(-1, 1024)  # [batch_size, 1024]
        label = self.linear(clsx)
        return label, logits


class CasDilatedUNet(nn.Module):
    def __init__(self, n_channels, n_classes, Method='maxpool', bilinear=False):
        super(CasDilatedUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear  # bilinear表示是否使用双线性插值
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = InsertDilatedDown(64, 128, poolmethod=Method)
        self.down2 = InsertDilatedDown(128, 256, poolmethod=Method)
        self.down3 = InsertDilatedDown(256, 512, poolmethod=Method)
        factor = 2 if bilinear else 1
        self.down4 = InsertDilatedDown(512, 1024 // factor, poolmethod=Method)
        # self.upsample = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

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
        clsx = F.adaptive_avg_pool2d(x5, (1, 1))  # 维度变化为[batch_size, 1024, 1, 1]
        clsx = clsx.view(-1, 1024)  # [batch_size, 1024]
        label = self.linear(clsx)
        return label, logits


class ResUNet(nn.Module):
    def __init__(self, n_channels, n_classes, method='maxpool', bilinear=False):
        super(ResUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear  # bilinear表示是否使用双线性插值
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
        clsx = F.adaptive_avg_pool2d(x5, (1, 1))  # 维度变化为[batch_size, 1024, 1, 1]
        clsx = clsx.view(-1, 1024)  # [batch_size, 1024]
        label = self.linear(clsx)
        return label, logits


class AgUNet(nn.Module):
    def __init__(self, n_channels, n_classes, method='maxpool', bilinear=False):
        super(AgUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear  # bilinear表示是否使用双线性插值

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
        clsx = F.adaptive_avg_pool2d(x5, (1, 1))  # 维度变化为[batch_size, 1024, 1, 1]
        clsx = clsx.view(-1, 1024)  # [batch_size, 1024]
        label = self.linear(clsx)
        return label, logits


class AgUNetseg(nn.Module):
    def __init__(self, n_channels, n_classes, method='maxpool', bilinear=False):
        super(AgUNetseg, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear  # bilinear表示是否使用双线性插值

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
        self.bilinear = bilinear  # bilinear表示是否使用双线性插值

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
        clsx = F.adaptive_avg_pool2d(x5, (1, 1))  # 维度变化为[batch_size, 1024, 1, 1]
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
        clsx = F.adaptive_avg_pool2d(x5, (1, 1))  # 维度变化为[batch_size, 2048, 1, 1]
        clsx = clsx.view(-1, 2048)
        label = self.linear(clsx)
        return label, logits


class M_down(nn.Module):
    """
    M-UNet 的下采样加卷积模块
    包含一个maxpooling, 两个卷积层
    """

    def __init__(self, in_channels, out_channels, cat_channels):
        super(M_down, self).__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(in_channels + cat_channels, in_channels, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(in_channels * 2, out_channels, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, cat):
        x1 = self.pool(x)
        x = torch.cat([x1, cat], dim=1)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = torch.cat([x, x1], dim=1)
        x = self.relu2(self.bn2(self.conv2(x)))
        return x


class M_legleft(nn.Module):
    """
    M-UNet 的左侧卷积模块,包含一个maxpooling，就完了
    """

    def __init__(self):
        super(M_legleft, self).__init__()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.pool(x)
        return x


class M_up(nn.Module):
    """
    M-UNet 的上采样加卷积模块
    包含一个上采样，两个卷积层
    """

    def __init__(self, in_channels, out_channels):
        super(M_up, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(in_channels + in_channels, out_channels, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x1 = self.up(x)
        x = torch.cat([x1, skip], dim=1)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = torch.cat([x, x1], dim=1)
        x = self.relu2(self.bn2(self.conv2(x)))
        return x


class M_legright(nn.Module):
    """
    M-UNet 的右侧卷积模块,包含一个上采样, 以及在forward中的cat
    """

    def __init__(self):
        super(M_legright, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        return self.up(x)


class M_UNet_seg(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(M_UNet_seg, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear  # bilinear表示是否使用双线性插值

        self.inc1 = nn.Sequential(
            nn.Conv2d(n_channels, 16, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))
        self.inc2 = nn.Sequential(
            nn.Conv2d(16 + n_channels, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.cat_channels = n_channels
        self.down1 = M_down(32, 48, self.cat_channels)
        self.left1 = M_legleft()
        self.down2 = M_down(48, 64, self.cat_channels)
        self.left2 = M_legleft()
        self.down3 = M_down(64, 128, self.cat_channels)
        self.left3 = M_legleft()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 1)

        self.bottleneck = nn.Conv2d(128, 64, kernel_size=(3, 3), padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.up1 = M_up(64, 48)
        self.right1 = M_legright()
        self.up2 = M_up(48, 32)
        self.right2 = M_legright()
        self.up3 = M_up(32, 16)
        self.right3 = M_legright()

        self.outc = nn.Conv2d(64 + 48 + 32 + 16, n_classes, kernel_size=(1, 1))

    def forward(self, x):
        # encoder
        x1 = self.inc1(x)  # x: 256, 256, 3 ---- x1: 256, 256, 16
        xl1 = self.left1(x)  # xl1: 128 x 128 x 3
        x1 = self.inc2(torch.cat([x, x1], dim=1))  # x1: 256, 256, 32
        x2 = self.down1(x1, xl1)  # x2: 128, 128, 48
        xl2 = self.left2(xl1)  # xl2: 64, 64, 3
        x3 = self.down2(x2, xl2)  # x3: 64, 64, 64
        xl3 = self.left3(xl2)  # xl3: 32, 32, 3
        x4 = self.down3(x3, xl3)  # x4: 32, 32, 128

        # classifier
        clsout = self.avgpool(x4)
        clsout = clsout.view(clsout.size(0), -1)
        clsout = self.fc(clsout)

        # bottleneck
        x = self.bottleneck(x4)  # x: 32, 32, 64
        x = self.bn(x)
        x = self.relu(x)

        # decoder
        xr1 = self.right1(x)  # xr1: 64, 64, 64
        x = self.up1(x, x3)  # x: 64, 64, 48
        xr1 = torch.cat([xr1, x], dim=1)  # xr1: 64, 64, 64 + 48
        xr2 = self.right2(xr1)  # xr2: 128, 128, 64 + 48
        x = self.up2(x, x2)  # x: 128, 128, 32
        xr2 = torch.cat([xr2, x], dim=1)  # xr2: 128, 128, 64 + 48 + 32
        xr3 = self.right3(xr2)  # xr3: 256, 256, 32
        x = self.up3(x, x1)  # x: 256, 256, 16
        xr3 = torch.cat([xr3, x], dim=1)  # xr3: 256, 256, 64 + 48 + 32 + 16

        logits = self.outc(xr3)  # logits: 256, 256, 1

        return clsout, logits


if __name__ == '__main__':
    method = 'convpool'
    model = M_UNet_seg(3, 1)
    print(model)
    getModelSize(model)
    # model = UNet(3, 1)
    model.eval()
    input = torch.randn(10, 3, 256, 256)
    label = torch.randn(10, 3, 256, 256)
    labels, logits = model(input)
    print(labels.shape)
    print(logits.shape)
