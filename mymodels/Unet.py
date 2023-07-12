""" Full assembly of the parts to form the complete network """

""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # Attention gate
        self.att = AttentionGate(F_g=in_channels // 2, F_l=out_channels, F_int=in_channels // 4)

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

        # Attention gate
        x2 = self.att(g=x1, x=x2)

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


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



class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear    # bilinear表示是否使用双线性插值

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        # self.upsample = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))
        self.activation = nn.Sigmoid()

        # classification head
        self.linear = nn.Linear(1024, 1)
        # self.linear1 = nn.Linear(1024, 512)
        # self.dropout = nn.Dropout(0.2)
        # self.linear2 = nn.Linear(512, 1)
        # self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # print('x5.shape', x5.shape)
        # print('x4.shape', x4.shape)
        # print('x3.shape', x3.shape)
        # print('x2.shape', x2.shape)
        # print('x1.shape', x1.shape)
        #
        # x5_upsampled = self.upsample(x5)
        # x4_upsampled = self.upsample(x4)
        # x3_upsampled = self.upsample(x3)
        # x2_upsampled = self.upsample(x2)

        # print('x5_upsampled.shape', x5_upsampled.shape)
        # print('x4_upsampled.shape', x4_upsampled.shape)
        # print('x3_upsampled.shape', x3_upsampled.shape)
        # print('x2_upsampled.shape', x2_upsampled.shape)


        # 将特征图cat起来,这样可以捕获多尺度的特征，以获取更好的语义信息
        # features_concatenated = torch.cat([x1, x2_upsampled, x3_upsampled, x4_upsampled, x5_upsampled], dim=1)
        # print('features_concatenated.shape', features_concatenated.shape)

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

class UNetcls(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNetcls, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear    # bilinear表示是否使用双线性插值

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
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

if __name__ == '__main__':
    model = UNet(1, 1)
    model.eval()
    input = torch.randn(10, 1, 256, 256)
    label = torch.randn(10, 1, 256, 256)
    labels, logits = model(input)
    print(labels.shape)
    print(logits.shape)



