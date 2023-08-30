#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: 杨亚峰<yang_armageddon@163.com>
@version: 1.0.0
@license:  Apache Licence
@editor: Pycharm yyf
@file: unet_utils.py
@datatime: 8/18/2023 12:09 PM
"""
import torch
import torch.nn as nn
import torchvision.models as models


def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024

    # 单位转换
    param_size_MB = param_size / 1024 / 1024
    buffer_size_MB = buffer_size / 1024 / 1024

    # 参数量单位选择
    if param_sum >= 1e9:
        param_sum_str = f"{param_sum / 1e9}B"
    elif param_sum >= 1e6:
        param_sum_str = f"{param_sum / 1e6}M"
    else:
        param_sum_str = f"{param_sum / 1e3}K"

    # 格式化输出
    print(f"参数大小：{param_size_MB:.3f}MB， 参数量：{param_sum_str}， 缓存大小：{buffer_size_MB:.3f}MB， 缓存量：{buffer_sum}， 总大小：{all_size:.3f}MB")


class SideSEConv2d(nn.Module):
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
        # self.se = SEModule(out_channels)
        self.se1 = SEModule(out_channels)
        self.se2 = SEModule(out_channels)

    def forward(self, x, side):
        side = self.depthwise(side)
        side = self.pointwise(side)
        # side = self.sideconv1(side)
        side = self.bn_relu_1(side)
        side = self.se1(side)
        side = torch.cat([x, side], dim=1)  # 拼接
        # 按元素相加
        # side = x + side
        side = self.sideconv2(side)
        side = self.bn_relu_2(side)
        side = self.se2(side)
        return side


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
        side = torch.cat([x, side], dim=1)  # 拼接
        # 按元素相加
        # side = x + side
        side = self.sideconv2(side)
        side = self.bn_relu_2(side)
        return side


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


class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        # 挤压阶段：使用全局平均池化将空间维度挤压为1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 激励阶段：使用两个全连接层学习通道间的相互依赖关系
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),  # 减小维度
            nn.ReLU(inplace=True),                      # ReLU激活函数
            nn.Linear(channels // reduction, channels),  # 恢复原始维度
            nn.Sigmoid()                                # Sigmoid激活函数确保输出在0到1之间
        )

    def forward(self, x):
        # 输入x的维度为：(batch_size, channels, height, width)

        # 挤压阶段
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        # y的维度为：(batch_size, channels)

        # 激励阶段
        y = self.fc(y).view(b, c, 1, 1)
        # y的维度为：(batch_size, channels, 1, 1)

        # 将挤压和激励的结果与原始输入相乘
        return x * y.expand_as(x)
        # 输出的维度与输入相同：(batch_size, channels, height, width)


class MultiDilatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates=[1, 2, 3]):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(out_channels),  # 添加批量归一化层
                nn.ReLU(inplace=True)         # 添加ReLU激活函数
            )
            for d in dilation_rates
        ])
        self.combine_conv = nn.Conv2d(out_channels * len(dilation_rates), out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)  # 添加批量归一化层
        self.act = nn.SiLU(inplace=True)  # 添加SiLU激活函数

    def forward(self, x):
        features = [conv(x) for conv in self.convs]
        combined_features = torch.cat(features, dim=1)
        return self.act(self.bn(self.combine_conv(combined_features)))


class CascadedDilatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates=[1, 2]):
        super().__init__()
        self.convs = nn.Sequential(*[
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=d, dilation=d, bias=False)
            for d in dilation_rates
        ])

    def forward(self, x):
        return self.convs(x)


class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1):
        super(ResidualBlock, self).__init__()
        assert output_channels % 4 == 0, "Output channels must be divisible by 4"
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride
        self.bn1 = nn.BatchNorm2d(input_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input_channels, output_channels // 4, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(output_channels // 4)
        self.conv2 = nn.Conv2d(output_channels // 4, output_channels // 4, 3, stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(output_channels // 4)
        self.conv3 = nn.Conv2d(output_channels // 4, output_channels, 1, 1, bias=False)
        self.conv4 = nn.Conv2d(input_channels, output_channels, 1, stride, bias=False)

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out1 = self.relu(out)
        out = self.conv1(out1)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        if self.input_channels != self.output_channels or self.stride != 1:
            residual = self.conv4(residual)
        out += residual
        return out


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_w, F_int):
        super(AttentionGate, self).__init__()
        # F_g 是g的通道数，F_w 是x的通道数，F_int 是中间层的通道数
        # W_g用于对g进行特征转换，包含一个卷积层和一个批量归一化层
        # 卷积层的作用是对输入特征进行线性变换，批量归一化层的作用是对特征进行归一化，使得网络更容易训练
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        # W_x用于对x进行特征转换，包含一个卷积层和一个批量归一化层
        # 卷积层和批量归一化层的作用同上
        self.W_x = nn.Sequential(
            nn.Conv2d(F_w, F_int, kernel_size=1, stride=1, padding=0, bias=True),
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


class Res101Encoder(nn.Module):
    def __init__(self):
        super(Res101Encoder, self).__init__()
        resnet = models.resnet101(pretrained=True)
        self.layer1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer2 = resnet.layer1
        self.layer3 = resnet.layer2
        self.layer4 = resnet.layer3
        self.layer5 = resnet.layer4
        # Add the rest of the U-Net architecture here.

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        # Now, x1, x2, x3 and x4 are the output feature maps at different stages in the encoder.
        # You can use these in your decoder part of the U-Net.
        return x1, x2, x3, x4, x5