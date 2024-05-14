import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicDepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(DynamicDepthwiseSeparableConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        reduced_channels = max(8, in_channels // 32)  # 减少中间层的通道数，减少参数量
        self.depthwise_kernel_generator = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(reduced_channels, in_channels * kernel_size * kernel_size, kernel_size=1),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.pointwise_kernel_generator = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(reduced_channels, out_channels * in_channels, kernel_size=1),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        batch_size, in_channels, height, width = x.size()

        # 动态生成深度卷积核
        depthwise_kernel = self.depthwise_kernel_generator(x)
        depthwise_kernel = depthwise_kernel.view(batch_size, in_channels, 1, self.kernel_size, self.kernel_size)

        # 动态生成逐点卷积核
        pointwise_kernel = self.pointwise_kernel_generator(x)
        pointwise_kernel = pointwise_kernel.view(batch_size, self.out_channels, in_channels, 1, 1)

        output = []

        for i in range(batch_size):
            depthwise_output = F.conv2d(x[i:i+1], depthwise_kernel[i], stride=self.stride, padding=self.padding, groups=in_channels)
            output.append(F.conv2d(depthwise_output, pointwise_kernel[i], bias=self.bias))

        output = torch.cat(output, dim=0)
        return output

class DynamicUNet(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels=64):
        super(DynamicUNet, self).__init__()
        self.base_channels = base_channels

        self.enc1 = self.conv_block(in_channels, self.base_channels)
        self.enc2 = self.conv_block(self.base_channels, self.base_channels * 2)
        self.enc3 = self.conv_block(self.base_channels * 2, self.base_channels * 4)
        self.enc4 = self.conv_block(self.base_channels * 4, self.base_channels * 8)  # 保持最大通道数为512

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = self.conv_block(self.base_channels * 8, self.base_channels * 16)

        self.upconv4 = nn.ConvTranspose2d(self.base_channels * 16, self.base_channels * 8, 2, stride=2)
        self.dec4 = self.conv_block(self.base_channels * 16, self.base_channels * 8)
        self.upconv3 = nn.ConvTranspose2d(self.base_channels * 8, self.base_channels * 4, 2, stride=2)
        self.dec3 = self.conv_block(self.base_channels * 8, self.base_channels * 4)
        self.upconv2 = nn.ConvTranspose2d(self.base_channels * 4, self.base_channels * 2, 2, stride=2)
        self.dec2 = self.conv_block(self.base_channels * 4, self.base_channels * 2)
        self.upconv1 = nn.ConvTranspose2d(self.base_channels * 2, self.base_channels, 2, stride=2)
        self.dec1 = self.conv_block(self.base_channels * 2, self.base_channels)

        self.final_conv = nn.Conv2d(self.base_channels, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            DynamicDepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0),  # 使用1x1卷积
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))  # 保持四层

        bottleneck = self.bottleneck(self.pool(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)  # 确保拼接时通道数匹配
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)

        final_output = self.final_conv(dec1)
        return final_output


if __name__ == '__main__':
    # 检查是否有可用的GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建模型实例并移动到GPU
    model = DynamicUNet(in_channels=3, out_channels=1, base_channels=16).to(device)  # 这里可以调整base_channels

    # 创建输入张量并移动到GPU
    input_tensor = torch.randn(1, 3, 256, 256).to(device)

    # 运行模型
    output_tensor = model(input_tensor)

    # 打印输出张量的形状
    print(output_tensor.shape)  # 输出形状应该是 (1, 1, 256, 256)

    # 打印模型参数量
    num_params = sum(p.numel() for p in model.parameters())
    print(f'Number of model parameters: {num_params}')

    # 计算模型大小
    model_size = num_params * 4 / (1024 ** 2)  # assuming 32-bit (4 bytes) floats
    print(f'Model size: {model_size:.2f} MB')
