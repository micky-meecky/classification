import torch
from torch import nn
from segmentation_models_pytorch.encoders import get_encoder

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        return x*psi


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = get_encoder(
            'resnet50',
            in_channels=3,
            depth=5,
            weights='imagenet',
        )
        self.decoder = self._initialize_decoder()
        self.linear = nn.Linear(2048, 1)
        self.n_classes = 1

    def _initialize_decoder(self):
        decoder = nn.ModuleList([
            DecoderBlock(2048, 1024),
            AttentionGate(1024, 1024, 512),
            DecoderBlock(1024, 512),
            AttentionGate(512, 512, 256),
            DecoderBlock(512, 256),
            AttentionGate(256, 256, 128),
            DecoderBlock(256, 128),
            AttentionGate(128, 128, 64),
            DecoderBlock(128, 64),
        ])
        return decoder

    def forward(self, x):
        x = self.encoder(x)
        x = x[1:]
        # 将x逆序
        x = x[::-1]
        for i, module in enumerate(self.decoder):
            if isinstance(module, DecoderBlock):  # isinstance是判断module是否是DecoderBlock的实例
                x[i] = module(x[i])
            elif isinstance(module, AttentionGate):
                x[i] = module(x[i], x[i-1])
        mask = torch.sigmoid(x[-1])
        cls = self.linear(x[-1].view(x[-1].size(0), -1))
        return mask, cls


if __name__ == '__main__':
    model = UNet()
    print(model)
    x = torch.randn(1, 3, 256, 256)
    mask, cls = model(x)
    print(mask.shape)
    print(cls.shape)