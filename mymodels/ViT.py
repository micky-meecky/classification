import torch
from vit_pytorch import ViT
import torch.nn as nn


class ViT_model(nn.Module):
    def __init__(self, image_size=224,
                 patch_size=16,
                 num_classes=3):
        super(ViT_model, self).__init__()
        self.v = ViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=768,
            depth=12,
            heads=12,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )

    def forward(self, x):
        return self.v(x)


class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0):
        super(UpSampleBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        # self.bn = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        return self.relu2(x)


class UNetDecoder(nn.Module):
    def __init__(self):
        super(UNetDecoder, self).__init__()

        self.merge_conv = nn.Conv2d(1024 * 2, 1024, kernel_size=1)

        # Corresponding to layer3 output
        self.up1 = UpSampleBlock(1024, 512, skip_channels=512)

        # Corresponding to layer2 output
        self.up2 = UpSampleBlock(512, 256, skip_channels=256)

        # Corresponding to layer1 output
        self.up3 = UpSampleBlock(256, 128, skip_channels=128)

        # upsample to match original input size
        self.up4 = UpSampleBlock(128, 64)

        # upsample to match original input size
        # self.up5 = UpSampleBlock(64, 32)

        # Segmentation head
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, features):
        # 获取bs
        bs = features[0].shape[0]
        z3, z6, z9, z12 = features
        x = self.up1(z12, z9)  # Use layer3 output and skip from layer2
        x = self.up2(x, z6)  # Use the result and skip from layer1
        x = self.up3(x, z3)  # Use the result
        x = self.up4(x)
        segout = self.segmentation_head(x)
        return segout


class SingleDeconv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SingleDeconv2DBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = nn.ConvTranspose2d(self.in_channels, self.out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.up(x)
        return x


class Deconv2DBlock(nn.Module):
    r"""
    Deconvolutional block as in "U-Net: Convolutional Networks for Biomedical Image Segmentation" (https://arxiv.org/abs/1505.04597)

    Args:
        in_planes: Number of input channels
        out_planes: Number of output channels
        kernel_size: Kernel size of the convolutional layers
    """
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleDeconv2DBlock(in_planes, out_planes),
            # SingleConv2DBlock(out_planes, out_planes, kernel_size),
            nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size, stride=1, padding=((kernel_size - 1) // 2)),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class ViTseg(nn.Module):
    def __init__(self):
        super(ViTseg, self).__init__()
        self.encoder = ViT_model(image_size=224,
                                 patch_size=16,
                                 num_classes=1)
        self.decoder = UNetDecoder()
        self.embed_dim = 768
        self.patch_size = 16
        self.img_shape = (224, 224)
        self.patch_dim = [int(x / self.patch_size) for x in self.img_shape]
        self.extract_layers = [3, 6, 9, 12]
        self.deconv12 = nn.Sequential(SingleDeconv2DBlock(self.embed_dim, 512),
                                      nn.Conv2d(512, 1024, kernel_size=3, padding=1),
                                      nn.MaxPool2d(2),
                                      # pooling layer
                                        nn.BatchNorm2d(1024),
                                        nn.ReLU(inplace=True),
                                      )
        self.decoder9 = Deconv2DBlock(self.embed_dim, 512)
        self.decoder6 = nn.Sequential(Deconv2DBlock(self.embed_dim, 512), Deconv2DBlock(512, 256), )
        self.decoder3 = nn.Sequential(Deconv2DBlock(self.embed_dim, 512), Deconv2DBlock(512, 256),
                                      Deconv2DBlock(256, 128), )

    def forward(self, x):
        cls, encoder_output = self.encoder(x)
        # 根据extract_layers获取对应的encoder_output
        features = [encoder_output[layer - 1] for layer in self.extract_layers]
        features = [features[i][:, 1:, :] for i in range(len(features))]
        features[3] = features[3].transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        features[2] = features[2].transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        features[1] = features[1].transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        features[0] = features[0].transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        features[3] = self.deconv12(features[3])
        features[2] = self.decoder9(features[2])
        features[1] = self.decoder6(features[1])
        features[0] = self.decoder3(features[0])
        seg = self.decoder(features)
        return cls, seg


if __name__ == "__main__":
    img = torch.randn(3, 3, 224, 224)

    model = ViTseg()

    preds, seg = model(img)  # (1, 1000)

    print(preds.shape)
    print(seg.shape)
    # 输出类别
    print(torch.argmax(preds))
