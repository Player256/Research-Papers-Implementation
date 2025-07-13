import torch
import torch.nn as nn


class SRResNetInputLayer(nn.Module):
    def __init__(self):
        super(SRResNetInputLayer, self).__init__()
        self.conv2d = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=9, stride=1, padding=4
        )
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv2d(x)
        x = self.prelu(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        return out


class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(64)

    def forward(self, x, residual):
        x = self.conv(x)
        x = self.bn(x)
        x = x + residual  
        return x


class UpSampler(nn.Module):
    def __init__(self):
        super(UpSampler, self).__init__()
        self.conv = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


class SRResNet(nn.Module):
    def __init__(self, num_residual_blocks=16):
        super(SRResNet, self).__init__()
        self.input_layer = SRResNetInputLayer()
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock() for _ in range(num_residual_blocks)]
        )
        self.conv_block = ConvBlock()
        self.upsampler = nn.Sequential(UpSampler(), UpSampler())
        self.output_layer = nn.Conv2d(
            in_channels=64, out_channels=3, kernel_size=9, stride=1, padding=4
        )

    def forward(self, x):
        x = self.input_layer(x)
        residual = x  
        x = self.residual_blocks(x)
        x = self.conv_block(x, residual)
        x = self.upsampler(x)
        x = self.output_layer(x)
        return x
