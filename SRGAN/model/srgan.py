import torch
import torch.nn as nn
from model.srresnet import SRResNet


class InitialConvLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            stride=1,
            kernel_size=3,
            padding=1,
        )
        self.l_relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.l_relu(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            kernel_size=3,
            padding=1,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.l_relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.bn(x)
        x = self.l_relu(x)
        return x


class Discriminator(nn.Module):
    hyperparameters = [
        [64, 64, 2],
        [64, 128, 1],
        [128, 128, 2],
        [128, 256, 1],
        [256, 256, 2],
        [256, 512, 1],
        [512, 512, 2],
    ]

    def __init__(self):
        super().__init__()
        self.initial_conv_layer = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True)
        )
        blocks = []
        for in_ch, out_ch, stride in self.hyperparameters:
            blocks.append(ConvBlock(in_ch, out_ch, stride))
        self.conv_blocks = nn.Sequential(*blocks)
        self.flatten_dim = 512 * 6 * 6
        self.fc = nn.Linear(self.flatten_dim, 1024)
        self.l_relu = nn.LeakyReLU(0.2, inplace=True)
        self.fc_final = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.initial_conv_layer(x)
        x = self.conv_blocks(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.l_relu(x)
        x = self.fc_final(x)
        x = self.sigmoid(x)
        return x


class SRGAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.generator = SRResNet()
        self.discriminator = Discriminator()

    def forward(self, x):
        sr_image = self.generator(x)
        disc_output = self.discriminator(sr_image)
        return sr_image, disc_output
