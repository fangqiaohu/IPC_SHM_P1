""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SelfAdaption(nn.Module):
    """self-adaption, e.g: bs*3*28*28 -> bs*(3*4)*28*28 -> bs*(3*4)*28*28 -> bs*3*28*28, kernel_size=1"""
    # https://discuss.pytorch.org/t/how-to-modify-a-conv2d-to-depthwise-separable-convolution/15843

    def __init__(self, in_channels):
        super(_SelfAdaption, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = 1
        self.padding = 0
        self.stride = 1
        self.hidden_channels = 4

        self.add_module('relu1', nn.ReLU())
        self.add_module('convs1', nn.Conv2d(self.in_channels, self.hidden_channels * self.in_channels,
                                            kernel_size=self.kernel_size, groups=self.in_channels,
                                            stride=self.stride, padding=self.padding))

        # self.add_module('relu2', nn.ReLU())
        # self.add_module('convs2', nn.Conv2d(self.hidden_channels * self.in_channels, self.hidden_channels * self.in_channels,
        #                                     kernel_size=self.kernel_size, groups=self.in_channels,
        #                                     stride=self.stride, padding=self.padding))

        self.add_module('relu3', nn.ReLU())
        self.add_module('convs3', nn.Conv2d(self.hidden_channels * self.in_channels, self.in_channels,
                                            kernel_size=self.kernel_size, groups=self.in_channels,
                                            stride=self.stride, padding=self.padding))

    def forward(self, x):
        x = self.convs1(self.relu1(x))
        # x = self.convs2(self.relu2(x))
        x = self.convs3(self.relu3(x))
        return x


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.lambda1 = 1

        # DEBUG: adding SASA
        self.adp = _SelfAdaption(in_channels=out_channels)

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # self.double_conv1 = nn.Sequential(
        #     nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(mid_channels),
        #     nn.ReLU(inplace=True)
        # )
        #
        # self.double_conv2 = nn.Sequential(
        #     nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(inplace=True)
        # )

    def forward(self, x):
        # x = self.double_conv1(x)
        # # x = x + self.lambda1 * self.adp1(x)
        # x = self.double_conv2(x)
        x = self.double_conv(x)
        y = self.lambda1 * self.adp(x)
        return x + y


# class DoubleConv(nn.Module):
#     """(convolution => [BN] => ReLU) * 2"""

#     def __init__(self, in_channels, out_channels, mid_channels=None):
#         super().__init__()
#         if not mid_channels:
#             mid_channels = out_channels

#         self.double_conv = nn.Sequential(
#             nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(mid_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.double_conv(x)


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

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
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


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
