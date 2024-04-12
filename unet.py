""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DemodulatedConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=0, bias=False, dilation=1):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channel))

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, input):
        batch, in_channel, height, width = input.shape

        demod = torch.rsqrt(self.weight.pow(2).sum([2, 3, 4]) + 1e-8)
        
        weight = self.weight * demod.view(1, self.out_channel, 1, 1, 1)

        weight = weight.view(self.out_channel, in_channel, self.kernel_size, self.kernel_size)

        if self.bias is None:
            out = F.conv2d(input, weight, padding=self.padding, dilation=self.dilation, stride=self.stride)
        else:
            out = F.conv2d(input, weight, bias=self.bias, padding=self.padding, dilation=self.dilation, stride=self.stride)

        return out

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, demodulated, mid_channels=None, norm_type="batch"):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if norm_type == "batch":
            norm_layer1 = nn.BatchNorm2d(mid_channels)
            norm_layer2 = nn.BatchNorm2d(out_channels)
        elif norm_type == "instance":
            norm_layer1 = nn.InstanceNorm2d(mid_channels, momentum=0.001)
            norm_layer2 = nn.InstanceNorm2d(out_channels, momentum=0.001)
        else:
            norm_layer1 = nn.Identity()
            norm_layer2 = nn.Identity()
        if demodulated:
            conv1 = DemodulatedConv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
            conv2 = DemodulatedConv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        else:
            conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
            conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.double_conv = nn.Sequential(
            conv1,
            norm_layer1,
            nn.ReLU(inplace=True),
            conv2,
            norm_layer2,
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, norm_type, demodulated):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, norm_type=norm_type, demodulated=demodulated)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, demodulated, bilinear=True, norm_type="batch"):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, mid_channels=in_channels // 2, norm_type=norm_type, demodulated=demodulated)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, demodulated=demodulated)

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
    def __init__(self, in_channels, out_channels, demodulated):
        super(OutConv, self).__init__()
        if demodulated:
            self.conv = DemodulatedConv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
""" Full assembly of the parts to form the complete network """

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, mid_channels=[64,128,256,512,1024], bilinear=True, kl_reg=False, norm_type="batch", demodulated=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.kl_reg = kl_reg

        self.inc = (DoubleConv(n_channels, mid_channels[0], norm_type=norm_type, demodulated=demodulated))

        self.downs = []

        for ch_i in range(len(mid_channels)-2):
            self.downs.append(Down(mid_channels[ch_i], mid_channels[ch_i+1], norm_type=norm_type, demodulated=demodulated))
        factor = 2 if bilinear else 1
        self.downs.append(Down(mid_channels[len(mid_channels)-2], mid_channels[len(mid_channels)-1] // factor, norm_type=norm_type, demodulated=demodulated))
        self.downs = nn.ModuleList(self.downs)

        self.ups = []
        for ch_i in range(len(mid_channels)-1, 1, -1):
            self.ups.append(Up(mid_channels[ch_i], mid_channels[ch_i-1] // factor, bilinear=bilinear, norm_type=norm_type, demodulated=demodulated))
        self.ups.append(Up(mid_channels[1], mid_channels[0], bilinear=bilinear, norm_type=norm_type, demodulated=demodulated))
        self.ups = nn.ModuleList(self.ups)

        self.outc = (OutConv(mid_channels[0], n_classes, demodulated=demodulated))

    def reparameterization(self, mu):
        z = torch.tensor(np.random.normal(0, 1, mu.shape), dtype=torch.float32, device = mu.device)
        return z + mu

    def forward(self, x):
        
        x = self.inc(x)
        x_downs = []
        for down in self.downs:
            x_downs.append(x)
            x = down(x)
        for up, x_down in zip(self.ups, x_downs[::-1]):
            x = up(x, x_down)
        x = self.outc(x)

        if self.kl_reg:
            mu = x
            x = self.reparameterization(mu)
            return mu, x
        else:
            return x
