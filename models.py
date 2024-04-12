import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.init as init
import functools
import math
import numbers
from skimage import io
from math import log2, sqrt

def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun

class LS_Discriminator(nn.Module):
    def __init__(self, channels=1, img_size=128):
        super(LS_Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2 ** 4
        self.adv_layer = nn.Linear(128 * ds_size ** 2, 1)

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


class MultiDiscriminator(nn.Module):
    def __init__(self, channels=1, num_scales=3, downsample_stride=2):
        super(MultiDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # Extracts three discriminator models
        self.models = nn.ModuleList()
        for i in range(num_scales):
            self.models.add_module(
                "disc_%d" % i,
                nn.Sequential(
                    *discriminator_block(channels, 64, normalize=False),
                    *discriminator_block(64, 128),
                    *discriminator_block(128, 256),
                    *discriminator_block(256, 512),
                    nn.Conv2d(512, 1, 3, padding=1)
                ),
            )

        self.downsample = nn.AvgPool2d(3, stride=downsample_stride, padding=[1, 1], 
                                       count_include_pad=False)

    def forward(self, x):
        outputs = []
        # cnt = 0
        for m in self.models:
            # print("Discriminator count: %d. Image shape is: %s." % (cnt, str(x.shape)))
            # out_save = x[0].cpu().float().detach().numpy()        
            # io.imsave(f"out_{cnt}.tif", out_save, metadata={'axes': 'TYX'})
            outputs.append(m(x))
            x = self.downsample(x)  # Downsample resolution
            # cnt += 1
        return outputs

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim=256, n_blk=3, activ="relu"):
        super(MLP, self).__init__()
        layers = [nn.Linear(input_dim, dim), nn.ReLU(inplace=True)]
        for _ in range(n_blk - 2):
            layers += [nn.Linear(dim, dim), nn.ReLU(inplace=True)]
        layers += [nn.Linear(dim, output_dim)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))

class Unet(nn.Module):
    ''' kernel_size = 3, stride = 1 <-
    '''
    def __init__(self, input_channels = 1, output_channels = 1, upsampling_method = "InterpolationConv", std = 0, stretch=0, temperature=6, norm=True, conv_type="Conv2d"):
        super(Unet, self).__init__()
        
        self.temperature = temperature
        self.stretch = stretch
        self.std = std
        # self.sigmoid = sigmoid
        self.upsampling_method = upsampling_method
        self.conv_type = conv_type
        self.norm = norm

        def Upsample(in_channels, out_channels, kernel_size, bias, stride=1, padding=0):
            if self.upsampling_method == "ConvTranspose":
                return nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                          kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
            elif self.upsampling_method == "InterpolationConv":
                layers = []
                layers += [nn.Upsample(scale_factor=2, mode='nearest')]
                layers += [nn.ReflectionPad2d(1)]
                layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=3, stride=1, padding=0, bias=bias)]
                return nn.Sequential(*layers)

        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            if self.conv_type == "Conv2d":
                layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size, stride=stride, padding=padding,
                                bias=bias)]
                if self.norm:
                    layers += [nn.BatchNorm2d(num_features=out_channels)]
            elif self.conv_type == "DemodulatedConv2d":
                layers += [DemodulatedConv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size, stride=stride, padding=padding,
                                bias=bias)]
            layers += [nn.ReLU()]
            cbr = nn.Sequential(*layers)
            return cbr

        # def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        #     layers = []
        #     layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
        #                          kernel_size=kernel_size, stride=stride, padding=padding,
        #                          bias=bias)]
        #     layers += [nn.BatchNorm2d(num_features=out_channels)]
        #     layers += [nn.ReLU()]
        #     cbr = nn.Sequential(*layers)
        #     return cbr
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        # Contracting path
        self.enc1_1 = CBR2d(in_channels=self.input_channels, out_channels=16)
        self.enc1_2 = CBR2d(in_channels=16, out_channels=16)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.enc2_1 = CBR2d(in_channels=16, out_channels=32)
        self.enc2_2 = CBR2d(in_channels=32, out_channels=32)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.enc3_1 = CBR2d(in_channels=32, out_channels=64)
        self.enc3_2 = CBR2d(in_channels=64, out_channels=64)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.enc4_1 = CBR2d(in_channels=64, out_channels=128)
        self.enc4_2 = CBR2d(in_channels=128, out_channels=128)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        self.enc5_1 = CBR2d(in_channels=128, out_channels=256)
        # Expansive path
        self.dec5_1 = CBR2d(in_channels=256, out_channels=128)
        self.unpool4 = Upsample(in_channels=128, out_channels=128,
                                          kernel_size=2, stride=2, padding=0, bias=True)
        # self.unpool4 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
        #                                   kernel_size=2, stride=2, padding=0, bias=True)
        self.dec4_2 = CBR2d(in_channels=2 * 128, out_channels=128)
        self.dec4_1 = CBR2d(in_channels=128, out_channels=64)
        self.unpool3 = Upsample(in_channels=64, out_channels=64,
                                          kernel_size=2, stride=2, padding=0, bias=True)
        # self.unpool3 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
        #                                   kernel_size=2, stride=2, padding=0, bias=True)
        self.dec3_2 = CBR2d(in_channels=2 * 64, out_channels=64)
        self.dec3_1 = CBR2d(in_channels=64, out_channels=32)
        self.unpool2 = Upsample(in_channels=32, out_channels=32,
                                          kernel_size=2, stride=2, padding=0, bias=True)
        # self.unpool2 = nn.ConvTranspose2d(in_channels=32, out_channels=32,
        #                                   kernel_size=2, stride=2, padding=0, bias=True)
        self.dec2_2 = CBR2d(in_channels=2 * 32, out_channels=32)
        self.dec2_1 = CBR2d(in_channels=32, out_channels=16)
        self.unpool1 = Upsample(in_channels=16, out_channels=16,
                                          kernel_size=2, stride=2, padding=0, bias=True)
        # self.unpool1 = nn.ConvTranspose2d(in_channels=16, out_channels=16,
        #                                   kernel_size=2, stride=2, padding=0, bias=True)
        self.dec1_2 = CBR2d(in_channels=2 * 16, out_channels=16)
        self.dec1_1 = CBR2d(in_channels=16, out_channels=16)
        # self.fc1 = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc2 = nn.Conv2d(in_channels=16, out_channels=self.output_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)
        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)
        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)
        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)
        enc5_1 = self.enc5_1(pool4)
        dec5_1 = self.dec5_1(enc5_1)
        # print(dec5_1.shape)
        unpool4 = self.unpool4(dec5_1)
        # print(unpool4.shape)
        # exit()
        cat4 = torch.cat((unpool4, enc4_2), dim=1)
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)
        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc3_2), dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)
        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)
        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        # weight map
        pred_w = self.fc2(dec1_1)
        # pred_w = self.fc2(dec2_1)
        # interpolate w
        # pred_w = F.interpolate(pred_w, scale_factor=2, mode='nearest')

        # print(pred_w.shape)
        # exit()

        if self.std == 1:
            eps = 1e-8
            pred_w = (pred_w - torch.mean(pred_w).detach()) / torch.sqrt(torch.var(pred_w).detach() + eps)
        # # if self.sigmoid == 1:
        # #     pred_w = torch.sigmoid(pred_w)
        #     # print("sigmoid")

        # if self.stretch == 1:
        #     # if self.std == 1:
        #     #     pred_w = (pred_w - torch.mean(pred_w).detach()) / torch.sqrt(torch.var(pred_w).detach())
        #     pred_w = torch.sigmoid(pred_w * self.temperature)
        return pred_w


class DemodulatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False, dilation=1):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channels
        self.out_channel = out_channels

        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, input):
        batch, in_channel, height, width = input.shape

        demod = torch.rsqrt(self.weight.pow(2).sum([2, 3, 4]) + 1e-8)
        weight = self.weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        input = input.view(1, batch * in_channel, height, width)
        if self.bias is None:
            out = F.conv2d(input, weight, padding=self.padding, groups=batch, dilation=self.dilation, stride=self.stride)
        else:
            out = F.conv2d(input, weight, bias=self.bias, padding=self.padding, groups=batch, dilation=self.dilation, stride=self.stride)
        _, _, height, width = out.shape
        out = out.view(batch, self.out_channel, height, width)

        return out


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from math import sqrt

# class EqualizedWeight(nn.Module):
#     def __init__(self, shape):
#         super().__init__()
#         self.c = 1 / sqrt(sum(shape[1:]))
#         self.weight = nn.Parameter(torch.randn(shape))

#     def forward(self):
#         return self.weight * self.c

# class DemodulatedConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, demodulate=True, eps=1e-8):
#         super(DemodulatedConv2d, self).__init__()
#         self.demodulate = demodulate
#         self.eps = eps

#         # Create an equalized weight
#         self.weight = EqualizedWeight([out_channels, in_channels, kernel_size, kernel_size])
#         self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
#         self.stride = stride
#         self.padding = padding
#         self.dilation = dilation
#         self.groups = groups

#     def forward(self, x):
#         weights = self.weight()[None, :, :, :, :]

#         if self.demodulate:
#             # Demodulate the weights
#             sigma_inv = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
#             weights = weights * sigma_inv

#         return F.conv2d(x, weights[0], self.bias, self.stride, self.padding, self.dilation, self.groups)




# class EqualizedWeight(nn.Module):

#     def __init__(self, shape):

#         super().__init__()

#         self.c = 1 / sqrt(np.prod(shape[1:]))
#         self.weight = nn.Parameter(torch.randn(shape))

#     def forward(self):
#         return self.weight * self.c

# class Conv2dWeightModulate(nn.Module):

#     def __init__(self, in_features, out_features, kernel_size, demodulate = True, eps = 1e-8):

#         super().__init__()
#         self.out_features = out_features
#         self.demodulate = demodulate
#         self.padding = (kernel_size - 1) // 2

#         self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size])
#         self.eps = eps

#     def forward(self, x, s):

#         b, _, h, w = x.shape

#         s = s[:, None, :, None, None]
#         weights = self.weight()[None, :, :, :, :]
#         weights = weights * s

#         if self.demodulate:
#             sigma_inv = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
#             weights = weights * sigma_inv

#         x = x.reshape(1, -1, h, w)

#         _, _, *ws = weights.shape
#         weights = weights.reshape(b * self.out_features, *ws)

#         x = F.conv2d(x, weights, padding=self.padding, groups=b)

#         return x.reshape(-1, self.out_features, h, w)










# class Unet(nn.Module):
#     ''' kernel_size = 3, stride = 1 <-
#     '''
#     def __init__(self, input_channels = 1, output_channels = 1, upsampling_method = "InterpolationConv", std = 0, stretch=0, temperature=6, norm=True):
#         super(Unet, self).__init__()
        
#         self.temperature = temperature
#         self.stretch = stretch
#         self.std = std
#         # self.sigmoid = sigmoid
#         self.upsampling_method = upsampling_method
#         self.norm = norm

#         def Upsample(in_channels, out_channels, kernel_size, bias, stride=1, padding=0):
#             if self.upsampling_method == "ConvTranspose":
#                 return nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
#                                           kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
#             elif self.upsampling_method == "InterpolationConv":
#                 layers = []
#                 layers += [nn.Upsample(scale_factor=2, mode='nearest')]
#                 layers += [nn.ReflectionPad2d(1)]
#                 layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
#                                      kernel_size=3, stride=1, padding=0, bias=bias)]
#                 return nn.Sequential(*layers)

#         def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
#             layers = []
#             layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
#                                  kernel_size=kernel_size, stride=stride, padding=padding,
#                                  bias=bias)]
#             if self.norm:
#                 layers += [nn.BatchNorm2d(num_features=out_channels)]
#             layers += [nn.ReLU()]
#             cbr = nn.Sequential(*layers)
#             return cbr
        
#         self.input_channels = input_channels
#         self.output_channels = output_channels
#         # Contracting path
#         self.enc1_1 = CBR2d(in_channels=self.input_channels, out_channels=16)
#         self.enc1_2 = CBR2d(in_channels=16, out_channels=16)
#         self.pool1 = nn.MaxPool2d(kernel_size=2)
#         self.enc2_1 = CBR2d(in_channels=16, out_channels=32)
#         self.enc2_2 = CBR2d(in_channels=32, out_channels=32)
#         self.pool2 = nn.MaxPool2d(kernel_size=2)
#         self.enc3_1 = CBR2d(in_channels=32, out_channels=64)
#         self.enc3_2 = CBR2d(in_channels=64, out_channels=64)
#         self.pool3 = nn.MaxPool2d(kernel_size=2)
#         self.enc4_1 = CBR2d(in_channels=64, out_channels=128)
#         self.enc4_2 = CBR2d(in_channels=128, out_channels=128)
#         self.pool4 = nn.MaxPool2d(kernel_size=2)
#         self.enc5_1 = CBR2d(in_channels=128, out_channels=256)
#         # Expansive path
#         self.dec5_1 = CBR2d(in_channels=256, out_channels=128)

#         self.unpool4 = Upsample(in_channels=128, out_channels=128,
#                                           kernel_size=2, stride=2, padding=0, bias=True)
#         # self.unpool4 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
#         #                                   kernel_size=2, stride=2, padding=0, bias=True)
#         self.dec4_2 = CBR2d(in_channels=2 * 128, out_channels=128)
#         self.dec4_1 = CBR2d(in_channels=128, out_channels=64)
#         self.unpool3 = Upsample(in_channels=64, out_channels=64,
#                                           kernel_size=2, stride=2, padding=0, bias=True)
#         # self.unpool3 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
#         #                                   kernel_size=2, stride=2, padding=0, bias=True)
#         self.dec3_2 = CBR2d(in_channels=2 * 64, out_channels=64)
#         self.dec3_1 = CBR2d(in_channels=64, out_channels=32)
#         self.unpool2 = Upsample(in_channels=32, out_channels=32,
#                                           kernel_size=2, stride=2, padding=0, bias=True)
#         # self.unpool2 = nn.ConvTranspose2d(in_channels=32, out_channels=32,
#         #                                   kernel_size=2, stride=2, padding=0, bias=True)
#         self.dec2_2 = CBR2d(in_channels=2 * 32, out_channels=32)
#         self.dec2_1 = CBR2d(in_channels=32, out_channels=16)
#         self.unpool1 = Upsample(in_channels=16, out_channels=16,
#                                           kernel_size=2, stride=2, padding=0, bias=True)
#         # self.unpool1 = nn.ConvTranspose2d(in_channels=16, out_channels=16,
#         #                                   kernel_size=2, stride=2, padding=0, bias=True)
#         self.dec1_2 = CBR2d(in_channels=2 * 16, out_channels=16)
#         self.dec1_1 = CBR2d(in_channels=16, out_channels=16)
#         # self.fc1 = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=1, stride=1, padding=0, bias=True)
#         self.fc2 = nn.Conv2d(in_channels=16, out_channels=self.output_channels, kernel_size=1, stride=1, padding=0, bias=True)

#     def forward(self, x):
#         enc1_1 = self.enc1_1(x)
#         enc1_2 = self.enc1_2(enc1_1)
#         pool1 = self.pool1(enc1_2)
#         enc2_1 = self.enc2_1(pool1)
#         enc2_2 = self.enc2_2(enc2_1)
#         pool2 = self.pool2(enc2_2)
#         enc3_1 = self.enc3_1(pool2)
#         enc3_2 = self.enc3_2(enc3_1)
#         pool3 = self.pool3(enc3_2)
#         enc4_1 = self.enc4_1(pool3)
#         enc4_2 = self.enc4_2(enc4_1)
#         pool4 = self.pool4(enc4_2)
#         enc5_1 = self.enc5_1(pool4)
#         dec5_1 = self.dec5_1(enc5_1)
#         # print(dec5_1.shape)
#         unpool4 = self.unpool4(dec5_1)
#         # print(unpool4.shape)
#         # exit()
#         cat4 = torch.cat((unpool4, enc4_2), dim=1)
#         dec4_2 = self.dec4_2(cat4)
#         dec4_1 = self.dec4_1(dec4_2)
#         unpool3 = self.unpool3(dec4_1)
#         cat3 = torch.cat((unpool3, enc3_2), dim=1)
#         dec3_2 = self.dec3_2(cat3)
#         dec3_1 = self.dec3_1(dec3_2)
#         unpool2 = self.unpool2(dec3_1)
#         cat2 = torch.cat((unpool2, enc2_2), dim=1)
#         dec2_2 = self.dec2_2(cat2)
#         dec2_1 = self.dec2_1(dec2_2)
#         unpool1 = self.unpool1(dec2_1)
#         cat1 = torch.cat((unpool1, enc1_2), dim=1)
#         dec1_2 = self.dec1_2(cat1)
#         dec1_1 = self.dec1_1(dec1_2)

#         # weight map
#         pred_w = self.fc2(dec1_1)
#         # pred_w = self.fc2(dec2_1)
#         # interpolate w
#         # pred_w = F.interpolate(pred_w, scale_factor=2, mode='nearest')

#         # print(pred_w.shape)
#         # exit()

#         if self.std == 1:
#             eps = 1e-8
#             pred_w = (pred_w - torch.mean(pred_w).detach()) / torch.sqrt(torch.var(pred_w).detach() + eps)
#         # # if self.sigmoid == 1:
#         # #     pred_w = torch.sigmoid(pred_w)
#         #     # print("sigmoid")

#         # if self.stretch == 1:
#         #     # if self.std == 1:
#         #     #     pred_w = (pred_w - torch.mean(pred_w).detach()) / torch.sqrt(torch.var(pred_w).detach())
#         #     pred_w = torch.sigmoid(pred_w * self.temperature)
#         return pred_w
