"""

TODO List:
1. Weight initialization (same)
2. (Optional) Lambda learning rate scheduler
3. Residual Block (same)
4. Encoder (Output feature shape = Input image shape)
5. Decoder/Generator (Input feature shape = Output image shape) --> Test if pure resnet works or copy the encoder
6. Discriminator (the modified version with more options)

"""

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.init as init
import functools

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

"""
Encoder
Input: Image (C x W x H)
Output: Feature (C x W x H)
Layers:
Initial ConvBlock: (1, 512, 512) --> (64, 512, 512)
Downsample: (64, 512, 512) --> (256, 128, 128)
Resblocks: (256, 128, 128) --> (256, 128, 128)
Upsample: (256, 128, 128) --> (1, 512, 512)
"""

class Encoder(nn.Module):
    def __init__(self, in_channels=1, feat_channels=1, dim=64, n_residual=3, n_downsample=2, output_activation = "tanh"):
        super(Encoder, self).__init__()

        # Initial convolution block
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, dim, 7),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
        ]

        # Downsampling
        for _ in range(n_downsample):
            layers += [
                nn.Conv2d(dim, dim * 2, 4, stride=2, padding=1),
                nn.InstanceNorm2d(dim * 2),
                nn.ReLU(inplace=True),
            ]
            dim *= 2

        # Residual blocks
        for _ in range(n_residual):
            layers += [ResidualBlock(dim)]

        # dim = 256 at this point: dim = dim * 2 ** n_downsample

        # Upsampling
        for _ in range(n_downsample):
            layers += [
                nn.ConvTranspose2d(dim, dim // 2, 4, stride=2, padding=1),
                nn.InstanceNorm2d(dim // 2),
                nn.ReLU(inplace=True),
            ]
            dim = dim // 2

        # Output layer
        if output_activation == "tanh":
            layers += [nn.ReflectionPad2d(3), nn.Conv2d(dim, feat_channels, 7), nn.Tanh()]
        elif output_activation == "relu":
            layers += [nn.ReflectionPad2d(3), nn.Conv2d(dim, feat_channels, 7), nn.ReLU()]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    

"""

Decoder (Generator)
Input: Feature (C x W x H)
Output: Image (C x W x H)
Layers:
Downsample
Resblocks: (C x W x H) --> (C x W x H)
Upsample

"""

# class Decoder(nn.Module):
#     def __init__(self, dim_residual=1, n_residual=3):
#         super(Decoder, self).__init__()

#         layers = []
#         # Residual blocks
#         for _ in range(n_residual):
#             layers += [ResidualBlock(dim_residual)]

#         self.model = nn.Sequential(*layers)

#     def forward(self, x):
#         x = self.model(x)
#         return x

class Decoder(nn.Module):
    def __init__(self, feat_channels=1, out_channels=1, dim=64, n_residual=3, n_downsample=2, output_activation = "tanh"):
        super(Decoder, self).__init__()

        # Initial convolution block
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(feat_channels, dim, 7),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
        ]

        # Downsampling
        for _ in range(n_downsample):
            layers += [
                nn.Conv2d(dim, dim * 2, 4, stride=2, padding=1),
                nn.InstanceNorm2d(dim * 2),
                nn.ReLU(inplace=True),
            ]
            dim *= 2

        # Residual blocks
        for _ in range(n_residual):
            layers += [ResidualBlock(dim)]

        # dim = 256 at this point: dim = dim * 2 ** n_downsample

        # Upsampling
        for _ in range(n_downsample):
            layers += [
                nn.ConvTranspose2d(dim, dim // 2, 4, stride=2, padding=1),
                nn.InstanceNorm2d(dim // 2),
                nn.ReLU(inplace=True),
            ]
            dim = dim // 2

        # Output layer
        if output_activation == "tanh":
            layers += [nn.ReflectionPad2d(3), nn.Conv2d(dim, out_channels, 7), nn.Tanh()]
        elif output_activation == "relu":
            layers += [nn.ReflectionPad2d(3), nn.Conv2d(dim, out_channels, 7), nn.ReLU()]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

"""

Residual Block
Input: Tensor (C x W x H)
Output: Tensor (C x W x H)

"""
class ResidualBlock(nn.Module):
    def __init__(self, features):
        super(ResidualBlock, self).__init__()

        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, 3),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, 3),
            nn.InstanceNorm2d(features),
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        # print("Shape of input is: ", x.shape)
        # print("Shape of res is: ", self.conv_block(x).shape)
        return x + self.conv_block(x)
    

"""

Discriminator
Input: Image (C x W x H)
Output: 

"""

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

    # def compute_loss(self, x, gt):
    #     """Computes the MSE between model output and scalar gt"""
    #     # loss = sum([torch.mean((out - gt) ** 2) for out in self.forward(x)])
    #     loss = 0
    #     n=0
    #     for out in self.forward(x):
    #         squared_diff = (out - gt) ** 2
    #         loss += torch.mean(squared_diff)
    #         # print(f"{n}: Mean squared_diff: {torch.mean(squared_diff)} and current loss: {loss}")
    #         # print(f"{n}: Out shape: {out.shape} ")
    #         n=n+1
    #     return loss

    def forward(self, x):
        outputs = []
        # cnt = 0
        for m in self.models:
            # print("Discriminator count: %d. Image shape is: %s." % (cnt, str(x.shape)))
            outputs.append(m(x))
            x = self.downsample(x)  # Downsample resolution
            # cnt += 1
        return outputs
    


"""

Multi-scale Multi-class Discriminator
Input: 3D tensor (C x W x H)
Output: 1D Class

"""  

class MultiScaleMultiClassDiscriminator(nn.Module):
    def __init__(self, channels=1, num_classes=4, num_scales=3, downsample_stride=2, last_layer='adaptAvgPool'):
        super(MultiScaleMultiClassDiscriminator, self).__init__()

        self.last_layer = last_layer

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.models = nn.ModuleList()
        if self.last_layer == 'adaptAvgPool':
            for i in range(num_scales):
                self.models.add_module(
                    f"disc_{i}",
                    nn.Sequential(
                        *discriminator_block(channels, 64, normalize=False),
                        *discriminator_block(64, 128),
                        *discriminator_block(128, 256),
                        *discriminator_block(256, 512),
                        nn.Conv2d(512, num_classes, 3, padding=1),  # Output is num_classes probabilities
                        nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
                        nn.Flatten(),  # Flatten the output
                        # nn.LogSoftmax(dim=1)
                        # nn.Softmax(dim=1),  # Apply softmax to output probabilities
                    )
                )
        elif self.last_layer == 'linear':
            for i in range(num_scales):
                self.models.add_module(
                    f"disc_{i}",
                    nn.Sequential(
                        *discriminator_block(channels, 64, normalize=False),
                        *discriminator_block(64, 128),
                        *discriminator_block(128, 256),
                        *discriminator_block(256, 512),
                        nn.Conv2d(512, num_classes, 3, padding=1),  # Output is num_classes probabilities
                        # nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
                        nn.Flatten(),  # Flatten the output
                        # nn.LogSoftmax(dim=1)
                        # nn.Softmax(dim=1),  # Apply softmax to output probabilities
                    )
                )

        self.downsample = nn.AvgPool2d(3, stride=downsample_stride, padding=[1, 1], count_include_pad=False)

    def forward(self, input):
        outputs = []
        if self.last_layer == 'adaptAvgPool':
            for m in self.models:
                outputs.append(m(input))
                input = self.downsample(input)  # Downsample resolution
        elif self.last_layer == 'linear':
            for m in self.models:
                # TODO: need to test the following shape outputs for both rgb and grayscale
                size = self.num_class * input.shape[1] * input.shape[2] * input.shape[3]
                fc = nn.Linear(size, self.num_classes)
                outputs.append(fc(m(input)))
                input = self.downsample(input)  # Downsample resolution
        return outputs
    

# Not using adaptive avg pooling
# class MultiScaleMultiClassDiscriminator(nn.Module):
#     def __init__(self, channels=1, num_classes=4, num_scales=3, downsample_stride=2):
#         super(MultiScaleMultiClassDiscriminator, self).__init__()

#         self.num_class = num_classes

#         def discriminator_block(in_filters, out_filters, normalize=True):
#             """Returns downsampling layers of each discriminator block"""
#             layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
#             if normalize:
#                 layers.append(nn.InstanceNorm2d(out_filters))
#             layers.append(nn.LeakyReLU(0.2, inplace=True))
#             return layers
        
#         self.models = nn.ModuleList()
#         for i in range(num_scales):
#             self.models.add_module(
#                 f"disc_{i}",
#                 nn.Sequential(
#                     *discriminator_block(channels, 64, normalize=False),
#                     *discriminator_block(64, 128),
#                     *discriminator_block(128, 256),
#                     *discriminator_block(256, 512),
#                     nn.Conv2d(512, num_classes, 3, padding=1),  # Output is num_classes probabilities
#                     # nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
#                     nn.Flatten(),  # Flatten the output
#                     # nn.LogSoftmax(dim=1)
#                     # nn.Softmax(dim=1),  # Apply softmax to output probabilities
#                 )
#             )

#         self.downsample = nn.AvgPool2d(3, stride=downsample_stride, padding=[1, 1], count_include_pad=False)

#     def forward(self, input):
#         outputs = []
#         for m in self.models:
#             # TODO: need to test the following shape outputs for both rgb and grayscale
#             size = self.num_class * input.shape[1] * input.shape[2] * input.shape[3]
#             fc = nn.Linear(size, self.num_classes)
#             outputs.append(fc(m(input)))
#             input = self.downsample(input)  # Downsample resolution
#         return outputs


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)