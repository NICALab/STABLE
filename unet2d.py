import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResidualBlock(nn.Module):
    def __init__(self, features, norm_type):
        super(ResidualBlock, self).__init__()

        if norm_type == 'batch':
            norm_layer = nn.BatchNorm2d(features)
        elif norm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(features)
        else:
            norm_layer = nn.Identity()

        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, 3),
            norm_layer,
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, 3),
            norm_layer,
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class DemodulatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(DemodulatedConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

    def forward(self, x):
        # This is a simplified placeholder for the actual demodulation process
        weight = self.conv.weight
        demodulated_weight = weight / torch.sqrt(torch.sum(weight ** 2, dim=[1, 2, 3], keepdim=True) + 1e-8)
        return F.conv2d(x, demodulated_weight, self.conv.bias, self.conv.stride, self.conv.padding)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, conv_method='conv2d', norm_type=None, n_conv=2):
        super(ConvBlock, self).__init__()
        layers = []
        for _ in range(n_conv):
            if conv_method == 'conv2d':
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
                if norm_type == 'batch':
                    layers.append(nn.BatchNorm2d(out_channels))
                elif norm_type == 'instance':
                    layers.append(nn.InstanceNorm2d(out_channels))
            elif conv_method == 'demodulatedconv':
                layers.append(DemodulatedConv2d(in_channels, out_channels, kernel_size, stride, padding))
            layers.append(nn.ReLU())
            in_channels = out_channels  # For the next layer, if n_conv > 1

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class DownConvBlock(ConvBlock):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, conv_method='conv2d', downsample_scale=2, norm_type=None, n_conv=2):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, conv_method, norm_type, n_conv)
        self.downsample = nn.MaxPool2d(kernel_size=downsample_scale)

    def forward(self, x):
        x = self.downsample(x)
        x = super().forward(x)
        return x

class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, up_method='ConvTranspose', upsample_scale=2, conv_method='conv2d', norm_type=None, n_conv=2):
        super(UpConvBlock, self).__init__()
        if up_method == 'ConvTranspose':
            self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=upsample_scale, stride=upsample_scale)
        elif up_method == 'UpsampleConv':
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=upsample_scale, mode='nearest'),
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            )
        self.conv_block = ConvBlock(out_channels, out_channels, kernel_size, stride, padding, conv_method, norm_type, n_conv)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv_block(x)
        return x

class UNet2D(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, final_channels, kernel_size, stride, padding, norm_type='batch', skip_comb='concat', up_method='ConvTranspose', conv_method='conv2d', n_conv=2, kl_reg=False, mu_dim=64):
        super(UNet2D, self).__init__()
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.skip_combination = skip_comb

        self.init_conv = ConvBlock(in_channels, mid_channels[0], kernel_size, stride, padding, conv_method, norm_type=norm_type, n_conv=n_conv)

        # Creating down blocks
        channels = mid_channels[0]
        for mc in mid_channels[1:]:
            self.down_blocks.append(DownConvBlock(channels, mc, kernel_size, stride, padding, conv_method, norm_type=norm_type, n_conv=n_conv))
            channels = mc
        # Creating up blocks
        for mc in reversed(mid_channels[:-1]):  # Skip last one as it's handled separately
            # print(channels, mc)
            self.up_blocks.append(UpConvBlock(channels, mc, kernel_size, stride, padding, up_method, conv_method=conv_method, norm_type=norm_type, n_conv=n_conv))
            channels = mc * 2 if skip_comb == 'concat' else mc

        # Final up block to match output channels
        self.final_conv = []
        for fc in final_channels:
            self.final_conv.append(ConvBlock(channels, fc, kernel_size, stride, padding, conv_method, norm_type=norm_type, n_conv=n_conv))
            channels = fc
        self.final_conv.append(nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0))
        self.final_conv = nn.Sequential(*self.final_conv)

        self.kl_reg = kl_reg
        if self.kl_reg:
            self.mu_conv = ResidualBlock(mu_dim * 2 ** len(mid_channels[1:]), norm_type=norm_type)

    def reparameterization(self, mu):
        z = torch.tensor(np.random.normal(0, 1, mu.shape), dtype=torch.float32, device = mu.device)
        return z + mu

    def forward(self, x, return_intermediate=False):
        intermediate_features = []  # Initialize the list to collect intermediate features only if needed
        
        skip_connections = []

        x = self.init_conv(x)
        if return_intermediate:
            intermediate_features.append(x)  # Collect initial features after the initial convolution

        for down in self.down_blocks:
            skip_connections.append(x)
            x = down(x)
            if return_intermediate:
                intermediate_features.append(x)  # Collect features after each down block
        
        # if self.kl_reg:
        #     mu = self.mu_conv(x)
        #     x = self.reparameterization(mu)

        # if self.kl_reg:
        #     mu = x
        #     x = self.reparameterization(mu)

        # Reverse the skip connections for the upward path
        skip_connections = skip_connections[::-1]

        for i, up in enumerate(self.up_blocks):
            x = up(x)
            if return_intermediate:
                intermediate_features.append(x)  # Collect features after each up block (before combining with skip connection)
            
            skip_x = skip_connections[i]  # Get the corresponding skip connection
            
            # Combine the skip connection based on the specified method
            if self.skip_combination == 'concat':
                x = torch.cat((x, skip_x), dim=1)
            elif self.skip_combination == 'add':
                x = x + skip_x
            else:
                pass
            
            if return_intermediate:
                intermediate_features.append(x)  # Optionally, collect features after combining with skip connection

        x = self.final_conv(x)
        if return_intermediate:
            intermediate_features.append(x)  # Collect final output features

        if self.kl_reg:
            mu = x
            x = self.reparameterization(mu)

        if return_intermediate:
            if self.kl_reg:
                return mu, x, intermediate_features  # Return the list of all collected intermediate features
            else:
                return x, intermediate_features
        else:
            if self.kl_reg:
                return mu, x  # Return only the final output
            else:
                return x
            

    # def forward(self, x):
    #     skip_connections = []

    #     x = self.init_conv(x)

    #     for down in self.down_blocks:
    #         skip_connections.append(x)
    #         x = down(x)

    #     # print("bottle",x.shape)
    #     skip_connections = skip_connections[::-1]

    #     for i, up in enumerate(self.up_blocks):
    #         x = up(x)
    #         skip_x = skip_connections[i]  # Skip the last one as it's not used for skip connections
    #         if self.skip_combination == 'concat':
    #             x = torch.cat((x, skip_x), dim=1)
    #         elif self.skip_combination == 'add':
    #             x = x + skip_x

    #     x = self.final_conv(x)
    #     return 

def test_DemodulatedConv2d():
    print("Testing DemodulatedConv2d...")
    test_input = torch.randn(1, 3, 256, 256)  # Batch size of 1, 3 channels, 256x256 pixels
    conv = DemodulatedConv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
    output = conv(test_input)
    print(f"Output shape: {output.shape}")

def test_DownConvBlock():
    print("Testing DownConvBlock...")
    test_input = torch.randn(1, 3, 256, 256)
    down_block = DownConvBlock(3, 64, 3, 1, 1, conv_method='conv2d', downsample_scale=2, norm_type='batch')
    output = down_block(test_input)
    print(f"Output shape: {output.shape}")

def test_UpConvBlock():
    print("Testing UpConvBlock...")
    # Assuming the input is after some downsampling, let's say it's now 64x64.
    test_input = torch.randn(1, 64, 64, 64)
    up_block = UpConvBlock(64, 32, 3, 1, 1, up_method='ConvTranspose', upsample_scale=2, conv_method='conv2d')
    output = up_block(test_input)
    print(f"Output shape: {output.shape}")

def test_UNet2D():
    print("Testing UNet2D...")
    in_channels = 15
    out_channels = 2
    test_input = torch.randn(1, in_channels, 256, 256)
    unet = UNet2D(in_channels=in_channels, out_channels=out_channels, mid_channels=[64, 128, 256, 512, 1024], final_channels=[32,16], kernel_size=3, stride=1, padding=1, norm_type='batch', skip_comb='concat', up_method='UpsampleConv', conv_method='conv2d')
    output = unet(test_input)
    print(f"Output shape: {output.shape}")
    import torchsummary as summary
    summary.summary(unet, (in_channels, 256, 256), device='cpu')

# Example test code
if __name__ == "__main__":
    test_DemodulatedConv2d()
    test_DownConvBlock()
    test_UpConvBlock()
    test_UNet2D()
