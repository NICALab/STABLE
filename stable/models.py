import torch.nn as nn
import torch.nn.functional as F
import torch

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

class MultiDiscriminator(nn.Module):
    def __init__(self, channels=1, num_scales=3, num_layers=5, downsample_stride=2, kernel_size=4, stride=2, padding=1, norm_type='none', momentum=0.1):
        super(MultiDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=False):
            """Returns downsampling layers of each discriminator block"""
            conv = nn.Conv2d(in_filters, out_filters, kernel_size, stride=stride, padding=1, padding_mode='zeros')
            layers = [conv]
            if normalize:
                if norm_type == "batch":
                    layers.append(nn.BatchNorm2d(out_filters, momentum=momentum))
                elif norm_type == "instance":
                    layers.append(nn.InstanceNorm2d(out_filters))
                elif norm_type == "none":
                    layers.append(nn.Identity())

            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.models = nn.ModuleList()
        for i in range(num_scales):
            layers = []
            current_channels = channels
            next_channels = 64
            for j in range(num_layers):
                layers.extend(discriminator_block(current_channels, next_channels, normalize=(j > 0)))
                current_channels = next_channels
                if next_channels < 512:
                    next_channels *= 2

            layers.append(nn.Conv2d(current_channels, 1, 3, padding=1, padding_mode='zeros'))
            self.models.add_module("disc_%d" % i, nn.Sequential(*layers))

        self.downsample = nn.AvgPool2d(3, stride=downsample_stride, padding=1, count_include_pad=False)

    def forward(self, x):
        outputs = []
        for m in self.models:
            outputs.append(m(x))
            x = self.downsample(x) 
        return outputs

    def compute_loss(self, model, x, gt):
        """Computes the MSE between model output and scalar gt"""
        loss = 0
        output = model.forward(x)
        for out in output:
            squared_diff = (out - gt) ** 2
            loss += torch.mean(squared_diff)
        return loss

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, demodulated, mid_channels=None, norm_type="batch", act='relu', momentum=0.1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if demodulated:
            conv1 = DemodulatedConv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
            conv2 = DemodulatedConv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        else:
            conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False, padding_mode='zeros')
            conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False, padding_mode='zeros')
        if norm_type == "batch":
            norm_layer1 = nn.BatchNorm2d(mid_channels, momentum=momentum)
            norm_layer2 = nn.BatchNorm2d(out_channels, momentum=momentum)
        elif norm_type == "instance":
            norm_layer1 = nn.InstanceNorm2d(mid_channels, momentum=momentum)
            norm_layer2 = nn.InstanceNorm2d(out_channels, momentum=momentum)
        else:
            norm_layer1 = nn.Identity()
            norm_layer2 = nn.Identity()
        if act == 'relu':
            act_layer = nn.ReLU(inplace=True)
        elif act == 'leakyrelu':
            act_layer = nn.LeakyReLU(0.2, inplace=True)
        
        self.double_conv = nn.Sequential(
            conv1,
            norm_layer1,
            act_layer,
            conv2,
            norm_layer2,
            act_layer
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, norm_type, demodulated, momentum=0.1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, norm_type=norm_type, demodulated=demodulated, momentum=momentum)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, demodulated, norm_type="batch", momentum=0.1):
        super().__init__()
        self.up = DySample(in_channels)
        self.upconv = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False, padding_mode='zeros')
        self.conv = DoubleConv(in_channels, out_channels, mid_channels=in_channels // 2, norm_type=norm_type, demodulated=demodulated, momentum=momentum)

    def forward(self, x_in, x_skip):
        x_in = self.up(x_in)
        x_in = self.upconv(x_in)    
        
        diffY = x_skip.size()[2] - x_in.size()[2]
        diffX = x_skip.size()[3] - x_in.size()[3]

        x_in = F.pad(x_in, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
    
        x = torch.cat([x_skip, x_in], dim=1)

        ret = self.conv(x)

        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, act='relu'):
        super(OutConv, self).__init__()
        
        if act == 'relu':
            act_layer = nn.ReLU(inplace=True)
        elif act == 'leakyrelu':
            act_layer = nn.LeakyReLU(0.2, inplace=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False, padding_mode='zeros'),
            act_layer,
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding_mode='zeros'),
        )

    def forward(self, x):
        return self.conv(x)
    
class UNet(nn.Module):
    def __init__(self, n_in, n_out, mid_channels=[64,128,256,512,1024], norm_type="batch", demodulated=False, act='relu', momentum=0.1):
        super(UNet, self).__init__()
        self.n_in = n_in
        self.n_out = n_out

        self.inc = (DoubleConv(n_in, mid_channels[0], norm_type=norm_type, demodulated=demodulated, act=act))

        self.downs = []
        for ch_i in range(len(mid_channels)-1):
            self.downs.append(Down(mid_channels[ch_i], mid_channels[ch_i+1], norm_type=norm_type, demodulated=demodulated, momentum=momentum))
        self.downs = nn.ModuleList(self.downs)
        
        self.ups = []
        for ch_i in range(len(mid_channels)-1, 0, -1):
            self.ups.append(Up(mid_channels[ch_i], mid_channels[ch_i-1], norm_type=norm_type, demodulated=demodulated, momentum=momentum))
        self.ups = nn.ModuleList(self.ups)

        self.outc = (OutConv(mid_channels[0], n_out, act))

    def forward(self, x):

        x = self.inc(x)
        
        down_skips = []
        for i, down_layer in enumerate(self.downs):
            down_skips.append(x)
            x = down_layer(x)

        for i, (up, skip) in enumerate(zip(self.ups, down_skips[::-1])):
            x = up(x, skip)    

        x = self.outc(x)

        return x

# Learning to Upsample by Learning to Sample
# https://github.com/tiny-smart/dysample

def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class DySample(nn.Module):
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)