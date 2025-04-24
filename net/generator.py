import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Logging ----------
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------- Helper Modules ----------
class InstanceNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.offset = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        mean = torch.mean(x, dim=(2, 3), keepdim=True)
        variance = torch.var(x, dim=(2, 3), keepdim=True, unbiased=False)
        return self.scale * (x - mean) / torch.sqrt(variance + self.eps) + self.offset

# ---------- Core Layers ----------
class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding='valid', use_bias=False):
        super().__init__()
        self.pad = nn.ReflectionPad2d(1) if kernel_size == 3 else None
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                             stride, padding=0 if kernel_size == 3 else 1, bias=use_bias)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.conv.weight, std=0.02)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        if self.pad is not None:
            x = self.pad(x)
        return self.conv(x)

class Conv2DNormLReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding='valid', use_bias=False):
        super().__init__(
            Conv2D(in_channels, out_channels, kernel_size, stride, padding, use_bias),
            InstanceNorm(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

class DepthwiseConv(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=1, use_bias=False):
        super().__init__()
        self.pad = nn.ReflectionPad2d(1)
        self.dw_conv = nn.Conv2d(channels, channels, kernel_size,
                                stride, groups=channels, bias=use_bias)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.dw_conv.weight, std=0.02)
        if self.dw_conv.bias is not None:
            nn.init.zeros_(self.dw_conv.bias)

    def forward(self, x):
        return self.dw_conv(self.pad(x))

class SeparableConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__(
            DepthwiseConv(in_channels, kernel_size, stride),
            Conv2D(in_channels, out_channels, 1, 1, use_bias=False),
            InstanceNorm(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

# ---------- Generator Components ----------
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            SeparableConv2d(in_channels, out_channels)
        )

    def forward(self, x):
        return self.upsample(x)

class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True),
            SeparableConv2d(in_channels, out_channels)
        )

    def forward(self, x):
        return self.downsample(x)

class InvertedResBlock(nn.Module):
    def __init__(self, channels, expansion_ratio=2, stride=1):
        super().__init__()
        bottleneck_dim = round(channels * expansion_ratio)
        
        self.block = nn.Sequential(
            Conv2DNormLReLU(channels, bottleneck_dim, 1),
            DepthwiseConv(bottleneck_dim, stride=stride),
            InstanceNorm(bottleneck_dim),
            nn.LeakyReLU(0.2, inplace=True),
            Conv2D(bottleneck_dim, channels, 1),
            InstanceNorm(channels)
        )
        self.stride = stride
        self.channels = channels

    def forward(self, x):
        residual = self.block(x)
        if self.stride == 1 and residual.shape[1:] == x.shape[1:]:
            return x + residual
        return residual

# ---------- Generator Network ----------
class G_net(nn.Module):
    def __init__(self):
        super().__init__()
        logging.debug("Initializing G_net...")
        
        # Initial block
        self.b1 = nn.Sequential(
            Conv2DNormLReLU(3, 64),
            Conv2DNormLReLU(64, 64),
            SeparableConv2d(64, 128)
        )
        self.b1_down = Downsample(128, 128)

        # Second block
        self.b2 = nn.Sequential(
            Conv2DNormLReLU(128, 128),
            SeparableConv2d(128, 128),
            SeparableConv2d(128, 256)
        )
        self.b2_down = Downsample(256, 256)

        # Middle block
        self.middle = nn.Sequential(
            Conv2DNormLReLU(256, 256),
            *[InvertedResBlock(256) for _ in range(8)],
            Conv2DNormLReLU(256, 256)
        )

        # Decoder
        self.u2 = nn.Sequential(
            Upsample(256, 128),
            SeparableConv2d(128, 128),
            Conv2DNormLReLU(128, 128)
        )

        self.u1 = nn.Sequential(
            Upsample(128, 128),
            Conv2DNormLReLU(128, 64),
            Conv2DNormLReLU(64, 64)
        )

        self.final = nn.Sequential(
            Conv2D(64, 3, 1),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder
        x = self.b1(x)
        down1 = self.b1_down(x)
        x = F.interpolate(x, size=down1.shape[2:], mode='bilinear', align_corners=True)
        x = x + down1

        x = self.b2(x)
        down2 = self.b2_down(x)
        x = F.interpolate(x, size=down2.shape[2:], mode='bilinear', align_corners=True)
        x = x + down2

        # Middle
        x = self.middle(x)

        # Decoder
        x = self.u2(x)
        x = self.u1(x)
        
        return self.final(x)