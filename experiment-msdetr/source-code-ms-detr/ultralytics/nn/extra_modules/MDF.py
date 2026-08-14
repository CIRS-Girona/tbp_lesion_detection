import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..modules.conv import *

__all__ = ['MDF']

class Channel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.Apt = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x): return self.sigmoid(self.Apt(self.dwconv(x)))


class Spatial(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, 1, 1, 1)
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x): return self.sigmoid(self.bn(self.conv1(x)))


class FCMBlock(nn.Module):
    def __init__(self, dim, ratio=0.5):
        super().__init__()
        c1, c2 = int(dim * ratio), dim - int(dim * ratio)
        self.c1, self.c2 = c1, c2
        self.conv1 = Conv(c1, c1, 3)
        self.conv12 = Conv(c1, c1, 3)
        self.conv123 = Conv(c1, dim, 1)

        self.conv2 = Conv(c2, dim, 1)
        
        self.spatial = Spatial(dim)
        self.channel = Channel(dim)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.c1, self.c2], dim=1)
        x3 = self.conv123(self.conv12(self.conv1(x1)))
        x4 = self.conv2(x2)
        x33 = self.spatial(x4) * x3
        x44 = self.channel(x3) * x4
        return x33 + x44


class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=4):
        super().__init__()
        self.groups = groups
        self.bn = nn.BatchNorm2d(in_channels * 2)
        self.fdc = nn.Conv2d(in_channels * 2, out_channels * 2 * self.groups, 1, groups=self.groups)
        self.weight = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels * 2, self.groups, 1),
                                    nn.Softmax(dim=1))
        self.fpe = nn.Conv2d(in_channels * 2, in_channels * 2, 3, 1, 1, groups=in_channels * 2)

    def forward(self, x):
        b, c, h, w = x.shape
        ffted = torch.fft.rfft2(x, norm='ortho')  #FFT变换到频域
        ffted = torch.stack((ffted.real, ffted.imag), -1) #分离实部和虚部
        ffted = rearrange(ffted, 'b c h w d -> b (c d) h w').contiguous()  #重排通道
        ffted = self.bn(ffted)
        ffted = self.fpe(ffted) + ffted #频域位置编码 + 残差
        dy_weight = self.weight(ffted)  #生成动态权重
        ffted_grouped = self.fdc(ffted).view(b, self.groups, -1, h, ffted.shape[-1])  #分组频域卷积
        ffted = torch.einsum('bgchw, bg->bchw', ffted_grouped, dy_weight.squeeze(-1).squeeze(-1)) #动态加权聚合
        ffted = rearrange(ffted, 'b (c d) h w -> b c h w d', d=2).contiguous()  #重组复数
        ffted = torch.complex(ffted[..., 0], ffted[..., 1]) #转换为复数
        return torch.fft.irfft2(ffted, s=(h, w), norm='ortho')  #逆FFT回空域


class FCFourier1(nn.Module):
    def __init__(self, c1, c2, n=1, ratio=0.5, shortcut=True):
        super().__init__()
        assert c1 == c2
        self.dim = c1
        self.shortcut = shortcut
        self.pre_bn = nn.BatchNorm2d(self.dim)
        self.pre_pw_conv = Conv(self.dim, self.dim, 1)
        self.fcm = FCMBlock(self.dim, ratio)
        self.gelu = nn.GELU()
        self.fourier = FourierUnit(self.dim, self.dim)
        self.post_pw_conv = Conv(self.dim, self.dim, 1)

    def forward(self, x):
        identity = x
        x_p = self.pre_pw_conv(self.pre_bn(x))
        x_fcm = self.fcm(x_p)
        x_fourier = self.fourier(self.gelu(x_fcm))
        x_out = self.post_pw_conv(x_fourier)
        return identity + x_out


class MultiDWConvBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert dim % 4 == 0
        self.branch_dim = dim // 4
        self.pwconv1 = Conv(dim, dim, 1)
        self.dwconv1 = Conv(self.branch_dim, self.branch_dim, 3, 1, g=self.branch_dim)
        self.dwconv2 = Conv(self.branch_dim, self.branch_dim, 5, 1, g=self.branch_dim)
        self.dwconv3 = Conv(self.branch_dim, self.branch_dim, 7, 1, g=self.branch_dim)
        self.gelu = nn.GELU()
        self.pwconv2 = Conv(dim, dim, 1)

    def forward(self, x):
        identity = x
        x_p = self.pwconv1(x)
        x1, x2, x3, x4 = torch.chunk(x_p, 4, dim=1)
        merged = torch.cat((self.dwconv1(x1), self.dwconv2(x2), self.dwconv3(x3), x4), dim=1)
        x_out = self.pwconv2(self.gelu(merged))
        return identity + x_out


class FCFourierPlus(nn.Module):
    def __init__(self, c1, c2, n=1, ratio=0.5, shortcut=True):
        super().__init__()
        assert c1 == c2
        self.fcfourier = FCFourier1(c1, c1, n, ratio, shortcut)
        self.mdwc_block = MultiDWConvBlock(c1)

    def forward(self, x):
        return self.mdwc_block(self.fcfourier(x))


class MDF(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, ratio=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels

        assert self.c % 4 == 0, f'Hidden channels ({self.c}) must be divisible by 4 for C2f_FCFPlus'

        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)


        self.m = nn.ModuleList(FCFourierPlus(self.c, self.c, n=1, ratio=ratio, shortcut=shortcut) for _ in range(n))

    def forward(self, x):
        """Forward pass adopting the standard C2f structure."""
        # 复用 C2f 的前向传播逻辑
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))