import torch
import torch.nn as nn
import torch.nn.functional as F
from ..modules import *



__all__ = ['DFFB']


# ========== MANet ==========
class MANet(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, p=1, kernel_size=3, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv_first = Conv(c1, 2 * self.c, 1, 1)
        self.cv_final = Conv((4 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.cv_block_1 = Conv(2 * self.c, self.c, 1, 1)
        dim_hid = int(p * 2 * self.c)
        self.cv_block_2 = nn.Sequential(
            Conv(2 * self.c, dim_hid, 1, 1),
            DWConv(dim_hid, dim_hid, kernel_size, 1),
            Conv(dim_hid, self.c, 1, 1)
        )

    def forward(self, x):
        y = self.cv_first(x)
        y0 = self.cv_block_1(y)
        y1 = self.cv_block_2(y)
        y2, y3 = y.chunk(2, 1)
        y = list((y0, y1, y2, y3))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv_final(torch.cat(y, 1))

class DFFB(MANet):
    def __init__(self, c1, c2, n=3, shortcut=False, p=2, kernel_size=3, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, p, kernel_size, g, e)
        self.m = nn.ModuleList(DFFB_1(self.c) for _ in range(n))


#DFFB (Dual-path Feature Fusion Block)
class DFFB_1(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.mid_dim = dim // 2
        self.dim = dim
        self.act = nn.SiLU()

        self.last_fc = nn.Conv2d(self.dim, self.dim, 1)

        # High-frequency enhancement branch
        self.fc = nn.Conv2d(self.mid_dim, self.mid_dim, 1)
        self.max_pool = nn.MaxPool2d(3, 1, 1)

        # Local feature extraction branch (使用深度可分离卷积)
        self.conv_dw = DWConv(self.mid_dim, self.mid_dim, 3, 1)
        self.conv_pw = nn.Conv2d(self.mid_dim, self.mid_dim, 1)

    def forward(self, x):
        short = x

        # Local feature branch
        lfe = self.act(self.conv_pw(self.conv_dw(x[:, :self.mid_dim, :, :])))

        # High-frequency branch
        hfe = self.act(self.fc(self.max_pool(x[:, self.mid_dim:, :, :])))

        # Fusion
        x_fused = torch.cat([lfe, hfe], dim=1)

        # Residual connection
        out = short + self.last_fc(x_fused)
        return out