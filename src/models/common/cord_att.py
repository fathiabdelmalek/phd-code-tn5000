import torch
import torch.nn as nn


class CoordAtt(nn.Module):
    def __init__(self, inp=None, oup=None, reduction=32):
        super(CoordAtt, self).__init__()

        self.inp = inp
        self.reduction = reduction
        self.mip = max(8, (inp or 64) // reduction) if inp else 8

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Identity()  # Will be set dynamically
        self.bn1 = nn.Identity()
        self.act = nn.SiLU()
        self.conv_h = nn.Identity()
        self.conv_w = nn.Identity()
        self._built = False

    def _build_layers(self, inp):
        if self._built and self.inp == inp:
            return
        self.inp = inp
        self.mip = max(8, inp // self.reduction)
        self.conv1 = nn.Conv2d(inp, self.mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(self.mip)
        self.conv_h = nn.Conv2d(self.mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(self.mip, inp, kernel_size=1, stride=1, padding=0)
        self._built = True

    def forward(self, x):
        n, c, h, w = x.size()

        if not self._built:
            self._build_layers(c)

        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        return x * a_h * a_w
