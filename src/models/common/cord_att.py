import torch
import torch.nn as nn

class CoordAtt(nn.Module):
    def __init__(self, inp, oup=None, reduction=32):
        super(CoordAtt, self).__init__()

        mip = max(8, inp // reduction)

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # squeeze W → (B,C,H,1)
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # squeeze H → (B,C,1,W)
        self.conv1  = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1    = nn.BatchNorm2d(mip)
        self.act    = nn.SiLU()
        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        n, c, h, w = x.size()

        # 1. Squeeze spatial dims separately
        x_h = self.pool_h(x)               # (B, C, H, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (B, C, W, 1)

        # 2. Concat along H+W and encode together
        y = torch.cat([x_h, x_w], dim=2)   # (B, C, H+W, 1)
        y = self.act(self.bn1(self.conv1(y)))  # (B, mip, H+W, 1)

        # 3. Split back and generate attention maps
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)      # (B, mip, 1, W)

        a_h = self.conv_h(x_h).sigmoid()   # (B, C, H, 1)
        a_w = self.conv_w(x_w).sigmoid()   # (B, C, 1, W)

        # 4. Recalibrate input
        return x * a_h * a_w               # broadcast: (B,C,H,W)