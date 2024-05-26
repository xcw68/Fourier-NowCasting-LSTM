import torch
from torch import nn

from modules.IDE import Ide


class DoubleConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, mid_channel=None):
        super(DoubleConv, self).__init__()
        if not mid_channel:
            mid_channel = out_channel
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(True),
            nn.Conv2d(mid_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True)
        )
        self.ide = nn.Sequential(Ide(out_channel),
                                 nn.ReLU(inplace=True))

    def forward(self, x):
        x1 = self.ide(x)
        x2 = x1 + x  # (B,C,H,W)
        return x2


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class CA(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16):
        super(CA, self).__init__()
        self.input_channels = input_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        middle_channel = input_channels // reduction_ratio
        if middle_channel <= 0:
            middle_channel = input_channels
        self.MLP = nn.Sequential(
            Flatten(),
            nn.Linear(input_channels, middle_channel),
            nn.ReLU(),
            nn.Linear(middle_channel, input_channels)
        )

    def forward(self, x):
        avg_values = self.avg_pool(x)
        max_values = self.max_pool(x)
        out = self.MLP(avg_values) + self.MLP(max_values)
        scale = x * torch.sigmoid(out).unsqueeze(2).unsqueeze(3).expand_as(x)
        return scale


class SA(nn.Module):
    def __init__(self, kernel_size=3):
        super(SA, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.bn(out)
        scale = x * torch.sigmoid(out)
        return scale


class CASA(nn.Module):
    def __init__(self, in_channel):
        super(CASA, self).__init__()
        self.ca = CA(in_channel)
        self.sa = SA()

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


class GFU(nn.Module):
    def __init__(self, channel):
        super(GFU, self).__init__()
        self.ide = nn.Sequential(Ide(channel * 2),
                                 nn.ReLU(inplace=True))

    def forward(self, global_x, local_x):
        cat_1 = torch.cat((global_x, local_x), dim=1)

        x = self.ide(cat_1)

        w1, w2 = torch.chunk(x, 2, dim=1)
        w1 = torch.sigmoid(w1)
        w2 = torch.sigmoid(w2)

        res = w1 * global_x + w2 * local_x

        return res
