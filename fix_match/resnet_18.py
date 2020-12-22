"""
The implement of ResNet.
18-layers
"""

import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.layer(x)
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet18(nn.Module):

    def __init__(self, num_classes, n_cov1=2, n_cov2=2, n_cov3=2, n_cov4=2):

        super(ResNet18, self).__init__()
        self.in_channels = 32
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.layer1 = self.make_layer(64, n_cov1, 1)
        self.layer2 = self.make_layer(128, n_cov2, 2)
        self.layer3 = self.make_layer(256, n_cov3, 2)
        self.layer4 = self.make_layer(512, n_cov4, 2)
        self.fc = nn.Linear(2048, num_classes)
        self._init()

    def _init(self):
        for p in self.parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p)

    def make_layer(self, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResBlock(self.in_channels, channels, stride))
            self.in_channels = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        for layer in self.layer1:
            out = layer(out)
        for layer in self.layer2:
            out = layer(out)
        for layer in self.layer3:
            out = layer(out)
        for layer in self.layer4:
            out = layer(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return self.fc(out)


def resNet_34(num_classes: int) -> ResNet18:
    return ResNet18(num_classes, 3, 4, 6, 3)


def resNet_18(num_classes: int) -> ResNet18:
    return ResNet18(num_classes)


def resNet(num_classes: int, ty: int = 34):
    if ty == 34:
        return resNet_34(num_classes)
    else:
        return resNet_18(num_classes)
