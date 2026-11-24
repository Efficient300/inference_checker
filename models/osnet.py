"""OSNet (Zhou et al., 2019) — архитектура совместима по именам слоёв с torchreid,
поэтому state_dict из model zoo torchreid загружается напрямую."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvLayer(nn.Module):
    def __init__(self, cin, cout, k, stride=1, padding=0, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride=stride, padding=padding, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv1x1(nn.Module):
    def __init__(self, cin, cout, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 1, stride=stride, padding=0, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv1x1Linear(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        return self.bn(self.conv(x))


class LightConv3x3(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(cout, cout, 3, stride=1, padding=1, bias=False, groups=cout)
        self.bn = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv2(self.conv1(x))))


class ChannelGate(nn.Module):
    def __init__(self, c, reduction=16):
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(c, c // reduction, 1, bias=True, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(c // reduction, c, 1, bias=True, padding=0)
        self.gate_activation = nn.Sigmoid()

    def forward(self, x):
        g = self.gate_activation(self.fc2(self.relu(self.fc1(self.global_avgpool(x)))))
        return x * g


class OSBlock(nn.Module):
    def __init__(self, cin, cout, bottleneck_reduction=4):
        super().__init__()
        mid = cout // bottleneck_reduction
        self.conv1 = Conv1x1(cin, mid)
        self.conv2a = LightConv3x3(mid, mid)
        self.conv2b = nn.Sequential(LightConv3x3(mid, mid), LightConv3x3(mid, mid))
        self.conv2c = nn.Sequential(LightConv3x3(mid, mid), LightConv3x3(mid, mid), LightConv3x3(mid, mid))
        self.conv2d = nn.Sequential(*(LightConv3x3(mid, mid) for _ in range(4)))
        self.gate = ChannelGate(mid)
        self.conv3 = Conv1x1Linear(mid, cout)
        self.downsample = Conv1x1Linear(cin, cout) if cin != cout else None

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2 = self.gate(self.conv2a(x1)) + self.gate(self.conv2b(x1)) + self.gate(self.conv2c(x1)) + self.gate(self.conv2d(x1))
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        return F.relu(x3 + identity)


class OSNet(nn.Module):
    def __init__(self, channels=(64, 256, 384, 512), layers=(2, 2, 2), feature_dim=512, num_classes=1000):
        super().__init__()
        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._make_layer(layers[0], channels[0], channels[1], True)
        self.conv3 = self._make_layer(layers[1], channels[1], channels[2], True)
        self.conv4 = self._make_layer(layers[2], channels[2], channels[3], False)
        self.conv5 = Conv1x1(channels[3], channels[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channels[3], feature_dim), nn.BatchNorm1d(feature_dim), nn.ReLU(inplace=True))
        self.classifier = nn.Linear(feature_dim, num_classes)

    @staticmethod
    def _make_layer(n, cin, cout, reduce):
        layers = [OSBlock(cin, cout)] + [OSBlock(cout, cout) for _ in range(1, n)]
        if reduce:
            layers.append(nn.Sequential(Conv1x1(cout, cout), nn.AvgPool2d(2, stride=2)))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv5(self.conv4(self.conv3(self.conv2(self.maxpool(self.conv1(x))))))
        return self.fc(self.global_avgpool(x).flatten(1))   # эмбеддинг 512 (режим eval)


VARIANTS = {
    "x1_0": (64, 256, 384, 512),
    "x0_75": (48, 192, 288, 384),
    "x0_5": (32, 128, 192, 256),
    "x0_25": (16, 64, 96, 128),
}


def build(variant: str = "x1_0", weights: str | None = None) -> nn.Module:
    m = OSNet(channels=VARIANTS[variant])
    if weights:
        sd = torch.load(weights, map_location="cpu")
        sd = sd.get("state_dict", sd)
        sd = {k.removeprefix("module."): v for k, v in sd.items() if not k.startswith(("classifier", "module.classifier"))}
        missing, unexpected = m.load_state_dict(sd, strict=False)
        missing = [k for k in missing if not k.startswith("classifier")]
        if missing or unexpected:
            raise RuntimeError(f"веса OSNet не совпали: missing={missing[:5]} unexpected={unexpected[:5]}")
    return m.eval()
