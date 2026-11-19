"""Custom layers: residual CBAM attention and learnable weighted concatenation.

CBAM (Woo et al., ECCV 2018): channel attention (shared MLP over avg- and max-pooled descriptors)
followed by spatial attention (7x7 conv over channel-wise avg/max maps), reduction ratio 16.
Differences from the paper: SiLU instead of ReLU in the MLP (YOLO convention) and a learnable
residual gate gamma (starts at 0, so pretrained features are untouched at the start of fine-tuning)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, c, reduction=16):
        super().__init__()
        h = max(c // reduction, 8)
        self.mlp = nn.Sequential(nn.Conv2d(c, h, 1, bias=False), nn.SiLU(inplace=True), nn.Conv2d(h, c, 1, bias=False))

    def forward(self, x):
        a = self.mlp(F.adaptive_avg_pool2d(x, 1)) + self.mlp(F.adaptive_max_pool2d(x, 1))
        return x * torch.sigmoid(a)


class SpatialAttention(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, padding=k // 2, bias=False)

    def forward(self, x):
        a = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)
        return x * torch.sigmoid(self.conv(a))


class CBAMAttention(nn.Module):
    """Residual CBAM: y = x + gamma * (CBAM(x) - x), gamma initialised to 0 (identity)."""
    def __init__(self, c1, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(c1, reduction)
        self.sa = SpatialAttention(kernel_size)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        refined = self.sa(self.ca(x))
        return x + self.gamma * (refined - x)


class WeightedConcat(nn.Module):
    """BiFPN-style fast normalised fusion followed by concatenation (replaces ultralytics Concat).
    Weights start at 1, i.e. identical to plain Concat."""
    def __init__(self, dimension=1, n=2):
        super().__init__()
        self.d = dimension
        self.w = nn.Parameter(torch.ones(n))
        self.eps = 1e-4

    def forward(self, x):
        n = len(x)
        w = F.relu(self.w)
        w = w / (w.sum() + self.eps) * n
        return torch.cat([x[i] * w[i] for i in range(n)], self.d)


def register(weighted_fusion=True):
    """Make the custom layers usable by name in YAML files (ultralytics resolves names in tasks.py).
    weighted_fusion=True  -> 'Concat' in YAML builds WeightedConcat
    weighted_fusion=False -> 'Concat' is the original ultralytics Concat"""
    import ultralytics.nn.tasks as tasks
    if not hasattr(tasks, "_OrigConcat"):
        tasks._OrigConcat = tasks.Concat
    tasks.CBAMAttention = CBAMAttention
    tasks.Concat = WeightedConcat if weighted_fusion else tasks._OrigConcat
