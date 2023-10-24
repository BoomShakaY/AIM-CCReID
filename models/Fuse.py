import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

EPSILON = 1e-12

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class Fuse(nn.Module):
    def __init__(self, feature_dim):
        super(Fuse, self).__init__()

        self.trans = nn.Linear(8*2048, feature_dim, bias=False)
        self.bn = nn.BatchNorm1d(feature_dim)
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.M = 8
        self.attentions = BasicConv2d(2048, self.M, kernel_size=1)
        self.trans.weight.data.normal_(0, 0.001)
        init.normal_(self.bn.weight.data, 1.0, 0.02)
        init.constant_(self.bn.bias.data, 0.0)

    def forward(self, feat, counter_feat_in):

        counter_feat = self.attentions(counter_feat_in)

        B, C, H, W = feat.size()
        _, M, AH, AW = counter_feat.size()

        x = (torch.einsum('imjk,injk->imn', (counter_feat, feat)) / float(H * W)).view(B, -1)
        x = torch.sign(x) * torch.sqrt(torch.abs(x) + EPSILON)
        x = F.normalize(x, dim=-1)
        x = self.trans(x)
        x = self.bn(x)

        return x