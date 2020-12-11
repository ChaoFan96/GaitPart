import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class BasicConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(BasicConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, bias=False, **kwargs)

    def forward(self, x):
        ret = self.conv(x)
        return ret

class FocalConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, halving, **kwargs):
        super(FocalConv2d, self).__init__()
        self.halving = halving
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, **kwargs)

    def forward(self, x):
        h = x.size(2)
        split_size = int(h // 2**self.halving)
        z = x.split(split_size, 2)
        z = torch.cat([self.conv(_) for _ in z], 2)
        return F.leaky_relu(z, inplace=True)

class TemporalFeatureAggregator(nn.Module):
    def __init__(self, in_channels, squeeze=4, part_num=16):
        super(TemporalFeatureAggregator, self).__init__()
        hidden_dim = int(in_channels // squeeze)
        self.part_num = part_num
        
        # MTB1
        conv3x1 = nn.Sequential(
                BasicConv1d(in_channels, hidden_dim, 3, padding=1), 
                nn.LeakyReLU(inplace=True), 
                BasicConv1d(hidden_dim, in_channels, 1))
        self.conv1d3x1 = clones(conv3x1, part_num)
        self.avg_pool3x1 = nn.AvgPool1d(3, stride=1, padding=1)
        self.max_pool3x1 = nn.MaxPool1d(3, stride=1, padding=1)
        
        # MTB1
        conv3x3 = nn.Sequential(
                BasicConv1d(in_channels, hidden_dim, 3, padding=1), 
                nn.LeakyReLU(inplace=True), 
                BasicConv1d(hidden_dim, in_channels, 3, padding=1))
        self.conv1d3x3 = clones(conv3x3, part_num)
        self.avg_pool3x3 = nn.AvgPool1d(5, stride=1, padding=2)
        self.max_pool3x3 = nn.MaxPool1d(5, stride=1, padding=2)

    def forward(self, x):
        """
          Input: x, [p, n, c, s]
        """
        p, n, c, s = x.size()
        feature = x.split(1, 0)
        x = x.view(-1, c, s)
        
        # MTB1: ConvNet1d & Sigmoid
        logits3x1 = torch.cat([conv(_.squeeze(0)).unsqueeze(0)
            for conv, _ in zip(self.conv1d3x1, feature)], 0)
        scores3x1 = torch.sigmoid(logits3x1)
        # MTB1: Template Function
        feature3x1 = self.avg_pool3x1(x) + self.max_pool3x1(x)
        feature3x1 = feature3x1.view(p, n, c, s)
        feature3x1 = feature3x1 * scores3x1
        
        # MTB2: ConvNet1d & Sigmoid
        logits3x3 = torch.cat([conv(_.squeeze(0)).unsqueeze(0)
            for conv, _ in zip(self.conv1d3x3, feature)], 0)
        scores3x3 = torch.sigmoid(logits3x3)
        # MTB2: Template Function
        feature3x3 = self.avg_pool3x3(x) + self.max_pool3x3(x)
        feature3x3 = feature3x3.view(p, n, c, s)
        feature3x3 = feature3x3 * scores3x3
        
        # Temporal Pooling
        ret = (feature3x1 + feature3x3).max(-1)[0]
        return ret
