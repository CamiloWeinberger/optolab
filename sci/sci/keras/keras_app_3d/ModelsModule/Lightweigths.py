import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from typing import Any, Dict, List, Tuple, Type
from torch import Tensor, group_norm, nn
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from functools import reduce
from operator import __add__

from sci.ModelsModule.ModelBase import ModelBase

class ConvBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding='same', groups=1):
    super(ConvBlock, self).__init__()
    # set padding same as tensorflow default
    if padding == 'same':
      if isinstance(kernel_size, list):
        padding = [k // 2 for k in kernel_size]
      elif isinstance(kernel_size, int):
        padding = kernel_size // 2
      else:
        raise ValueError('kernel_size must be int or list')

    self.conv     = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, groups=groups)
    group_norm    = 16
    if out_channels == 1:
      group_norm = 1
    self.bn       = nn.GroupNorm(group_norm, out_channels)
    self.relu     = nn.ReLU()

  def forward(self, x):
    x = self.conv(x)
    x = self.bn(x)
    x = self.relu(x)
    return x

class AttentionBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
    super(AttentionBlock, self).__init__()
    self.pool     = nn.AdaptiveAvgPool3d(1)
    self.linear   = nn.Linear(in_channels, out_channels)
    self.relu     = nn.ReLU()
    self.sigmoid  = nn.Sigmoid()

  def forward(self, inputs):
    x = self.pool(inputs)
    x = x.view(x.size(0), -1)
    x = self.linear(x)
    x = self.relu(x)
    x = self.sigmoid(x)
    x = x.view(x.size(0), x.size(1), 1, 1, 1)
    x = x * inputs
    return x

class FinalBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
    super(FinalBlock, self).__init__()
    self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, groups=groups)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    x = self.conv(x)
    x = self.sigmoid(x)
    return x

class DowngradeDim(nn.Module):
  def __init__(self, mode='max'):
    super(DowngradeDim, self).__init__()
    self.mode = mode

  def forward(self, x):
    if self.mode == 'max':
      x = F.max_pool3d(x, (1,2,1))
    elif self.mode == 'avg':
      x = F.avg_pool3d(x, (1,2,2))
    elif self.mode == 'conv':
      filters = x.shape[1]
      x = ConvBlock(filters, filters, stride=2)(x)
    else:
      raise Exception('not implemented!!!')

    return x

class UpgradeDim(nn.Module):
  def __init__(self, mode='up'):
    super(UpgradeDim, self).__init__()
    self.mode = mode

  def forward(self, x):
    if self.mode == 'up':
      x = F.interpolate(x, scale_factor=(1,2,2))
    elif self.mode == 'conv':
      filters = x.shape[1]
      x = ConvBlock(x, filters, stride=2)
    else:
      raise Exception('not implemented!!!')

    return x

class AddLayer(nn.Module):
  def __init__(self, mode='normal'):
    super(AddLayer, self).__init__()
    self.mode = mode

  def forward(self, x1, x2):
    if self.mode == 'normal':
      return x1 + x2
    elif self.mode == 'revsci':
      return self.add_revsci(x1, x2)
    elif self.mode == 'revsciloop':
      return self.add_revsci_loop(x1, x2)
    else:
      raise Exception('not implemented!!!')

class DropoutLayer(nn.Module):
  def __init__(self):
    super(DropoutLayer, self).__init__()

  def forward(self, x):
    return F.dropout(x, .2)

class Lightweight(ModelBase, pl.LightningModule):
  def __init__(self):
    super().__init__()
    self.filters = [1, 128]
    self.ConvBlock_list = nn.ModuleList()
    self.AttentionBlock_list = nn.ModuleList()
    self.DowngradeDim_list = nn.ModuleList()
    self.DropoutLayer_list = nn.ModuleList()
    self.AddLayer_list = nn.ModuleList()
    self.UpgradeDim_list = nn.ModuleList()

    self.ConvBlock_list.append(ConvBlock(self.filters[0], self.filters[1]))
    for i in range(20):
      self.ConvBlock_list.append(ConvBlock(self.filters[1], self.filters[1]))
    self.ConvBlock_list.append(ConvBlock(self.filters[1], self.filters[0]))

    for i in range(12):
      self.AttentionBlock_list.append(AttentionBlock(self.filters[1], self.filters[1]))
    self.AttentionBlock_list.append(AttentionBlock(self.filters[0], self.filters[0]))

    for i in range(4):
      self.DowngradeDim_list.append(DowngradeDim(mode='avg'))

    for i in range(8):
      self.DropoutLayer_list.append(DropoutLayer())

    for i in range(5):
      self.AddLayer_list.append(AddLayer(mode='normal'))

    for i in range(4):
      self.UpgradeDim_list.append(UpgradeDim(mode='up'))

    self.final = FinalBlock(self.filters[0], 1)

  def forward(self, x):

    x  = self.ConvBlock_list[0](x)
    x1 = self.ConvBlock_list[1](x)
    x11 = self.AttentionBlock_list[0](x1)

    x  = self.DowngradeDim_list[0](x1)
    x  = self.DropoutLayer_list[0](x)
    x  = self.ConvBlock_list[2](x)
    x  = self.ConvBlock_list[3](x)
    x2 = self.AttentionBlock_list[1](x)

    x  = self.DowngradeDim_list[1](x2)
    x  = self.DropoutLayer_list[1](x)
    x  = self.ConvBlock_list[4](x)
    x  = self.ConvBlock_list[5](x)
    x3 = self.AttentionBlock_list[2](x)

    x  = self.DowngradeDim_list[2](x3)
    x  = self.DropoutLayer_list[2](x)
    x  = self.ConvBlock_list[6](x)
    x  = self.ConvBlock_list[7](x)
    x4 = self.AttentionBlock_list[3](x)

    x  = self.DowngradeDim_list[3](x4)
    x  = self.DropoutLayer_list[3](x)
    x5 = self.ConvBlock_list[8](x)

    x  = self.AttentionBlock_list[4](x5)
    x  = self.ConvBlock_list[9](x)
    x  = self.AttentionBlock_list[5](x)
    x  = self.ConvBlock_list[10](x)
    x  = self.AttentionBlock_list[6](x)
    x  = self.ConvBlock_list[11](x)
    x  = self.AttentionBlock_list[7](x)
    x  = self.ConvBlock_list[12](x)

    x  = self.AddLayer_list[0](x, x5)

    x  = self.ConvBlock_list[13](x)
    x  = self.ConvBlock_list[14](x)
    x  = self.AttentionBlock_list[8](x)

    x  = self.UpgradeDim_list[0](x)
    x  = self.AddLayer_list[1](x, x4)
    x  = self.DropoutLayer_list[4](x)
    x  = self.ConvBlock_list[14](x)
    x  = self.ConvBlock_list[15](x)
    x  = self.AttentionBlock_list[9](x)

    x  = self.UpgradeDim_list[1](x)
    x  = self.AddLayer_list[2](x, x3)
    x  = self.DropoutLayer_list[5](x)
    x  = self.ConvBlock_list[16](x)
    x  = self.ConvBlock_list[17](x)
    x  = self.AttentionBlock_list[10](x)

    x  = self.UpgradeDim_list[2](x)
    x  = self.AddLayer_list[3](x, x2)
    x  = self.DropoutLayer_list[6](x)
    x  = self.ConvBlock_list[18](x)
    x  = self.ConvBlock_list[19](x)
    x  = self.AttentionBlock_list[11](x)

    x  = self.UpgradeDim_list[3](x)
    x  = self.AddLayer_list[4](x, x11)
    x  = self.DropoutLayer_list[7](x)
    x  = self.ConvBlock_list[20](x)
    x  = self.ConvBlock_list[21](x)
    x  = self.AttentionBlock_list[12](x)

    x = self.final(x)

    return x




if __name__ == '__main__':
  model = Lightweight()
  x = torch.randn(1, 1, 1, 256, 256)
  y = model(x)
  print(y.shape)
  print(y)
  print(y.sum())
  print(y.mean())
  print(y.std())
  print(y.min())
  print(y.max())
  print(y.var())
  print(y.median())
