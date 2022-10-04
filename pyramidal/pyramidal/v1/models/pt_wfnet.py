import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch import nn

from pyramidal.v1.models.pt_Base import ModelBase

class WFNet(ModelBase, pl.LightningModule):
  def __init__(self):
    super().__init__()

    self.loss_fn = nn.MSELoss()

    filters = [128, 64, 32]
    transpose_filters = [96, 64, 32]
    linear_units = [256, 64, 199]

    # filters = [12, 24, 32]
    # transpose_filters = [48, 64, 96]
    # linear_units = [128, 128, 199]

    self.conv_left_1 = nn.Conv2d(1, filters[0], kernel_size=[1, 12], stride=1, padding='same', bias=False)
    self.left = nn.Sequential(
      nn.ReLU(),
      nn.BatchNorm2d(filters[0]),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Conv2d(filters[0], filters[1], kernel_size=[1, 9], stride=1, padding='same', bias=False),
      nn.ReLU(),
      nn.BatchNorm2d(filters[1]),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Conv2d(filters[1], filters[2], kernel_size=[1, 3], stride=1, padding='same', bias=False),
      nn.ReLU(),
      nn.BatchNorm2d(filters[2]),
      nn.MaxPool2d(kernel_size=2, stride=1),
      nn.ZeroPad2d((0, 1, 0, 1)),
    )


    self.conv_right_1 = nn.Conv2d(1, filters[0], kernel_size=[12, 1], stride=1, padding='same', bias=False)
    self.right = nn.Sequential(
      nn.ReLU(),
      nn.BatchNorm2d(filters[0]),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Conv2d(filters[0], filters[1], kernel_size=[9, 1], stride=1, padding='same', bias=False),
      nn.ReLU(),
      nn.BatchNorm2d(filters[1]),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Conv2d(filters[1], filters[2], kernel_size=[3, 1], stride=1, padding='same', bias=False),
      nn.ReLU(),
      nn.BatchNorm2d(filters[2]),
      nn.MaxPool2d(kernel_size=2, stride=1),
      nn.ZeroPad2d((0, 1, 0, 1)),
    )

    # padding same like in tensorflow
    kernel_size = 5
    padding_same =  (kernel_size - 1) // 2

    self.middle = nn.Sequential(
      nn.Conv2d(filters[0] * 2, filters[0] * 2, kernel_size=5, stride=4, padding=padding_same, bias=False),
      nn.ReLU(),
      nn.BatchNorm2d(filters[0] * 2),
      nn.MaxPool2d(kernel_size=2, stride=1),
    )

    self.neck = nn.Sequential(
      nn.GroupNorm(num_groups=filters[-1] * 4 // 4, num_channels=filters[-1] * 10),
      nn.ConvTranspose2d(filters[-1] * 10, transpose_filters[0], kernel_size=3, stride=2, padding=(1, 1), output_padding=(1, 1), bias=False),
      nn.ConvTranspose2d(transpose_filters[0], transpose_filters[1],  kernel_size=9, stride=2, padding=(3, 3), output_padding=(1, 1), bias=False),
      nn.ConvTranspose2d(transpose_filters[1], transpose_filters[2], kernel_size=12, stride=2, padding=(4, 4), output_padding=(1, 1), bias=False),
      nn.AdaptiveAvgPool2d((1, 1)),
    )
    self.head = nn.Sequential(
      nn.Flatten(),
      nn.Linear(transpose_filters[2], linear_units[0]),
      nn.ReLU(),
      nn.Linear(linear_units[0], linear_units[1]),
      nn.ReLU(),
      nn.Linear(linear_units[1], linear_units[2]),
    )

  def forward(self, x):
    start_left   = self.conv_left_1(x)
    start_right  = self.conv_right_1(x)
    start_middle = torch.cat([start_left, start_right], dim=1)
    left         = self.left(start_left)
    right        = self.right(start_right)
    middle       = self.middle(start_middle)

    trunk       = torch.cat([left, right, middle], dim=1)
    neck        = self.neck(trunk)
    head        = self.head(neck)

    return head




if __name__ == '__main__':
  model = WFNet()
  x = torch.randn(1, 1, 134, 134)
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
