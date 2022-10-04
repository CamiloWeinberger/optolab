import torch
import numpy as np
import torch.nn as nn
from pytorch_lightning import LightningModule
import torch.nn.functional as F
from torch import nn

from pyramidal.pytorch.Models._base import ModelBase
from pyramidal.pytorch.Models._losses import (MSELoss, MAELoss, RMSELoss)
from pyramidal.pytorch.Utils._parser_utils import get_useful_kwargs


class wfnet(ModelBase, LightningModule):
  def __init__(self, **kwargs):
    kwargs = get_useful_kwargs(kwargs, __file__)
    ModelBase.__init__(self, **kwargs)
    LightningModule.__init__(self)

    self.save_hyperparameters(kwargs)

    self.metrics = {}
    self.metrics['mse']  = MSELoss()
    self.metrics['mae']  = MAELoss()
    self.metrics['rmse'] = RMSELoss()

    self.loss_fn = MSELoss() if kwargs['loss_fn'] == 'mse' else MAELoss()

    modes = int(kwargs['datavariant'])

    filters = [128, 64, 32]
    transpose_filters = [96, 64, 32]
    linear_units = [256, 64, modes]

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
    padding_same =  (kernel_size - 1) // 2 + 1

    self.middle = nn.Sequential(
      nn.Conv2d(filters[0] * 2, filters[0] * 2, kernel_size=5, stride=4, padding=padding_same, bias=False),
      nn.ReLU(),
      nn.BatchNorm2d(filters[0] * 2),
      nn.MaxPool2d(kernel_size=2, stride=1),
    )

    self.neck = nn.Sequential(
      nn.LocalResponseNorm(4, alpha=0.0001, beta=0.75, k=2),
      nn.ConvTranspose2d(filters[-1] * 10, transpose_filters[0], kernel_size=3, stride=2, padding=(1, 1), output_padding=(1, 1), bias=False),
      nn.ConvTranspose2d(transpose_filters[0], transpose_filters[1], kernel_size=9, stride=2, padding=(3, 3), output_padding=(1, 1), bias=False),
      nn.ConvTranspose2d(transpose_filters[1], transpose_filters[2], kernel_size=12, stride=2, padding=(4, 4), output_padding=(1, 1), bias=False),
      nn.ReLU(),
      # nn.AdaptiveAvgPool2d((1, 1)),
    )
    self.head = nn.Sequential(
      nn.Flatten(),
      nn.LazyLinear(linear_units[0]),
      nn.ReLU(),
      nn.Linear(linear_units[0], linear_units[1]),
      nn.ReLU(),
      nn.Linear(linear_units[1], linear_units[2]),
    )


  def forward(self, x):
    x = F.interpolate(x.unsqueeze(1), size=(64, 64), mode='bilinear', align_corners=True)
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

  @staticmethod
  def add_model_specific_args(parent_parser):
    parser = parent_parser.add_argument_group('gcvit')
    parser.add_argument('--loss_fn',       type=str,   default='mse', help='loss function')
    parser = ModelBase.add_model_specific_args(parent_parser)
    return parent_parser

  def prepare_weights(self):
    x = torch.randn(1, 224, 224)
    self(x)


if __name__ == '__main__':
  from torchviz import make_dot, make_dot_from_trace
  model = wfnet(**{'loss_fn': 'mse', 'datavariant': '54',
                   'lr': 1e-3,
                   'normalize_head': 'none',
                   'normalize_tail': 'none',
                   'is_float': False,
                   })
  x = torch.randn(1, 224, 224)
  y = model(x)
  print(y.shape)
  dot = make_dot(y.mean(), params=dict(model.named_parameters()))
  dot.render('wfnet', format='png')

