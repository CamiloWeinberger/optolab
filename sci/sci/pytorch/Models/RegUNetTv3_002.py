from turtle import forward
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from timm.utils import accuracy
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

from sci.pytorch.Models._base import ModelBase
from sci.pytorch.Models.regnet.regnet_v2 import RegUNet_model

# import all the losses
from sci.pytorch.Models.Losses import (DiceLoss, DiceBCELoss, IoULoss,
                                              FocalLoss, TverskyLoss, FocalTverskyLoss, ComboLoss,
                                              LovaszHingeLoss, MSELoss, MAELoss, RMSELoss,
                                              CrossEntropyLoss, BCELoss, BCEWithLogitsLoss,)

class RegUNetTv3_002(RegUNet_model, ModelBase, pl.LightningModule):
  def __init__(self):
    self.bn = False
    self.mode = 'concat'
    self.attn_transformer = True
    self.mult_filters = 1.
    self.activation_fn = 'LeakyReLU'
    self.loops = [1, 1, 4, 7]
    self.filters = [24, 56, 152, 368]
    self.groups = 8
    super().__init__()

    self.maxPool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0), dilation=1, ceil_mode=False, )

    self.metrics = {}
    self.metrics['mse']           = nn.MSELoss()
    self.metrics['mae']           = MAELoss()
    self.metrics['rmse']          = RMSELoss()
    self.metrics['r2']            = lambda x, y: 1 - self.metrics['mse'](x, y) / torch.var(y)
    self.metrics['psnr']          = lambda x, y: 10 * torch.log10(255. / self.metrics['mse'](x, y))
    self.metrics['ssim']          = SSIM(data_range=255, size_average=True, channel=16)
    self.metrics['ms_ssim']       = MS_SSIM(data_range=255, size_average=True, channel=16)

    # put your loss functions here
    self.loss_fn = MAELoss()

    self.lr = 1e-4

  def forward(self, x):
    original_input = x
    # use maxpool to reduce dimension recursively until reach 16x16x16 from 16x256x256
    downsample = [self.maxPool(x)]
    for i in range(1, 4):
      downsample.append(self.maxPool(downsample[i-1]))

    x = self.conv1(x)
    skip_connections = [x]
    for index, stage in enumerate(self.stages):
      # replace x with downsample[index] for all values > 0
      if self.is_half:
        mask = (downsample[index] > 0).half()
      else:
        mask = (downsample[index] > 0).float()

      x = x * (1 - mask) + downsample[index]
      x = stage(x)
      if index != len(self.stages) - 1:
        skip_connections.append(x)

    for index, (skip, stage_r) in enumerate(zip(reversed(skip_connections), reversed(self.reverse_stages))):
      x = stage_r(x)
      if self.mode == 'concat':
        x = torch.cat([x, skip], dim=1)
      elif self.mode == 'add':
        x = (x + skip) / 2
      elif self.mode.find('&') != -1:
        layer, last_layer = self.mode.split('&')
        if index == len(self.stages) - 1:
          if last_layer == 'concat':
            x = torch.cat([x, skip], dim=1)
          else:
            x = (x + skip) / 2
        else:
          if layer == 'concat':
            x = torch.cat([x, skip], dim=1)
          else:
            x = (x + skip) / 2
      else:
        raise ValueError('Unknown mode')

    x = self.conv2(x)

    # replace x with original_input for all values > 0
    if self.is_half:
      mask = (original_input > 0).half()
    else:
      mask = (original_input > 0).float()
    x = x * (1 - mask) + original_input

    return x


if __name__ == '__main__':
  import os
  from sci.pytorch.Generators.DataModule import DataModule
  model = RegUNetTv3_002()
  transform = lambda x: x.permute(3, 2, 0, 1)
  dm = DataModule(
    '/Storage1/Matias/Datasetx30',
    os.path.basename('/Storage1/Matias/Datasetx30'),
    batch_size=1,
    type_input='3d',
    type_output='3d',
    normalize_head='none',
    normalize_tail='none',
    custom_transform_head=transform,
    custom_transform_tail=transform
    )
  dm.setup('fit')
  x = next(iter(dm.val_dataloader()))[0]
  x = x.float()
  y = model(x)
  print(y.shape)
