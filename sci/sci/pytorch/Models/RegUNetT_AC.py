import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from timm.utils import accuracy
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

from sci.pytorch.Models._base import ModelBase
from sci.pytorch.Models.regnet.regnet import RegUNet_model

# import all the losses
from sci.pytorch.Models.Losses import (DiceLoss, DiceBCELoss, IoULoss,
                                              FocalLoss, TverskyLoss, FocalTverskyLoss, ComboLoss,
                                              LovaszHingeLoss, MSELoss, MAELoss, RMSELoss,
                                              CrossEntropyLoss, BCELoss, BCEWithLogitsLoss,)

class RegUNetT_AC(RegUNet_model, ModelBase, pl.LightningModule):
  def __init__(self):
    self.bn = False
    self.mode = 'add&concat'
    self.attn_transformer = True
    self.mult_filters = 0.5
    super().__init__()
    self.metrics = {}
    self.metrics['mse']     = nn.MSELoss()
    self.metrics['mae']     = MAELoss()
    self.metrics['rmse']    = RMSELoss()
    self.metrics['r2']      = lambda x, y: 1 - self.metrics['mse'](x, y) / torch.var(y)
    self.metrics['psnr']    = lambda x, y: 10 * torch.log10(255. / self.metrics['mse'](x, y))
    self.metrics['ssim']    = SSIM(data_range=255, size_average=True, channel=16)
    self.metrics['ms_ssim'] = MS_SSIM(data_range=255, size_average=True, channel=16)

    # put your loss functions here
    self.loss_fn = MAELoss()

    self.lr = 1e-4


if __name__ == '__main__':
  model = RegUNetT_AC()
  x = torch.randn(1, 1, 16, 256, 256)
  y = model(x)
  print(y.shape)
