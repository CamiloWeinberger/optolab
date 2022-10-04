import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from timm.utils import accuracy
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

from sci.pytorch.Models._base import ModelBase
from sci.pytorch.Models.gcvit.gcvit_uregnet import GCVit_URegnet_model

# import all the losses
from sci.pytorch.Models.Losses import (DiceLoss, DiceBCELoss, IoULoss,
                                              FocalLoss, TverskyLoss, FocalTverskyLoss, ComboLoss,
                                              LovaszHingeLoss, MSELoss, MAELoss, RMSELoss,
                                              CrossEntropyLoss, BCELoss, BCEWithLogitsLoss,)

class GCViT_URegnet(GCVit_URegnet_model, ModelBase, pl.LightningModule):
  def __init__(self,):
    super().__init__()
    self.metrics = {}
    self.metrics['mse']  = MSELoss()
    self.metrics['psnr'] = lambda x, y: 10 * torch.log10(255. / self.metrics['mse'](x, y))
    self.metrics['ssim'] = SSIM(data_range=255, size_average=True, channel=16)

    # put your loss functions here
    self.loss_fn = MAELoss()

    self.lr = 1e-5


if __name__ == '__main__':
  model = GCViT_URegnet()
  x = torch.randn(1, 1, 16, 256, 256)
  y = model(x)
  print(y.shape)
