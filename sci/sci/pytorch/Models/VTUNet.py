import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from timm.utils import accuracy
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

from sci.pytorch.Models._base import ModelBase
from sci.pytorch.Models.vtunet_own.vtunet_tumor import VTUNet as VTUNet_model


# import all the losses
from sci.pytorch.Models.Losses import (DiceLoss, DiceBCELoss, IoULoss,
                                              FocalLoss, TverskyLoss, FocalTverskyLoss, ComboLoss,
                                              LovaszHingeLoss, MSELoss, MAELoss, RMSELoss,
                                              CrossEntropyLoss, BCELoss, BCEWithLogitsLoss,)

class VTUNet(VTUNet_model, ModelBase, pl.LightningModule):
  def __init__(self):
    super().__init__()
    self.metrics = {}
    self.metrics['mse']           = nn.MSELoss()
    self.metrics['mae']           = MAELoss()
    self.metrics['rmse']          = RMSELoss()
    self.metrics['r2']            = lambda x, y: 1 - self.metrics['mse'](x, y) / torch.var(y)
    self.metrics['psnr']          = lambda x, y: 10 * torch.log10(255. / self.metrics['mse'](x, y))
    # self.metrics['ssim']          = SSIM(data_range=255, size_average=True, channel=16)
    # self.metrics['ms_ssim']       = MS_SSIM(data_range=255, size_average=True, channel=16)

    # put your loss functions here
    self.loss_fn = MAELoss()

    self.lr = 1e-4



if __name__ == '__main__':
  import yaml
  cfg = "sci/pytorch/Models/vtunet_own/vt_unet_base.yaml"
  with open(cfg, 'r') as f:
      yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)
  model = VTUNet(pretrain_ckpt='', num_classes=1,
                  embed_dim=96,
                  win_size=8).cuda()
  model.eval()
  x = torch.randn(1, 1, 16, 256, 256).cuda()
  with torch.no_grad():
    y = model(x)
    print(y.shape)


if __name__ == '__main__':
  model = VTUNet()
  x = torch.randn(1, 1, 16, 256, 256)
  y = model(x)
  print(y.shape)