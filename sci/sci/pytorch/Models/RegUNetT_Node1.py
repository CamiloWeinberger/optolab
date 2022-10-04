import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from timm.utils import accuracy
from os.path import dirname, exists

from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

from sci.pytorch.Models._base import ModelBase
from sci.pytorch.Models import load_model
from sci.pytorch.Models.regnet.regnet import ResBottleneckBlock_Reversed, AnyStage

# import all the losses
from sci.pytorch.Models.Losses import (DiceLoss, DiceBCELoss, IoULoss,
                                              FocalLoss, TverskyLoss, FocalTverskyLoss, ComboLoss,
                                              LovaszHingeLoss, MSELoss, MAELoss, RMSELoss,
                                              CrossEntropyLoss, BCELoss, BCEWithLogitsLoss,)



class RegUNetT_Node1(ModelBase, pl.LightningModule):
  def __init__(self):
    super().__init__()
    path_ckpt = f'{dirname(dirname(dirname(dirname(__file__))))}/saved_models/RegUNetT-v=02-epoch=55-loss_val=1.57.ckpt'
    model_to_load = load_model('RegUNetT')
    if exists(path_ckpt):
      self.model = model_to_load.load_from_checkpoint(path_ckpt)
      self.model.freeze()
    else:
      raise Exception(f'Checkpoint {path_ckpt} not found')

    self.node = AnyStage(64, 1, 2, 1, ResBottleneckBlock_Reversed, 1, 16, True, False, True)
    self.range = nn.Conv3d(1, 1, 1, 1, 0, bias=False)

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

    self.activation = {}
    self.model.reverse_stages[0].b1.relu.register_forward_hook(self.get_activation('node'))

    pass

  def get_activation(self, name):
    def hook(model, input, output):
      self.activation[name] = output
    return hook

  def forward(self, x):
    x = self.model(x)
    node = self.node(self.activation['node'])
    range = self.range(torch.tanh(node))

    output = x + range
    return output


if __name__ == '__main__':
  model = RegUNetT_Node1()
  x = torch.randn(1, 1, 16, 256, 256)
  y = model(x)
  print(y.shape)
