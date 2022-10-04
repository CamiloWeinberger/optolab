from typing import Callable, Dict, IO, Optional, Union
import os
from os.path import isdir, join, basename

import torch
import numpy as np
import torch.nn.functional as F

# from scipy.io import savemat
# from skimage.metrics import peak_signal_noise_ratio as compare_psnr
# from skimage.metrics import structural_similarity   as compare_ssim

class ModelBase:
  def __init__(self, **kwargs):
    if 'lr' not in kwargs:
      self.lr           = 1e-3
      self.is_half      = False
      self.is_deepspeed = False
    else:
      self.lr           = kwargs['lr']
      self.is_half      = not kwargs['is_float']

  def training_step(self, batch, batch_idx):
    x, y   = batch
    if self.is_half:
      x = x.half()
      y = y.half()
    else:
      x = x.float()
      y = y.float()

    y_hat  = self(x)

    x, y = self.preprocess_batch_for_loss(x, y)

    loss   = self.loss_fn(y_hat, y)
    self.log('loss_train', loss)

    for name, func in self.metrics.items():
      if name in ['ssim', 'ms_ssim']:
        self.log(name + '_train', func(y_hat[:, 0], y[:, 0]))
      else:
        self.log(name + '_train', func(y_hat, y))
    return loss


  def validation_step(self, batch, batch_idx):
    x, y   = batch
    if self.is_half:
      x = x.half()
      y = y.half()
    else:
      x = x.float()
      y = y.float()

    y_hat  = self(x)

    x, y = self.preprocess_batch_for_loss(x, y)

    loss   = self.loss_fn(y_hat, y)

    self.log('loss_val', loss, sync_dist=True)
    for name, func in self.metrics.items():
      if name in ['ssim', 'ms_ssim']:
        self.log(name + '_val', func(y_hat[:, 0], y[:, 0]), sync_dist=True)
      else:
        self.log(name + '_val', func(y_hat, y), sync_dist=True)
    return loss

  def configure_optimizers(self):
    optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
    return {
        'optimizer': optimizer,
        'lr_scheduler': {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=np.sqrt(0.2), patience=6, verbose=True),
            'monitor': 'loss_val',
            'frequency': 1,
        },
    }

  def preprocess_batch_for_loss(self, x, y):
    return x, y

  def predict(self, x):
    return self(x)

  def training_epoch_end(self, outputs):
    for name, metric in self.metrics.items():
      if name in ['sam', 'sdi', 'uiqi']:
        metric.reset()

  @classmethod
  def wrap_load_from_checkpoint(
    cls,
    checkpoint_path: Union[str, IO],
    map_location: Optional[Union[Dict[str, str], str, torch.device, int, Callable]] = None,
    hparams_file: Optional[str] = None,
    strict: bool = True,
    **kwargs,
  ):
    if isdir(checkpoint_path):
      # is deepspeed checkpoint
      checkpoint_path = join(checkpoint_path, 'checkpoint', 'mp_rank_00_model_states.pt')
    # call the base class implementation
    return super(cls, cls).load_from_checkpoint(checkpoint_path, map_location, hparams_file, strict, **kwargs)

  def on_load_checkpoint(self, checkpoint):
    try:
      state_dict = checkpoint['module']
      state_dict = {k.partition('module.')[2]: state_dict[k] for k in state_dict.keys()}
      checkpoint['state_dict'] = state_dict
    except:
      pass

  def prepare_weights(self):
    pass


  @staticmethod
  def add_model_specific_args(parent_parser):
    # DO NOT MODIFY
    parser = parent_parser.add_argument_group('ModelBase')
    parser.add_argument('--lr',      type=float, default=1e-3)
    return parent_parser