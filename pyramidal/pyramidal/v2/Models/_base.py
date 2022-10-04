import torch
import numpy as np
import torch.nn.functional as F

class ModelBase:
  lr = 1e-3
  load_model_is_deepspeed = False
  def training_step(self, batch, batch_idx):
    x, y   = batch
    y_hat  = self(x)
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
    y_hat  = self(x)
    loss   = self.loss_fn(y_hat, y)
    self.log('loss_val', loss, sync_dist=True)
    for name, func in self.metrics.items():
      if name in ['ssim', 'ms_ssim']:
        self.log(name + '_val', func(y_hat[:, 0], y[:, 0]), sync_dist=True)
      else:
        self.log(name + '_val', func(y_hat, y), sync_dist=True)
    return loss

  def test_step(self, batch, batch_idx):
    x, y   = batch
    y_hat  = self(x)
    loss   = self.loss_fn(y_hat, y)
    self.log('loss_loss', loss)
    for name, func in self.metrics.items():
      if name in ['ssim', 'ms_ssim']:
        self.log(name + '_test', func(y_hat[:, 0], y[:, 0]), sync_dist=True)
      else:
        self.log(name + '_test', func(y_hat, y), sync_dist=True)
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

  def predict(self, x):
    return self(x)

  def training_epoch_end(self, outputs):
    try:
      for name, metric in self.metrics.items():
        if name in ['sam', 'sdi', 'uiqi']:
          metric.reset()
    except:
      pass