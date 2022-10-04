import torch
import numpy as np
import torch.nn.functional as F

class ModelBase:
  def training_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self(x)
    loss = self.loss_fn(y_hat, y)
    train_mae_loss = F.l1_loss(y_hat, y, reduction='mean')

    self.log_dict({'train_loss':     loss,
                  'train_mae_loss': train_mae_loss,
                  'lr':             self.optimizers().defaults['lr'],
                  'global_step':    int(self.global_step),
                  'epoch':          int(self.current_epoch),
                  })
    return loss

  def validation_step(self, batch, batch_idx):
    x, y          = batch
    y_hat         = self(x)
    loss          = self.loss_fn(y_hat, y)
    val_mae_loss  = F.l1_loss(y_hat, y)
    val_rmse_loss = F.mse_loss(y_hat, y, reduction='mean').sqrt()
    self.log_dict({'val_loss': loss, 'val_mae_loss': val_mae_loss, 'val_rmse_loss': val_rmse_loss})
    return loss

  def test_step(self, batch, batch_idx):
    x, y           = batch
    y_hat          = self(x)
    loss           = self.loss_fn(y_hat, y)
    test_mae_loss  = F.l1_loss(y_hat, y)
    test_rmse_loss = F.mse_loss(y_hat, y, reduction='mean').sqrt()
    self.log_dict({'test_loss': loss, 'test_mae_loss': test_mae_loss, 'test_rmse_loss': test_rmse_loss})
    return loss

  def configure_optimizers(self):
    optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
    return {
        'optimizer': optimizer,
        'lr_scheduler': {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=np.sqrt(0.2), patience=6, verbose=True),
            'monitor': 'val_loss',
            'frequency': 1,
        },
    }

  def predict(self, x):
    return self(x)