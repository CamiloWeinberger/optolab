import torch
import torch.nn as nn

class MSELoss(nn.Module):
  def __init__(self, weight=None, size_average=True):
    super(MSELoss, self).__init__()
    self.loss = nn.MSELoss()

  def forward(self, inputs, targets):
    # inputs = F.sigmoid(inputs)
    MSE = self.loss(inputs, targets)
    return MSE

class MAELoss(nn.Module):
  def __init__(self, weight=None, size_average=True):
    super(MAELoss, self).__init__()
    self.loss = nn.L1Loss()

  def forward(self, inputs, targets):
    # inputs = F.sigmoid(inputs)
    MAE = self.loss(inputs, targets)
    return MAE

class RMSELoss(nn.Module):
  def __init__(self, weight=None, size_average=True):
    super(RMSELoss, self).__init__()
    self.loss = nn.MSELoss()

  def forward(self, inputs, targets):
    # inputs = F.sigmoid(inputs)
    RMSE = torch.sqrt(self.loss(inputs, targets))
    return RMSE

class CrossEntropyLoss(nn.Module):
  def __init__(self, weight=None, size_average=True):
    super(CrossEntropyLoss, self).__init__()
    self.loss = nn.CrossEntropyLoss()

  def forward(self, inputs, targets):
    # inputs = F.sigmoid(inputs)
    CE = self.loss(inputs, targets)
    return CE

class BCELoss(nn.Module):
  def __init__(self, weight=None, size_average=True):
    super(BCELoss, self).__init__()
    self.loss = nn.BCELoss()

  def forward(self, inputs, targets):
    # inputs = F.sigmoid(inputs)
    BCE = self.loss(inputs, targets)
    return BCE

class BCEWithLogitsLoss(nn.Module):
  def __init__(self, weight=None, size_average=True):
    super(BCEWithLogitsLoss, self).__init__()
    self.loss = nn.BCEWithLogitsLoss()

  def forward(self, inputs, targets):
    # inputs = F.sigmoid(inputs)
    BCE = self.loss(inputs, targets)
    return BCE