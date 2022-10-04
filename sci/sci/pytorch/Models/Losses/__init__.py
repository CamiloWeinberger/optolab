import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from sci.pytorch.Models.Losses.LovaszLoss import lovasz_hinge

r"""
Usage
Some tips

Tversky and Focal-Tversky loss benefit from very low learning rates, of the
order 5e-5 to 1e-4. They would not see much improvement in my kernels until
around 7-10 epochs, upon which performance would improve significantly.

In general, if a loss function does not appear to be working well (or at all),
experiment with modifying the learning rate before moving on to other options.

You can easily create your own loss functions by combining any of the above with
Binary Cross-Entropy or any combination of other losses. Bear in mind that loss
is calculated for every batch, so more complex losses will increase runtime.

Care must be taken when writing loss functions for PyTorch. If you call a function
to modify the inputs that doesn't entirely use PyTorch's numerical methods, the
tensor will 'detach' from the the graph that maps it back through the neural network
for the purposes of backpropagation, making the loss function unusable. Discussion
of this is available here.

"""

class DiceLoss(nn.Module):
  r"""
  The Dice coefficient, or Dice-Sørensen coefficient,
  is a common metric for pixel segmentation that can also be modified
  to act as a loss function:
  """
  def __init__(self, weight=None, size_average=True):
    super(DiceLoss, self).__init__()

  def forward(self, inputs, targets, smooth=1):
    #comment out if your model contains a sigmoid or equivalent activation layer
    # inputs = F.sigmoid(inputs)

    #flatten label and prediction tensors
    inputs = inputs.view(-1)
    targets = targets.view(-1)

    intersection = (inputs * targets).sum()
    dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

    return 1 - dice

class DiceBCELoss(nn.Module):
  r"""
  This loss combines Dice loss with the standard binary cross-entropy (BCE)
  loss that is generally the default for segmentation models.
  Combining the two methods allows for some diversity in the loss,
  while benefitting from the stability of BCE. The equation for multi-class
  BCE by itself will be familiar to anyone who has studied logistic regression:
  """
  def __init__(self, weight=None, size_average=True):
    super(DiceBCELoss, self).__init__()

  def forward(self, inputs, targets, smooth=1):
    #comment out if your model contains a sigmoid or equivalent activation layer
    # inputs = F.sigmoid(inputs)

    #flatten label and prediction tensors
    inputs = inputs.view(-1)
    targets = targets.view(-1)

    intersection = (inputs * targets).sum()
    dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
    BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
    Dice_BCE = BCE + dice_loss

    return Dice_BCE

class DiceLossMulticlass(nn.Module):
  def __init__(self, weights=None, size_average=False):
    raise NotImplementedError("This class is not implemented yet.")
    super(mIoULoss, self).__init__()

  def forward(self, inputs, targets, smooth=1):
    if self.weights is not None:
        assert self.weights.shape == (targets.shape[1], )

    # make a copy not to change the default weights in the instance of DiceLossMulticlass
    weights = self.weights.copy()

    #comment out if your model contains a sigmoid or equivalent activation layer
    # inputs = F.sigmoid(inputs)

    # flatten label and prediction images, leave BATCH and NUM_CLASSES
    # (BATCH, NUM_CLASSES, H, W) -> (BATCH, NUM_CLASSES, H * W)
    inputs = inputs.view(inputs.shape[0],inputs.shape[1],-1)
    targets = targets.view(targets.shape[0],targets.shape[1],-1)

    #intersection = (inputs * targets).sum()
    intersection = (inputs * targets).sum(0).sum(1)
    #dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
    dice = (2.*intersection + smooth)/(inputs.sum(0).sum(1) + targets.sum(0).sum(1) + smooth)

    if (weights is None) and self.size_average==True:
      weights = (targets == 1).sum(0).sum(1)
      weights /= weights.sum() # so they sum up to 1

    if weights is not None:
      return 1 - (dice*weights).mean()
    else:
      return 1 - weights.mean()


class IoULoss(nn.Module):
  r"""
  The IoU metric, or Jaccard Index, is similar to the Dice metric and
  is calculated as the ratio between the overlap of the positive instances
  between two sets, and their mutual combined values:

  Like the Dice metric, it is a common means of evaluating the performance
  of pixel segmentation models.
  """
  def __init__(self, weight=None, size_average=True):
    super(IoULoss, self).__init__()

  def forward(self, inputs, targets, smooth=1):
    #comment out if your model contains a sigmoid or equivalent activation layer
    # inputs = F.sigmoid(inputs)

    #flatten label and prediction tensors
    inputs = inputs.view(-1)
    targets = targets.view(-1)

    #intersection is equivalent to True Positive count
    #union is the mutually inclusive area of all labels & predictions
    intersection = (inputs * targets).sum()
    total = (inputs + targets).sum()
    union = total - intersection

    IoU = (intersection + smooth)/(union + smooth)

    return 1 - IoU

class FocalLoss(nn.Module):
  r"""
  Focal Loss was introduced by Lin et al of Facebook AI Research in 2017
  as a means of combatting extremely imbalanced datasets where positive
  cases were relatively rare. Their paper "Focal Loss for Dense Object Detection"
  is retrievable here: https://arxiv.org/abs/1708.02002. In practice, the
  researchers used an alpha-modified version of the function so I have included
  it in this implementation.
  """
  def __init__(self, weight=None, size_average=True):
    super(FocalLoss, self).__init__()

  def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1):
    #comment out if your model contains a sigmoid or equivalent activation layer
    # inputs = F.sigmoid(inputs)

    #flatten label and prediction tensors
    inputs = inputs.view(-1)
    targets = targets.view(-1)

    #first compute binary cross-entropy
    BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
    BCE_EXP = torch.exp(-BCE)
    focal_loss = alpha * (1-BCE_EXP)**gamma * BCE

    return focal_loss


class TverskyLoss(nn.Module):
  def __init__(self, weight=None, size_average=True):
    super(TverskyLoss, self).__init__()

  def forward(self, inputs, targets, smooth=1, alpha=0.5, beta=0.5):

    #comment out if your model contains a sigmoid or equivalent activation layer
    # inputs = F.sigmoid(inputs)

    #flatten label and prediction tensors
    inputs = inputs.view(-1)
    targets = targets.view(-1)

    #True Positives, False Positives & False Negatives
    TP = (inputs * targets).sum()
    FP = ((1-targets) * inputs).sum()
    FN = (targets * (1-inputs)).sum()

    Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)

    return 1 - Tversky

class FocalTverskyLoss(nn.Module):
  r"""
  A variant on the Tversky loss that also includes the gamma modifier from Focal Loss.
  """
  def __init__(self, weight=None, size_average=True):
    super(FocalTverskyLoss, self).__init__()

  def forward(self, inputs, targets, smooth=1, alpha=0.5, beta=0.5, gamma=1):
    #comment out if your model contains a sigmoid or equivalent activation layer
    # inputs = F.sigmoid(inputs)

    #flatten label and prediction tensors
    inputs = inputs.view(-1)
    targets = targets.view(-1)

    #True Positives, False Positives & False Negatives
    TP = (inputs * targets).sum()
    FP = ((1-targets) * inputs).sum()
    FN = (targets * (1-inputs)).sum()

    Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)
    FocalTversky = (1 - Tversky)**gamma

    return FocalTversky


class ComboLoss(nn.Module):
  r"""
  This loss was introduced by Taghanaki et al in their paper
  "Combo loss: Handling input and output imbalance in multi-organ segmentation",
  retrievable here: https://arxiv.org/abs/1805.02798. Combo loss is a combination
  of Dice Loss and a modified Cross-Entropy function that, like Tversky loss,
  has additional constants which penalise either false positives or false negatives
  more respectively.
  """
  def __init__(self, weight=None, size_average=True):
    super(ComboLoss, self).__init__()

  def forward(self, inputs, targets, smooth=1, alpha=.5, ce_ratio=.5):
    #flatten label and prediction tensors
    inputs = inputs.view(-1)
    targets = targets.view(-1)

    #True Positives, False Positives & False Negatives
    intersection = (inputs * targets).sum()
    dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

    inputs = torch.clamp(inputs, np.e, 1.0 - np.e)
    out = - (alpha * ((targets * torch.log(inputs)) + ((1 - alpha) * (1.0 - targets) * torch.log(1.0 - inputs))))
    weighted_ce = out.mean(-1)
    combo = (ce_ratio * weighted_ce) - ((1 - ce_ratio) * dice)

    return combo

class LovaszHingeLoss(nn.Module):
  r"""
  This complex loss function was introduced by Berman, Triki and Blaschko in their
  paper "The Lovasz-Softmax loss: A tractable surrogate for the optimization of the
  intersection-over-union measure in neural networks", retrievable here:
  https://arxiv.org/abs/1705.08790. It is designed to optimise the Intersection
  over Union score for semantic segmentation, particularly for multi-class instances.
  Specifically, it sorts predictions by their error before calculating cumulatively how
  each error affects the IoU score. This gradient vector is then multiplied with the
  initial error vector to penalise most strongly the predictions that decreased the IoU
  score the most. This procedure is detailed by jeandebleu in his excellent summary here.

  This code is taken directly from the author's github repo here:
  https://github.com/bermanmaxim/LovaszSoftmax and all credit is to them.

  In this kernel I have implemented the flat variant that uses reshaped rank-1 tensors as
  inputs for PyTorch. You can modify it accordingly with the dimensions and class number
  of your data as needed. This code takes raw logits so ensure your model does not contain
  an activation layer prior to the loss calculation.

  I have hidden the researchers' own code below for brevity; simply load it into your kernel
  for the losses to function. In the case of their tensorflow implementation, I am still
  working to make it compatible with Keras. There are differences between the Tensorflow
  and Keras function libraries that complicate this.
  """
  def __init__(self, weight=None, size_average=True):
    super(LovaszHingeLoss, self).__init__()

  def forward(self, inputs, targets):
    # inputs = F.sigmoid(inputs)
    Lovasz = lovasz_hinge(inputs, targets, per_image=False)
    return Lovasz

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