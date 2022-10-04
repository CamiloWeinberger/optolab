#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 14:34:03 2022

@author: mvaldipucv
"""

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

from pyramidal.v1.generators.generator_base import GeneratorBase


class Generator(GeneratorBase, Dataset):
  def __getitem__(self, index, inputs_model=None, labels=None):
    if inputs_model is None:
      indexs = self.indexed_data[index]
      paths  = self.partition[indexs]
      inputs_model, labels = self.process_path(paths[0])


    # normalize
    if self.normalize_head == 'mean_std':
      inputs_model = (inputs_model - self.values_normalize_head[0]) / self.values_normalize_head[1]
    elif self.normalize_head == 'mean_std_min_max':
      inputs_model = (inputs_model - self.values_normalize_head[0]) / self.values_normalize_head[1]
      inputs_model = (inputs_model - self.values_normalize_head[2]) / (self.values_normalize_head[3] - self.values_normalize_head[2])
    elif self.normalize_head == 'min_max':
      inputs_model = (inputs_model - self.values_normalize_head[0]) / (self.values_normalize_head[1] - self.values_normalize_head[0])

    if self.normalize_tail == 'mean_std':
      labels = (labels - self.values_normalize_tail[0]) / self.values_normalize_tail[1]
    elif self.normalize_tail == 'mean_std_min_max':
      labels = (labels - self.values_normalize_tail[0]) / self.values_normalize_tail[1]
      labels = (labels - self.values_normalize_tail[2]) / (self.values_normalize_tail[3] - self.values_normalize_tail[2])
    elif self.normalize_tail == 'min_max':
      labels = (labels - self.values_normalize_tail[0]) / (self.values_normalize_tail[1] - self.values_normalize_tail[0])

    inputs_model = torch.tensor(inputs_model).float()
    labels       = torch.tensor(labels).float()

    if len(inputs_model.shape) == 4:
      inputs_model = inputs_model.permute(0, 3, 1, 2)
    elif len(inputs_model.shape) == 3:
      inputs_model = inputs_model.permute(2, 0, 1)

    if self.is_gcvit:
      # resize to 224x224
      inputs_model = T.Resize((224, 224))(inputs_model)

    if self.return_name:
      return inputs_model, labels, paths
    return inputs_model, labels

def data_loader(dataset, batchsize, shuffle=True, num_workers=1, drop_last=False):
  return DataLoader(dataset, batch_size=batchsize, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)

if __name__ == '__main__':
  train_data = Generator(dataset='train', batchsize=1, normalize_head='mean_std', normalize_tail='none', is_gcvit=True)
  dataLoaded = DataLoader(train_data, batch_size=8, )

  type(train_data)

  X,y = next(iter(dataLoaded))

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  print(device)
