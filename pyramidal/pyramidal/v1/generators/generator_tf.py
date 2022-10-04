"""generator class for pyramidal data, used to generate data for the training/val/test of the model build it in keras.sequence"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 14:34:03 2022

@author: mvaldipucv
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from pyramidal.generators.generator_base import GeneratorBase

class Generator(GeneratorBase, keras.utils.Sequence):
  def __getitem__(self, index):
    indexs = self.indexed_data[index]
    paths  = self.partition[indexs]
    inputs_model = []
    labels       = []
    for path in paths:
      input_model, label = self.process_path(path)
      inputs_model.append(input_model)
      labels.append(label)
    inputs_model = np.float32(np.concatenate(inputs_model, axis=0))
    labels       = np.float32(np.concatenate(labels, axis=0))

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


    if self.return_name:
      return inputs_model, labels, paths
    return inputs_model, labels


if __name__ == '__main__':
  self = Generator(dataset='train', batchsize=1, normalize_head='mean_std', normalize_tail='mean_std')