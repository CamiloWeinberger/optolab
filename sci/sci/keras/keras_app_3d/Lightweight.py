#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 18:16:08 2022

@author: mvaldi
"""

import tensorflow as tf
from tensorflow.keras import layers, activations, models
from tensorflow_addons.layers import GroupNormalization
from sci.keras_app_3d.model_base import ModelBase
from tensorflow.python.framework import tensor_shape

class WrapGroupNormalization(GroupNormalization):
  def __init__(self, output_dim, *args, **kwargs):
    self.output_dim = output_dim
    super().__init__(*args, **kwargs)

  def compute_output_shape(self, input_shape):
    output = self.output_dim.as_list()
    return tensor_shape.TensorShape([input_shape[0]] + output)

class Lightweight(ModelBase):

  @staticmethod
  def f1(x):
    filters = x.shape[-1]
    x       = layers.Conv3D(filters, 3, padding='same')(x)
    x       = layers.LeakyReLU()(x)
    x       = layers.Conv3D(filters, 3, padding='same')(x)
    return    x

  @staticmethod
  def split_n_features_tf(x, n):
    return tf.split(x, n, axis=-1)

  def add_revsci(self, x1, x2):
    # x1, x2 = layers.Lambda(lambda x: self.split_n_features_tf(x, 2))(x)
    y1     = layers.Add()([x1, self.f1(x2)])
    y2     = layers.Add()([x2, self.f1(y1)])
    return   layers.Add()([y1, y2])

  def add_revsci_wo_add(self, x0, x1):
    y1     = layers.Add()([x1, self.f1(x0)])
    y2     = layers.Add()([x0, self.f1(y1)])
    return y1, y2

  def add_revsci_loop(self, x1, x2):
    for _ in range(4):
      x1, x2 = self.add_revsci_wo_add(x1, x2)
    x = layers.Add()([x1, x2])
    return x

  def conv_block(self, x, filters, stride=1):
    x    = layers.Conv3D(filters, strides=(stride, stride, 1), kernel_size=3, padding='same')(x)
    output_dim = x.shape[1:]
    x    = WrapGroupNormalization(groups=8, output_dim=output_dim)(x)
    # x    = GroupNormalization(groups=4 if self.model_type == 'high' else 16)(x)
    x    = layers.ReLU()(x)
    return x

  @staticmethod
  def attention_block(inputs, ):
    x     = layers.GlobalAvgPool3D()(inputs)
    units = x.shape[-1]
    x     = layers.Dense(units)(x)
    x     = layers.ReLU()(x)
    x     = layers.Activation(activations.sigmoid)(x)
    x     = layers.Multiply()([x, inputs])
    return  x

  @staticmethod
  def final_block(inputs, filters):
    x = layers.Conv3D(filters, kernel_size=3, padding='same', activation='sigmoid')(inputs)
    return layers.Rescaling(255.)(x)

  def downgrade_dim(self, inputs, mode='max'):
    if mode == 'max':
      x = layers.MaxPool3D((2,2,1))(inputs)
    elif mode == 'avg':
      x = layers.AvgPool3D((2,2,1))(inputs)
    elif mode == 'conv':
      filters = inputs.shape[-1]
      x = self.conv_block(inputs, filters, stride=2)
    else:
      raise Exception('not implemented!!!')

    return x

  @staticmethod
  def upgrade_dim(inputs, mode='up'):
    if mode == 'up':
      x = layers.UpSampling3D((2,2,1))(inputs)
    elif mode == 'conv':
      filters = inputs.shape[-1]
      x = layers.Conv3DTranspose(filters, kernel_size=3, strides=(2, 2, 1), padding='same')(inputs)
      x = GroupNormalization(groups=16)(x)
      x = layers.ReLU()(x)
    else:
      raise Exception('not implemented!!!')

    return x

  def add_layer(self, x1, x2, mode='normal'):
    if mode == 'normal':
      return layers.Add()([x1, x2])
    elif mode == 'revsci':
      return self.add_revsci(x1, x2)
    elif mode == 'revsciloop':
      return self.add_revsci_loop(x1, x2)
    else:
      raise Exception('not implemented!!!')

  @staticmethod
  def dropout_layer(inputs):
    return layers.Dropout(.2)(inputs)

  def __init__(self, model_type:str='lower'):
    assert model_type in ['ultratiny64', 'ultratiny', 'ultralower', 'lower', 'lowerloop', 'high']
    self.model_type = model_type

  def __call__(self,
               input_shape=(256, 256, 16, 1)
               ):
    if self.model_type == 'high':
      filters = [24, 48, 64, 128, 160, 256, 288]
    elif self.model_type.find('64') != -1:
      filters = [24, 48, 64, 64, 64, 64, 64]
    elif self.model_type.find('ultra') != -1:
      filters = [128, 128, 128, 128, 128, 128, 128]
    else:
      filters = [32, 64, 96, 128, 128, 128, 128]

    if self.model_type.find('ultra') != -1:
      mode = 'normal'
    elif self.model_type.find('loop') != -1:
      mode = 'revsciloop'
    else:
      mode = 'revsci'

    inputs = layers.Input(input_shape)
    final_filters = input_shape[-1]

    x  = self.conv_block(inputs, filters[0])
    x1 = self.conv_block(x, filters[1])
    x11 = self.attention_block(x1)

    x  = self.downgrade_dim(x1)
    x  = self.dropout_layer(x)
    x  = self.conv_block(x, filters[2])
    x  = self.conv_block(x, filters[2])
    x2 = self.attention_block(x)

    x  = self.downgrade_dim(x2)
    x  = self.dropout_layer(x)
    x  = self.conv_block(x, filters[3])
    x  = self.conv_block(x, filters[3])
    x3 = self.attention_block(x)

    x  = self.downgrade_dim(x3)
    x  = self.dropout_layer(x)
    x  = self.conv_block(x, filters[4])
    x  = self.conv_block(x, filters[4])
    x4 = self.attention_block(x)

    x  = self.downgrade_dim(x4)
    x  = self.dropout_layer(x)
    x5 = self.conv_block(x, filters[5])

    x  = self.attention_block(x5)
    x  = self.conv_block(x, filters[5] // 2)
    x  = self.attention_block(x)
    x  = self.conv_block(x, filters[5] // 4)
    x  = self.attention_block(x)
    x  = self.conv_block(x, filters[5] // 8)
    x  = self.attention_block(x)
    x  = self.conv_block(x, filters[5])

    x  = self.add_layer(x, x5, mode=mode)

    x  = self.conv_block(x, filters[5])
    x  = self.conv_block(x, filters[5])
    x  = self.attention_block(x)

    x  = self.upgrade_dim(x)
    x  = self.add_layer(x, x4, mode=mode)
    x  = self.dropout_layer(x)
    x  = self.conv_block(x, filters[4])
    x  = self.conv_block(x, filters[4])
    x  = self.attention_block(x)

    x  = self.upgrade_dim(x)
    x  = self.add_layer(x, x3, mode=mode)
    x  = self.dropout_layer(x)
    x  = self.conv_block(x, filters[3])
    x  = self.conv_block(x, filters[2])
    x  = self.attention_block(x)

    x  = self.upgrade_dim(x)
    x  = self.add_layer(x, x2, mode=mode)
    x  = self.dropout_layer(x)
    x  = self.conv_block(x, filters[2])
    x  = self.conv_block(x, filters[1])
    x  = self.attention_block(x)

    x  = self.upgrade_dim(x)
    x  = self.add_layer(x, x11, mode=mode)
    x  = self.dropout_layer(x)
    x  = self.conv_block(x, filters[1])
    x  = self.conv_block(x, filters[0])
    x  = self.attention_block(x)

    x = self.final_block(x, final_filters)

    self.model = models.Model(inputs, x, name=f'Lightweight_{self.model_type}')

    return self.model

if __name__ == '__main__':

  self = Lightweight()
  model = self()
  model.summary()
