#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 10:59:15 2022

@author: mvaldi-pucv-low
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class SlidingWindowApproach:
  do_pre_processing_img = None
  def __init__(self, multiply:int=1):
    self.m = int(multiply)

  def Header(self, inputs_shape=(256,256,16,1)):
    inputs = layers.Input(inputs_shape)
    pad = inputs.shape[-2]
    if self.m > 0:
      assert pad > 10

    self.add_spectral_channel = False
    if pad % 2 != 0:
      self.add_spectral_channel = True
      inputs = layers.ZeroPadding3D(((0,0), (0,0),(0,1)))(inputs)
      pad = inputs.shape[-2]

    pad *= self.m

    self.do_pre_processing_img = 'none'
    if pad != 0:
      if inputs.shape[1] % pad != 0:
        #le añado pixeles
        self.do_pre_processing_img = 'pad'
        total_pix = int(np.ceil(inputs.shape[1] / pad) * pad) - inputs.shape[1]
        self.top_left = total_pix // 2
        self.bot_right = total_pix - self.top_left
        inputs = layers.ZeroPadding3D((self.top_left, self.bot_right, 0))(inputs)

    self.inputs = inputs
    images = []
    size = inputs.shape[1] + pad
    crop_parameters = np.array(((0, size - (pad * 2))))
    do_padding = int(np.ceil(pad / 2))

    base = layers.ZeroPadding3D(padding=(do_padding, do_padding, 0))(inputs)
    if self.m == 0:
      pad = inputs.shape[1] // inputs.shape[-2]
      crop_parameters = np.array(((0, size - (inputs.shape[-2]))))
    else:
      assert list(range(0, inputs.shape[1], pad))[-1] == crop_parameters[-1]
    self.crop_shape = []

    for stride_y in list(range(0, inputs.shape[1], pad)):
      for stride_x in list(range(0, inputs.shape[2], pad)):
        crop_shape = ((crop_parameters[0] + stride_x, crop_parameters[1] - stride_x), (crop_parameters[0] + stride_y, crop_parameters[1] - stride_y), (0, 0))
        self.crop_shape += [crop_shape]
        images += [layers.Lambda(lambda x: tf.expand_dims(x, 1))(layers.Cropping3D(crop_shape)(base))]

    output = layers.Concatenate(axis=1)(images)
    self.pad = pad

    return inputs, output

  def test(self, filters=[16, 22, 25], multiply=[1,2,3]):
    for m in multiply:
      self.m = m
      m = self.m
      for f in filters:
        self.f = f
        f = self.f
        inputs = np.arange(0, 256*256*f*1, dtype=np.float32).reshape([1, 256, 256, f, 1])
        output = self.Header(inputs)
        if m > 0:
          pad = (inputs.shape[-2] + int(inputs.shape[-2] % 2 != 0)) * m
          output = layers.Lambda(lambda x: x[:,:,pad-pad//2:pad+pad//2,pad-pad//2:pad+pad//2])(output)
        x = self.Tail(output)

        x_inputs_ = inputs[0,...,0]
        x_ = x.numpy()[0,...,0]
        assert (x_inputs_ == x_).all()
        if (x_inputs_ == x_).all():
          print(f'all good m={m}\tfilters={f}')

  def HandleBackbone(self, backbone, inputs):
    end_backbone = layers.TimeDistributed(backbone)(inputs)
    pad = self.pad
    output = layers.Lambda(lambda x: x[:,:,pad-pad//2:pad+pad//2,pad-pad//2:pad+pad//2])(end_backbone)

    return output

  def Tail(self, output):
    assert self.do_pre_processing_img is not None

    images = []
    for index in range(output.shape[1]):
      images += [layers.Lambda(lambda x: x[:, index])(output)]

    x = layers.ReLU(max_value=0.)(self.inputs)
    for index, image in enumerate(images):
      x = layers.Add()([x, layers.ZeroPadding3D(self.crop_shape[index])(image)])

    if self.do_pre_processing_img == 'pad':
      crop_shape = ((self.top_left, self.bot_right), (self.top_left, self.bot_right), 0)
      x = layers.Cropping3D(crop_shape)(x)

    if self.add_spectral_channel:
      x = layers.Cropping3D(((0,0), (0,0),(0,1)))(x)

    return x




if __name__ == '__main__':
  self = SlidingWindowApproach()
  self.test()
