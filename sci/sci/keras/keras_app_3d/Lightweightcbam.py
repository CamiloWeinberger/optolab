#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 18:16:08 2022

@author: mvaldi
"""

from sci.keras_app_3d.cbam.attention_module import cbam_block
from sci.keras_app_3d.Lightweight import Lightweight

class LightweightCBAM(Lightweight):
  @staticmethod
  def attention_block(inputs):
    x = cbam_block(inputs)
    return  x

  def __call__(self,
               input_shape=(256, 256, 16, 1)
               ):
    model = super().__call__(input_shape=input_shape)
    model._name = f'LightweightCBAM_{self.model_type}'
    print(f'\nNombre del model: {model.name}\n')

    return model

if __name__ == '__main__':

  self = LightweightCBAM()
  model = self()
  model.summary()