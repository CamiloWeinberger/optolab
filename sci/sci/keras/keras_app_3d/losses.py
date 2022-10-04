#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 14:46:41 2022

@author: mvaldipucv
"""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import MeanAbsoluteError, MeanSquaredError

from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.python.keras.utils import losses_utils

# mae = MeanAbsoluteError()
# mse = MeanSquaredError()

mae_mean = MeanAbsoluteError()
mse_mean = MeanSquaredError()

mae_sum  = MeanAbsoluteError(reduction=losses_utils.ReductionV2.SUM)
mse_sum  = MeanSquaredError(reduction=losses_utils.ReductionV2.SUM,)

mae = 'mae'
mse = 'mse'

# def mse_mae(y_true, y_pred):
#   mse_error = mse(y_true, y_pred)
#   mae_error = mae(y_true, y_pred)
#   return mse_error + mae_error


def rme(y_true, y_pred):
  y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
  y_true = math_ops.cast(y_true, y_pred.dtype)
  return tf.reduce_mean(K.sqrt(K.abs(y_pred - y_true)))

def ssim_loss(y_true, y_pred):
  return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 255.0))

def ssim_loss_metric(y_true, y_pred):
  return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 255.0))