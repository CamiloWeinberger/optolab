#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 11:12:56 2022

@author: mvaldipucv
"""

"""
Creates a EfficientNetV2 Model as defined in:
Mingxing Tan, Quoc V. Le. (2021).
EfficientNetV2: Smaller Models and Faster Training
arXiv preprint arXiv:2104.00298.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Conv2D,
    Concatenate,
    DepthwiseConv2D,
    Dropout,
    Input,
    Multiply,
    Conv3D,
    Conv2DTranspose,
    Conv3DTranspose,
    Lambda,
)

from tensorflow_addons.layers import GroupNormalization

from sci.keras_app_3d.custom_layers import DepthwiseConv3D

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 0.001
TORCH_BATCH_NORM_EPSILON = 1e-5
CONV_KERNEL_INITIALIZER = keras.initializers.VarianceScaling(scale=2.0, mode="fan_out", distribution="truncated_normal")
# CONV_KERNEL_INITIALIZER = 'glorot_uniform'

BLOCK_CONFIGS = {
    "b0": {  # width 1.0, depth 1.0
        "first_conv_filter": 32,
        "expands": [1, 4, 4, 4, 6, 6],
        "out_channels": [16, 32, 48, 96, 112, 192],
        "depthes": [1, 2, 2, 3, 5, 8],
        "strides": [1, 2, 2, 2, 1, 2],
        "use_ses": [0, 0, 0, 1, 1, 1],
        "rescale_mode": "torch",
    },
    "b1": {  # width 1.0, depth 1.1
        "first_conv_filter": 32,
        "expands": [1, 4, 4, 4, 6, 6],
        "out_channels": [16, 32, 48, 96, 112, 192],
        "depthes": [2, 3, 3, 4, 6, 9],
        "strides": [1, 2, 2, 2, 1, 2],
        "use_ses": [0, 0, 0, 1, 1, 1],
        "rescale_mode": "torch",
    },
    "b2": {  # width 1.1, depth 1.2
        "first_conv_filter": 32,
        "output_conv_filter": 1408,
        "expands": [1, 4, 4, 4, 6, 6],
        "out_channels": [16, 32, 56, 104, 120, 208],
        "depthes": [2, 3, 3, 4, 6, 10],
        "strides": [1, 2, 2, 2, 1, 2],
        "use_ses": [0, 0, 0, 1, 1, 1],
        "rescale_mode": "torch",
    },
    "b3": {  # width 1.2, depth 1.4
        "first_conv_filter": 40,
        "output_conv_filter": 1536,
        "expands": [1, 4, 4, 4, 6, 6],
        "out_channels": [16, 40, 56, 112, 136, 232],
        "depthes": [2, 3, 3, 5, 7, 12],
        "strides": [1, 2, 2, 2, 1, 2],
        "use_ses": [0, 0, 0, 1, 1, 1],
        "rescale_mode": "torch",
    },
    "t": {  # width 1.4 * 0.8, depth 1.8 * 0.9, from timm
        "first_conv_filter": 24,
        "output_conv_filter": 1024,
        "expands": [1, 4, 4, 4, 6, 6],
        "out_channels": [24, 40, 48, 104, 128, 208],
        "depthes": [2, 4, 4, 6, 9, 14],
        "strides": [1, 2, 2, 2, 1, 2],
        "use_ses": [0, 0, 0, 1, 1, 1],
        "rescale_mode": "torch",
    },
    "s": {  # width 1.4, depth 1.8
        "first_conv_filter": 24,
        "output_conv_filter": 1280,
        "expands": [1, 4, 4, 4, 6, 6],
        "out_channels": [24, 48, 64, 128, 160, 256],
        "depthes": [2, 4, 4, 6, 9, 15],
        "strides": [1, 2, 2, 2, 1, 2],
        "use_ses": [0, 0, 0, 1, 1, 1],
        "rescale_mode": "tf",
    },
    "early": {  # S model discribed in paper early version https://arxiv.org/pdf/2104.00298v2.pdf
        "first_conv_filter": 24,
        "output_conv_filter": 1792,
        "expands": [1, 4, 4, 4, 6, 6],
        "out_channels": [24, 48, 64, 128, 160, 272],
        "depthes": [2, 4, 4, 6, 9, 15],
        "strides": [1, 2, 2, 2, 1, 2],
        "use_ses": [0, 0, 0, 1, 1, 1],
        "rescale_mode": "tf",
    },
    "m": {  # width 1.6, depth 2.2
        "first_conv_filter": 24,
        "output_conv_filter": 1280,
        "expands": [1, 4, 4, 4, 6, 6, 6],
        "out_channels": [24, 48, 80, 160, 176, 304, 512],
        "depthes": [3, 5, 5, 7, 14, 18, 5],
        "strides": [1, 2, 2, 2, 1, 2, 1],
        "use_ses": [0, 0, 0, 1, 1, 1, 1],
        "rescale_mode": "tf",
    },
    "l": {  # width 2.0, depth 3.1
        "first_conv_filter": 32,
        "output_conv_filter": 1280,
        "expands": [1, 4, 4, 4, 6, 6, 6],
        "out_channels": [32, 64, 96, 192, 224, 384, 640],
        "depthes": [4, 7, 7, 10, 19, 25, 7],
        "strides": [1, 2, 2, 2, 1, 2, 1],
        "use_ses": [0, 0, 0, 1, 1, 1, 1],
        "rescale_mode": "tf",
    },
    "xl": {
        "first_conv_filter": 32,
        "output_conv_filter": 1280,
        "expands": [1, 4, 4, 4, 6, 6, 6],
        "out_channels": [32, 64, 96, 192, 256, 512, 640],
        "depthes": [4, 8, 8, 16, 24, 32, 8],
        "strides": [1, 2, 2, 2, 1, 2, 1],
        "use_ses": [0, 0, 0, 1, 1, 1, 1],
        "rescale_mode": "tf",
    },
}

FILE_HASH_DICT = {
    "b0": {"21k-ft1k": "4e4da4eb629897e4d6271e131039fe75", "21k": "5dbb4252df24b931e74cdd94d150f25a", "imagenet": "9abdc43cb00f4cb06a8bdae881f412d6"},
    "b1": {"21k-ft1k": "5f1aee82209f4f0f20bd24460270564e", "21k": "a50ae65b50ceff7f5283be2f4506d2c2", "imagenet": "5d4223b59ff268828d5112a1630e234e"},
    "b2": {"21k-ft1k": "ec384b84441ddf6419938d1e5a0cbef2", "21k": "9f718a8bbb7b63c5313916c5e504790d", "imagenet": "1814bc08d4bb7a5e0ed3ccfe1cf18650"},
    "b3": {"21k-ft1k": "4a27827b0b2df508bed31ae231003bb1", "21k": "ade5bdbbdf1d54c4561aa41511525855", "imagenet": "cda85b8494c7ec5a68dffb335a254bab"},
    "l": {"21k-ft1k": "30327edcf1390d10e9a0de42a2d731e3", "21k": "7970f913eec1b4918e007c8580726412", "imagenet": "2b65f5789f4d2f1bf66ecd6d9c5c2d46"},
    "m": {"21k-ft1k": "0c236c3020e3857de1e5f2939abd0cc6", "21k": "3923c286366b2a5137f39d1e5b14e202", "imagenet": "ac3fd0ff91b35d18d1df8f1895efe1d5"},
    "s": {"21k-ft1k": "93046a0d601da46bfce9d4ca14224c83", "21k": "10b05d878b64f796ab984a5316a4a1c3", "imagenet": "3b91df2c50c7a56071cca428d53b8c0d"},
    "t": {"imagenet": "4a0ff9cb396665734d7ca590fa29681b"},
    "xl": {"21k-ft1k": "9aaa2bd3c9495b23357bc6593eee5bce", "21k": "c97de2770f55701f788644336181e8ee"},
    "v1-b0": {"noisy_student": "d125a518737c601f8595937219243432", "imagenet": "cc7d08887de9df8082da44ce40761986"},
    "v1-b1": {"noisy_student": "8f44bff58fc5ef99baa3f163b3f5c5e8", "imagenet": "a967f7be55a0125c898d650502c0cfd0"},
    "v1-b2": {"noisy_student": "b4ffed8b9262df4facc5e20557983ef8", "imagenet": "6c8d1d3699275c7d1867d08e219e00a7"},
    "v1-b3": {"noisy_student": "9d696365378a1ebf987d0e46a9d26ddd", "imagenet": "d78edb3dc7007721eda781c04bd4af62"},
    "v1-b4": {"noisy_student": "a0f61b977544493e6926186463d26294", "imagenet": "4c83aa5c86d58746a56675565d4f2051"},
    "v1-b5": {"noisy_student": "c3b6eb3f1f7a1e9de6d9a93e474455b1", "imagenet": "0bda50943b8e8d0fadcbad82c17c40f5"},
    "v1-b6": {"noisy_student": "20dd18b0df60cd7c0387c8af47bd96f8", "imagenet": "da13735af8209f675d7d7d03a54bfa27"},
    "v1-b7": {"noisy_student": "7f6f6dd4e8105e32432607ad28cfad0f", "imagenet": "d9c22b5b030d1e4f4c3a96dbf5f21ce6"},
    "v1-l2": {"noisy_student": "5fedc721febfca4b08b03d1f18a4a3ca"},
}


def _make_divisible(v, divisor=4, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv2d_no_bias(inputs, filters, kernel_size, strides=1, padding="VALID", use_torch_padding=False, name=""):
    pad = max(kernel_size) // 2 if isinstance(kernel_size, (list, tuple)) else kernel_size // 2
    if filters is None:
      filters = inputs.shape[-1]
    if use_torch_padding and padding.upper() == "SAME" and pad != 0:
        inputs = keras.layers.ZeroPadding2D(padding=pad, name=name and name + "pad")(inputs)
        padding = "VALID"

    return Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=False, kernel_initializer=CONV_KERNEL_INITIALIZER, name=name + "conv")(
        inputs
    )

def conv2d_no_bias_transpose(inputs, filters, kernel_size, strides=1, padding="VALID", use_torch_padding=False, name=""):
    pad = max(kernel_size) // 2 if isinstance(kernel_size, (list, tuple)) else kernel_size // 2
    if use_torch_padding and padding.upper() == "SAME" and pad != 0:
        inputs = keras.layers.ZeroPadding2D(padding=pad, name=name and name + "pad")(inputs)
        padding = "VALID"

    return Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding, use_bias=False, kernel_initializer=CONV_KERNEL_INITIALIZER, name=name + "conv")(
        inputs
    )

def batchnorm_with_activation(inputs, activation="swish", use_torch_eps=False, name=""):
    """Performs a batch normalization followed by an activation. """
    bn_axis = 1 if K.image_data_format() == "channels_first" else -1
    # filters = inputs.shape[-1]
    nn = GroupNormalization(8,
        axis=bn_axis,
        # momentum=BATCH_NORM_DECAY,
        # epsilon=TORCH_BATCH_NORM_EPSILON if use_torch_eps else BATCH_NORM_EPSILON,
        name=name + "bn",
    )(inputs)
    if activation:
        nn = Activation(activation=activation, name='Unet_' + name + activation)(nn)
        # nn = PReLU(shared_axes=[1, 2], alpha_initializer=tf.initializers.Constant(0.25), name=name + "PReLU")(nn)
    return nn


def se_module(inputs, se_ratio=4, name=""):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    h_axis, w_axis = [2, 3] if K.image_data_format() == "channels_first" else [1, 2]

    filters = inputs.shape[channel_axis]
    # reduction = _make_divisible(filters // se_ratio, 8)
    reduction = filters // se_ratio
    # se = GlobalAveragePooling2D()(inputs)
    # se = Reshape((1, 1, filters))(se)
    se = Lambda(lambda x: tf.reduce_mean(x, [h_axis, w_axis], keepdims=True))(inputs)
    se = Conv2D(reduction, kernel_size=1, use_bias=True, kernel_initializer=CONV_KERNEL_INITIALIZER, name=name + "1_conv")(se)
    # se = PReLU(shared_axes=[1, 2])(se)
    se = Activation("swish", name='Unet_swish_' + name)(se)
    se = Conv2D(filters, kernel_size=1, use_bias=True, kernel_initializer=CONV_KERNEL_INITIALIZER, name=name + "2_conv")(se)
    se = Activation("sigmoid", name='Unet_sigmoid_' + name)(se)
    return Multiply()([inputs, se])

def se_module_transpose(inputs, se_ratio=4, name=""):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    h_axis, w_axis = [2, 3] if K.image_data_format() == "channels_first" else [1, 2]

    filters = inputs.shape[channel_axis]
    # reduction = _make_divisible(filters // se_ratio, 8)
    reduction = filters // se_ratio
    # se = GlobalAveragePooling2D()(inputs)
    # se = Reshape((1, 1, filters))(se)
    se = Lambda(lambda x: tf.reduce_mean(x, [h_axis, w_axis], keepdims=True))(inputs)
    se = Conv2DTranspose(reduction, kernel_size=1, use_bias=True, kernel_initializer=CONV_KERNEL_INITIALIZER, name=name + "1_conv")(se)
    # se = PReLU(shared_axes=[1, 2])(se)
    se = Activation("swish", name='Unet_swish_transpose_' + name)(se)
    se = Conv2DTranspose(filters, kernel_size=1, use_bias=True, kernel_initializer=CONV_KERNEL_INITIALIZER, name=name + "2_conv")(se)
    se = Activation("sigmoid", name='Unet_sigmoid_transpose_' + name)(se)
    return Multiply()([inputs, se])

# inputs, output_channel, stride, expand_ratio, shortcut, kernel_size, drop_rate, use_se, is_fused, is_torch_mode = nn, out, stride, expand, shortcut, kernel_size, block_drop_rate, se, is_fused, is_torch_mode
def MBConv(inputs, output_channel, stride, expand_ratio, shortcut, kernel_size=3, drop_rate=0, use_se=0, is_fused=False, is_torch_mode=False, name=""):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    input_channel = inputs.shape[channel_axis]

    if is_fused and expand_ratio != 1:
        nn = conv2d_no_bias(inputs, input_channel * expand_ratio, (3, 3), stride, padding="same", use_torch_padding=is_torch_mode, name=name + "sortcut_")
        nn = batchnorm_with_activation(nn, use_torch_eps=is_torch_mode, name=name + "sortcut_")
    elif expand_ratio != 1:
        nn = conv2d_no_bias(inputs, input_channel * expand_ratio, (1, 1), strides=(1, 1), padding="valid", name=name + "sortcut_")
        nn = batchnorm_with_activation(nn, use_torch_eps=is_torch_mode, name=name + "sortcut_")
    else:
        nn = inputs

    if not is_fused:
        if is_torch_mode and kernel_size // 2 > 0:
            nn = keras.layers.ZeroPadding2D(padding=kernel_size // 2, name=name + "pad")(nn)
            pad = "VALID"
        else:
            pad = "SAME"
        nn = DepthwiseConv2D(kernel_size, padding=pad, strides=stride, use_bias=False, depthwise_initializer=CONV_KERNEL_INITIALIZER, name=name + "MB_dw_")(nn)
        nn = batchnorm_with_activation(nn, use_torch_eps=is_torch_mode, name=name + "MB_dw_")

    if use_se:
        nn = se_module(nn, se_ratio=4 * expand_ratio, name=name + "se_")

    # pw-linear
    if is_fused and expand_ratio == 1:
        nn = conv2d_no_bias(nn, output_channel, (3, 3), strides=stride, padding="same", use_torch_padding=is_torch_mode, name=name + "fu_")
        nn = batchnorm_with_activation(nn, use_torch_eps=is_torch_mode, name=name + "fu_")
    else:
        nn = conv2d_no_bias(nn, output_channel, (1, 1), strides=(1, 1), padding="valid", name=name + "MB_pw_")
        nn = batchnorm_with_activation(nn, use_torch_eps=is_torch_mode, activation=None, name=name + "MB_pw_")

    if shortcut:
        if drop_rate > 0:
            nn = Dropout(drop_rate, noise_shape=(None, 1, 1, 1), name=name + "drop")(nn)
        return Add()([inputs, nn])
    else:
        return nn

# inputs, output_channel, stride, expand_ratio, shortcut, kernel_size, drop_rate, use_se, is_fused, is_torch_mode, name = nn, out, stride, expand, shortcut, kernel_size, block_drop_rate, se, is_fused, is_torch_mode, name
def MBConv_Transpose(inputs, output_channel, stride, expand_ratio, shortcut, kernel_size=3, drop_rate=0, use_se=0, is_fused=False, is_torch_mode=False, name=""):
  channel_axis = 1 if K.image_data_format() == "channels_first" else -1
  input_channel = inputs.shape[channel_axis]

  if is_fused and expand_ratio != 1:
      nn = conv2d_no_bias_transpose(inputs, input_channel * expand_ratio, 3, stride, padding="same", use_torch_padding=is_torch_mode, name=name + "sortcut_")
      nn = batchnorm_with_activation(nn, use_torch_eps=is_torch_mode, name=name + "sortcut_")
  elif expand_ratio != 1:
      nn = conv2d_no_bias_transpose(inputs, input_channel * expand_ratio, 1, strides=1, padding="valid", name=name + "sortcut_")
      nn = batchnorm_with_activation(nn, use_torch_eps=is_torch_mode, name=name + "sortcut_")
  else:
      nn = inputs

  if not is_fused:
      if is_torch_mode and kernel_size // 2 > 0:
          nn = keras.layers.ZeroPadding2D(padding=kernel_size // 2, name=name + "pad")(nn)
          pad = "VALID"
      else:
          pad = "SAME"
      nn = conv2d_no_bias_transpose(nn, input_channel, kernel_size, stride, padding=pad, name=name + "MB_dw_")
      nn = batchnorm_with_activation(nn, use_torch_eps=is_torch_mode, name=name + "MB_dw_")

  if use_se:
      nn = se_module_transpose(nn, se_ratio=4 * expand_ratio, name=name + "se_")

  # pw-linear
  if is_fused and expand_ratio == 1:
      nn = conv2d_no_bias_transpose(nn, output_channel, (3, 3), strides=stride, padding="same", use_torch_padding=is_torch_mode, name=name + "fu_")
      nn = batchnorm_with_activation(nn, use_torch_eps=is_torch_mode, name=name + "fu_")
  else:
      nn = conv2d_no_bias_transpose(nn, output_channel, (1, 1), strides=(1, 1), padding="valid", name=name + "MB_pw_")
      nn = batchnorm_with_activation(nn, use_torch_eps=is_torch_mode, activation=None, name=name + "MB_pw_")

  if shortcut:
      if drop_rate > 0:
          nn = Dropout(drop_rate, noise_shape=(None, 1, 1, 1), name=name + "drop")(nn)
      return Add()([inputs, nn])
  else:
      return nn


def EfficientNetV2(
    model_type,
    input_shape=(256, 256, 1),
    num_classes=1000,
    dropout=0.2,
    first_strides=2,
    is_torch_mode=False,
    drop_connect_rate=0,
    classifier_activation="softmax",
    include_preprocessing=False,
    model_name="EfficientNetV2",
    kwargs=None,  # Not used, just recieving parameter
):
    model_name += f'_{model_type}'
    if isinstance(model_type, dict):  # For EfficientNetV1 configures
        model_type, blocks_config = model_type.popitem()
    else:
        blocks_config = BLOCK_CONFIGS.get(model_type.lower(), BLOCK_CONFIGS["s"])
    expands = blocks_config["expands"]
    out_channels = blocks_config["out_channels"]
    depthes = blocks_config["depthes"]
    strides = blocks_config["strides"]
    use_ses = blocks_config["use_ses"]
    first_conv_filter = blocks_config.get("first_conv_filter", out_channels[0])
    kernel_sizes = blocks_config.get("kernel_sizes", [3] * len(depthes))
    # "torch" for all V1 models
    rescale_mode = "tf"

    inputs = Input(shape=input_shape)
    if include_preprocessing and rescale_mode == "torch":
        channel_axis = 1 if K.image_data_format() == "channels_first" else -1
        Normalization = keras.layers.Normalization if hasattr(keras.layers, "Normalization") else keras.layers.experimental.preprocessing.Normalization
        mean = tf.constant([0.485, 0.456, 0.406]) * 255.0
        std = (tf.constant([0.229, 0.224, 0.225]) * 255.0) ** 2
        nn = Normalization(mean=mean, variance=std, axis=channel_axis)(inputs)
    elif include_preprocessing and rescale_mode == "tf":
        Rescaling = keras.layers.Rescaling if hasattr(keras.layers, "Rescaling") else keras.layers.experimental.preprocessing.Rescaling
        nn = Rescaling(scale=1.0 / 128.0, offset=-1)(inputs)
    else:
        nn = inputs
    out_channel = _make_divisible(first_conv_filter, 8)
    nn = conv2d_no_bias(nn, out_channel, 3, strides=first_strides, padding="same", use_torch_padding=is_torch_mode, name="stem_")
    nn = batchnorm_with_activation(nn, use_torch_eps=is_torch_mode, name="stem_")

    pre_out = out_channel
    global_block_id = 0
    total_blocks = sum(depthes)
    nn_sum = []
    for id, (expand, out_channel, depth, stride, se, kernel_size) in enumerate(zip(expands, out_channels, depthes, strides, use_ses, kernel_sizes)):
      out = _make_divisible(out_channel, 8)
      is_fused = True if se == 0 else False
      for block_id in range(depth):
          stride = stride if block_id == 0 else 1
          shortcut = True if out == pre_out and stride == 1 else False
          name = "stack_{}_block{}_".format(id, block_id)
          block_drop_rate = drop_connect_rate * global_block_id / total_blocks

          nn = MBConv(nn, out, stride, expand, shortcut, kernel_size, block_drop_rate, se, is_fused, is_torch_mode, name=name)

          nn_sum.append(nn)
          pre_out = out
          global_block_id += 1

    global_block_id -= 1

    do_stride = None
    for id, (expand, out_channel, depth, stride_, se, kernel_size) in reversed(list(enumerate(zip(expands, out_channels, depthes, strides, use_ses, kernel_sizes)))):
      out = _make_divisible(out_channel, 8)
      is_fused = True if se == 0 else False
      for block_id in reversed(range(depth)):
        stride = stride_ if block_id == 0 else 1
        if stride != 1:
          do_stride = False
          num_stride = stride
          stride = 1
        if do_stride == True:
          stride = num_stride
          do_stride = None
        elif do_stride == False:
          do_stride = True
        shortcut = True if out == pre_out and stride == 1 else False
        name = "stack_{}_block{}_transposed_".format(id, block_id)
        block_drop_rate = drop_connect_rate * global_block_id / total_blocks

        nn = MBConv_Transpose(nn, out, stride, expand, shortcut, kernel_size, block_drop_rate, se, is_fused, is_torch_mode, name=name)

        nn = Add()([nn, nn_sum[global_block_id]])
        pre_out = out
        global_block_id -= 1

    nn = conv2d_no_bias_transpose(nn, out_channel, 3, strides=first_strides, padding="same", use_torch_padding=is_torch_mode, name="end_stem_1")
    nn = batchnorm_with_activation(nn, use_torch_eps=is_torch_mode, name="end_stem_")

    nn = conv2d_no_bias_transpose(nn, 1, 3, strides=1, padding="same", use_torch_padding=is_torch_mode, name="end_stem_2")
    nn = keras.layers.ReLU(max_value=1., negative_slope=0.)(nn)

    model = Model(inputs=inputs, outputs=nn, name=model_name)
    model.rescale_mode = rescale_mode
    return model

def EfficientNetV2_1D(
    model_type,
    input_shape=(256, 256, 16),
    num_classes=1000,
    dropout=0.2,
    first_strides=2,
    is_torch_mode=False,
    drop_connect_rate=0,
    classifier_activation="softmax",
    include_preprocessing=False,
    model_name="EfficientNetV2",
    kwargs=None,  # Not used, just recieving parameter
):
    model_name += f'_{model_type}'
    if isinstance(model_type, dict):  # For EfficientNetV1 configures
        model_type, blocks_config = model_type.popitem()
    else:
        blocks_config = BLOCK_CONFIGS.get(model_type.lower(), BLOCK_CONFIGS["s"])
    expands = blocks_config["expands"]
    out_channels = blocks_config["out_channels"]
    depthes = blocks_config["depthes"]
    strides = blocks_config["strides"]
    use_ses = blocks_config["use_ses"]
    first_conv_filter = blocks_config.get("first_conv_filter", out_channels[0])
    kernel_sizes = blocks_config.get("kernel_sizes", [3] * len(depthes))
    # "torch" for all V1 models
    rescale_mode = "tf"

    inputs = Input(shape=input_shape)
    if include_preprocessing and rescale_mode == "torch":
        channel_axis = 1 if K.image_data_format() == "channels_first" else -1
        Normalization = keras.layers.Normalization if hasattr(keras.layers, "Normalization") else keras.layers.experimental.preprocessing.Normalization
        mean = tf.constant([0.485, 0.456, 0.406]) * 255.0
        std = (tf.constant([0.229, 0.224, 0.225]) * 255.0) ** 2
        nn = Normalization(mean=mean, variance=std, axis=channel_axis)(inputs)
    elif include_preprocessing and rescale_mode == "tf":
        Rescaling = keras.layers.Rescaling if hasattr(keras.layers, "Rescaling") else keras.layers.experimental.preprocessing.Rescaling
        nn = Rescaling(scale=1.0 / 128.0, offset=-1)(inputs)
    else:
        nn = inputs
    out_channel = _make_divisible(first_conv_filter, 8)
    nn = conv2d_no_bias(nn, out_channel, 3, strides=first_strides, padding="same", use_torch_padding=is_torch_mode, name="stem_")
    nn = batchnorm_with_activation(nn, use_torch_eps=is_torch_mode, name="stem_")

    pre_out = out_channel
    global_block_id = 0
    total_blocks = sum(depthes)
    nn_sum = []
    for id, (expand, out_channel, depth, stride, se, kernel_size) in enumerate(zip(expands, out_channels, depthes, strides, use_ses, kernel_sizes)):
      out = _make_divisible(out_channel, 8)
      is_fused = True if se == 0 else False
      for block_id in range(depth):
          stride = stride if block_id == 0 else 1
          shortcut = True if out == pre_out and stride == 1 else False
          name = "stack_{}_block{}_".format(id, block_id)
          block_drop_rate = drop_connect_rate * global_block_id / total_blocks

          nn = MBConv(nn, out, stride, expand, shortcut, kernel_size, block_drop_rate, se, is_fused, is_torch_mode, name=name)

          nn_sum.append(nn)
          pre_out = out
          global_block_id += 1

    global_block_id -= 1

    do_stride = None
    for id, (expand, out_channel, depth, stride_, se, kernel_size) in reversed(list(enumerate(zip(expands, out_channels, depthes, strides, use_ses, kernel_sizes)))):
      out = _make_divisible(out_channel, 8)
      is_fused = True if se == 0 else False
      for block_id in reversed(range(depth)):
        stride = stride_ if block_id == 0 else 1
        if stride != 1:
          do_stride = False
          num_stride = stride
          stride = 1
        if do_stride == True:
          stride = num_stride
          do_stride = None
        elif do_stride == False:
          do_stride = True
        shortcut = True if out == pre_out and stride == 1 else False
        name = "stack_{}_block{}_transposed_".format(id, block_id)
        block_drop_rate = drop_connect_rate * global_block_id / total_blocks

        nn = MBConv_Transpose(nn, out, stride, expand, shortcut, kernel_size, block_drop_rate, se, is_fused, is_torch_mode, name=name)

        nn = Add()([nn, nn_sum[global_block_id]])
        pre_out = out
        global_block_id -= 1

    saved_nn = nn
    all_nn = []
    for i in range(16):
      nn = saved_nn
      nn = conv2d_no_bias_transpose(nn, out_channel, 3, strides=first_strides, padding="same", use_torch_padding=is_torch_mode, name=f"end_stem_1_transposed_{i}_")
      nn = batchnorm_with_activation(nn, use_torch_eps=is_torch_mode, name=f"end_stem_transposed_{i}_")
      nn = conv2d_no_bias_transpose(nn, 1, 3, strides=1, padding="same", use_torch_padding=is_torch_mode, name=f"end_stem_2_transposed_{i}_")
      all_nn.append(nn)

    nn = keras.layers.Concatenate()(all_nn)
    nn = keras.layers.Reshape([256, 256, 16, 1])(nn)
    nn = keras.layers.ReLU(max_value=1., negative_slope=0.)(nn)


    model = Model(inputs=inputs, outputs=nn, name=model_name)
    model.rescale_mode = rescale_mode
    return model


"""Model 3D"""
def split_n_features_tf(x, n):
  x_list = tf.split(x, n, axis=-1)
  return x_list

def f1(x):
  filters = x.shape[-1]
  x = Conv3D(filters, 3, padding='same')(x)
  x = keras.layers.LeakyReLU()(x)
  x = Conv3D(filters, 3, padding='same')(x)
  return x

def rev_3d_part(x):
  x1, x2 = split_n_features_tf(x, 2)
  y1 = Add()([x1, f1(x2)])
  y2 = Add()([x2, f1(y1)])
  return Add()([y1, y2])

def conv3d_no_bias(inputs, filters, kernel_size, strides=1, padding="VALID", use_torch_padding=False, name=""):
    pad = max(kernel_size) // 2 if isinstance(kernel_size, (list, tuple)) else kernel_size // 2
    if filters is None:
      filters = inputs.shape[-1]
    if use_torch_padding and padding.upper() == "SAME" and pad != 0:
        inputs = keras.layers.ZeroPadding2D(padding=pad, name=name and name + "pad")(inputs)
        padding = "VALID"

    return keras.layers.ReLU()(Conv3D(filters, kernel_size, strides=(strides, strides, 1), padding=padding, use_bias=False, kernel_initializer=CONV_KERNEL_INITIALIZER, name=name + "conv")(
        inputs
    ))

def conv3d_no_bias_transpose(inputs, filters, kernel_size, strides=1, padding="VALID", use_torch_padding=False, name=""):
    pad = max(kernel_size) // 2 if isinstance(kernel_size, (list, tuple)) else kernel_size // 2
    if use_torch_padding and padding.upper() == "SAME" and pad != 0:
        inputs = keras.layers.ZeroPadding2D(padding=pad, name=name and name + "pad")(inputs)
        padding = "VALID"

    return keras.layers.LeakyReLU()(Conv3DTranspose(filters, kernel_size, strides=(strides, strides, 1), padding=padding, use_bias=False, kernel_initializer=CONV_KERNEL_INITIALIZER, name=name + "conv")(inputs))


def se_module_3d(inputs, se_ratio=4, name=""):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    h_axis, w_axis = [2, 3] if K.image_data_format() == "channels_first" else [1, 2]

    filters = inputs.shape[channel_axis]
    # reduction = _make_divisible(filters // se_ratio, 8)
    reduction = filters // se_ratio
    # se = GlobalAveragePooling2D()(inputs)
    # se = Reshape((1, 1, filters))(se)
    se = Lambda(lambda x: tf.reduce_mean(x, [h_axis, w_axis], keepdims=True), name='Unet_lambda_' + name)(inputs)
    se = Conv3D(reduction, kernel_size=1, use_bias=True, kernel_initializer=CONV_KERNEL_INITIALIZER, name=name + "1_conv")(se)
    # se = PReLU(shared_axes=[1, 2])(se)
    se = Activation("swish", name='Unet_swish_' + name)(se)
    se = Conv3D(filters, kernel_size=1, use_bias=True, kernel_initializer=CONV_KERNEL_INITIALIZER, name=name + "2_conv")(se)
    se = Activation("sigmoid", name='Unet_sigmoid_' + name)(se)
    return Multiply(name='Unet_mult_' + name)([inputs, se])

def se_module_transpose_3d(inputs, se_ratio=4, name=""):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    h_axis, w_axis = [2, 3] if K.image_data_format() == "channels_first" else [1, 2]

    filters = inputs.shape[channel_axis]
    # reduction = _make_divisible(filters // se_ratio, 8)
    reduction = filters // se_ratio
    # se = GlobalAveragePooling2D()(inputs)
    # se = Reshape((1, 1, filters))(se)
    se = Lambda(lambda x: tf.reduce_mean(x, [h_axis, w_axis], keepdims=True), name='Unet_lambda_transpose_' + name)(inputs)
    se = Conv3DTranspose(reduction, kernel_size=1, use_bias=True, kernel_initializer=CONV_KERNEL_INITIALIZER, name=name + "1_conv")(se)
    # se = PReLU(shared_axes=[1, 2])(se)
    se = Activation("swish", name='Unet_swish_transpose_' + name)(se)
    se = Conv3DTranspose(filters, kernel_size=1, use_bias=True, kernel_initializer=CONV_KERNEL_INITIALIZER, name=name + "2_conv")(se)
    se = Activation("sigmoid", name='Unet_sigmoid_transpose_' + name)(se)
    return Multiply(name='Unet_mult_transpose_' + name)([inputs, se])

# inputs, output_channel, stride, expand_ratio, shortcut, kernel_size, drop_rate, use_se, is_fused, is_torch_mode = nn, out, stride, expand, shortcut, kernel_size, block_drop_rate, se, is_fused, is_torch_mode
def MBConv_3d(inputs, output_channel, stride, expand_ratio, shortcut, kernel_size=3, drop_rate=0, use_se=0, is_fused=False, is_torch_mode=False, name=""):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    input_channel = inputs.shape[channel_axis]

    if is_fused and expand_ratio != 1:
        nn = conv3d_no_bias(inputs, input_channel * expand_ratio, 3, stride, padding="same", use_torch_padding=is_torch_mode, name=name + "sortcut_")
        nn = batchnorm_with_activation(nn, use_torch_eps=is_torch_mode, name=name + "sortcut_")
    elif expand_ratio != 1:
        nn = conv3d_no_bias(inputs, input_channel * expand_ratio, 1, strides=1, padding="valid", name=name + "sortcut_")
        nn = batchnorm_with_activation(nn, use_torch_eps=is_torch_mode, name=name + "sortcut_")
    else:
        nn = inputs

    if not is_fused:
        if is_torch_mode and kernel_size // 2 > 0:
            nn = keras.layers.ZeroPadding3D(padding=kernel_size // 2, name=name + "pad")(nn)
            pad = "VALID"
        else:
            pad = "SAME"
        nn = DepthwiseConv3D(kernel_size, padding=pad, strides=(stride, stride, 1), use_bias=False, depthwise_initializer=CONV_KERNEL_INITIALIZER, name=name + "MB_dw_")(nn)
        nn = batchnorm_with_activation(nn, use_torch_eps=is_torch_mode, name=name + "MB_dw_")

    if use_se:
        nn = se_module_3d(nn, se_ratio=4 * expand_ratio, name=name + "se_")

    # pw-linear
    if is_fused and expand_ratio == 1:
        nn = conv3d_no_bias(nn, output_channel, 3, strides=stride, padding="same", use_torch_padding=is_torch_mode, name=name + "fu_")
        nn = batchnorm_with_activation(nn, use_torch_eps=is_torch_mode, name=name + "fu_")
    else:
        nn = conv3d_no_bias(nn, output_channel, 1, strides=1, padding="valid", name=name + "MB_pw_")
        nn = batchnorm_with_activation(nn, use_torch_eps=is_torch_mode, activation=None, name=name + "MB_pw_")

    if shortcut:
        if drop_rate > 0:
            nn = Dropout(drop_rate, noise_shape=(None, 1, 1, 1, 1), name=name + "drop")(nn)
        return Add(name='Unet_add_' + name)([inputs, nn])
    else:
        return nn

# inputs, output_channel, stride, expand_ratio, shortcut, kernel_size, drop_rate, use_se, is_fused, is_torch_mode, name = nn, out, stride, expand, shortcut, kernel_size, block_drop_rate, se, is_fused, is_torch_mode, name
def MBConv_Transpose_3d(inputs, output_channel, stride, expand_ratio, shortcut, kernel_size=3, drop_rate=0, use_se=0, is_fused=False, is_torch_mode=False, name=""):
  channel_axis = 1 if K.image_data_format() == "channels_first" else -1
  input_channel = inputs.shape[channel_axis]

  if is_fused and expand_ratio != 1:
      nn = conv3d_no_bias_transpose(inputs, input_channel * expand_ratio, 3, stride, padding="same", use_torch_padding=is_torch_mode, name=name + "sortcut_")
      nn = batchnorm_with_activation(nn, use_torch_eps=is_torch_mode, name=name + "sortcut_")
  elif expand_ratio != 1:
      nn = conv3d_no_bias_transpose(inputs, input_channel * expand_ratio, 1, strides=1, padding="valid", name=name + "sortcut_")
      nn = batchnorm_with_activation(nn, use_torch_eps=is_torch_mode, name=name + "sortcut_")
  else:
      nn = inputs

  if not is_fused:
      if is_torch_mode and kernel_size // 2 > 0:
          nn = keras.layers.ZeroPadding2D(padding=kernel_size // 2, name=name + "pad")(nn)
          pad = "VALID"
      else:
          pad = "SAME"
      nn = conv3d_no_bias_transpose(nn, input_channel, kernel_size, stride, padding=pad, name=name + "MB_dw_")
      nn = batchnorm_with_activation(nn, use_torch_eps=is_torch_mode, name=name + "MB_dw_")

  if use_se:
      nn = se_module_transpose_3d(nn, se_ratio=4 * expand_ratio, name=name + "se_")

  # pw-linear
  if is_fused and expand_ratio == 1:
      nn = conv3d_no_bias_transpose(nn, output_channel, 3, strides=stride, padding="same", use_torch_padding=is_torch_mode, name=name + "fu_")
      nn = batchnorm_with_activation(nn, use_torch_eps=is_torch_mode, name=name + "fu_")
  else:
      nn = conv3d_no_bias_transpose(nn, output_channel, 1, strides=1, padding="valid", name=name + "MB_pw_")
      nn = batchnorm_with_activation(nn, use_torch_eps=is_torch_mode, activation=None, name=name + "MB_pw_")

  if shortcut:
      if drop_rate > 0:
          nn = Dropout(drop_rate, noise_shape=(None, 1, 1, 1, 1), name=name + "drop")(nn)
      return Add(name='Unet_add_transpose_' + name)([inputs, nn])
  else:
      return nn

#TODO: here
def efficientNetV2_3D(
  model_type,
  input_shape=(256, 256, 16, 1),
  input_tensor=None,
  num_classes=1000,
  dropout=0.2,
  first_strides=2,
  is_torch_mode=False,
  drop_connect_rate=0,
  classifier_activation="softmax",
  include_preprocessing=False,
  model_name="EfficientNetV2_3D",
  return_ouputs=False,
  kwargs=None,  # Not used, just recieving parameter
):
  model_name += f'_{model_type}'
  if isinstance(model_type, dict):  # For EfficientNetV1 configures
      model_type, blocks_config = model_type.popitem()
  else:
      blocks_config = BLOCK_CONFIGS.get(model_type.lower(), BLOCK_CONFIGS["s"])
  expands = blocks_config["expands"]
  out_channels = blocks_config["out_channels"]
  depthes = blocks_config["depthes"]
  strides = blocks_config["strides"]
  use_ses = blocks_config["use_ses"]
  first_conv_filter = blocks_config.get("first_conv_filter", out_channels[0])
  kernel_sizes = blocks_config.get("kernel_sizes", [3] * len(depthes))
  # "torch" for all V1 models
  # for V2 models, "21k" pretrained are all "tf", "imagenet" pretrained "bx" models are all "torch", ["s", "m", "l", "xl"] are "tf"
  rescale_mode = "tf"

  if input_tensor is None:
    inputs = Input(shape=input_shape)
  else:
    inputs = input_tensor

  nn = inputs
  # Normalization = keras.layers.Normalization if hasattr(keras.layers, "Normalization") else keras.layers.experimental.preprocessing.Normalization
  # mean = tf.constant([0.485, 0.456, 0.406])
  # std = (tf.constant([0.229, 0.224, 0.225])) ** 2
  # nn = Normalization(mean=mean, variance=std)(inputs)
  out_channel = _make_divisible(first_conv_filter, 8)

  nn = keras.layers.ReLU(name='Unet_relu_first')(nn)

  nn = conv3d_no_bias(nn, out_channel, 3, strides=first_strides, padding="same", use_torch_padding=is_torch_mode, name="Unet_stem_")
  nn = batchnorm_with_activation(nn, use_torch_eps=is_torch_mode, name="Unet_stem_")

  pre_out = out_channel
  global_block_id = 0
  total_blocks = sum(depthes)
  nn_sum = []
  for id, (expand, out_channel, depth, stride, se, kernel_size) in enumerate(zip(expands, out_channels, depthes, strides, use_ses, kernel_sizes)):
    out = _make_divisible(out_channel, 8)
    is_fused = True if se == 0 else False
    for block_id in range(depth):
        stride = stride if block_id == 0 else 1
        shortcut = True if out == pre_out and stride == 1 else False
        name = "Unet_stack_{}_block{}_".format(id, block_id)
        block_drop_rate = drop_connect_rate * global_block_id / total_blocks

        nn = MBConv_3d(nn, out, stride, expand, shortcut, kernel_size, block_drop_rate, se, is_fused, is_torch_mode, name=name)

        nn_sum.append(nn)
        pre_out = out
        global_block_id += 1

  global_block_id -= 1

  do_stride = None
  for id, (expand, out_channel, depth, stride_, se, kernel_size) in reversed(list(enumerate(zip(expands, out_channels, depthes, strides, use_ses, kernel_sizes)))):
    out = _make_divisible(out_channel, 8)
    is_fused = True if se == 0 else False
    for block_id in reversed(range(depth)):
      stride = stride_ if block_id == 0 else 1
      if stride != 1:
        do_stride = False
        num_stride = stride
        stride = 1
      if do_stride == True:
        stride = num_stride
        do_stride = None
      elif do_stride == False:
        do_stride = True
      shortcut = True if out == pre_out and stride == 1 else False
      name = "Unet_stack_{}_block{}_transposed_".format(id, block_id)
      block_drop_rate = drop_connect_rate * global_block_id / total_blocks

      nn = MBConv_Transpose_3d(nn, out, stride, expand, shortcut, kernel_size, block_drop_rate, se, is_fused, is_torch_mode, name=name)

      nn = Concatenate(name='Unet_' + str(global_block_id))([rev_3d_part(nn), rev_3d_part(nn_sum[global_block_id])])

      pre_out = out
      global_block_id -= 1

  nn = conv3d_no_bias_transpose(nn, out_channel, 3, strides=first_strides, padding="same", use_torch_padding=is_torch_mode, name="Unet_end_stem_1")
  nn = batchnorm_with_activation(nn, use_torch_eps=is_torch_mode, name="Unet_end_stem_")

  nn = conv3d_no_bias_transpose(nn, 1, 3, strides=1, padding="same", use_torch_padding=is_torch_mode, name="Unet_end_stem_2")

  mask     = Lambda(lambda x: tf.cast(x <= 0., tf.float32), name='Unet_mask_last')(inputs)
  mask_inv = Lambda(lambda x: tf.cast(x >  0., tf.float32), name='Unet_mask_inv_last')(inputs)

  nn = Add(name='Unet_last_add')([Multiply(name='Unet_mult_nn_mask_last')([nn, mask]), Multiply(name='Unet_mult_inputs_mask_inv')([inputs, mask_inv])])

  nn = keras.layers.ReLU(max_value=255., negative_slope=0.0, name='Unet_relu_last')(nn)

  if return_ouputs:
    return nn
  model = Model(inputs=inputs, outputs=nn, name=model_name)
  model.rescale_mode = rescale_mode
  return model

def EfficientNetV2_3D_custom(
  model_type,
  input_shape=(256, 256, 16, 1),
  first_strides=2,
  is_torch_mode=False,
  drop_connect_rate=0,
  include_preprocessing=False,
  model_name="efficientnetv2_3d_custom",
  kwargs=None,  # Not used, just recieving parameter
):
  model_name += f'_{model_type}'
  if isinstance(model_type, dict):  # For EfficientNetV1 configures
      model_type, blocks_config = model_type.popitem()
  else:
      blocks_config = BLOCK_CONFIGS.get(model_type.lower(), BLOCK_CONFIGS["s"])
  expands = blocks_config["expands"]
  out_channels = blocks_config["out_channels"]
  depthes = blocks_config["depthes"]
  strides = blocks_config["strides"]
  use_ses = blocks_config["use_ses"]
  first_conv_filter = blocks_config.get("first_conv_filter", out_channels[0])
  kernel_sizes = blocks_config.get("kernel_sizes", [3] * len(depthes))
  # "torch" for all V1 models
  # for V2 models, "21k" pretrained are all "tf", "imagenet" pretrained "bx" models are all "torch", ["s", "m", "l", "xl"] are "tf"
  rescale_mode = "tf"

  inputs = Input(shape=input_shape)
  if include_preprocessing and rescale_mode == "torch":
      channel_axis = 1 if K.image_data_format() == "channels_first" else -1
      Normalization = keras.layers.Normalization if hasattr(keras.layers, "Normalization") else keras.layers.experimental.preprocessing.Normalization
      mean = tf.constant([0.485, 0.456, 0.406]) * 255.0
      std = (tf.constant([0.229, 0.224, 0.225]) * 255.0) ** 2
      nn = Normalization(mean=mean, variance=std, axis=channel_axis)(inputs)
  elif include_preprocessing and rescale_mode == "tf":
      Rescaling = keras.layers.Rescaling if hasattr(keras.layers, "Rescaling") else keras.layers.experimental.preprocessing.Rescaling
      nn = Rescaling(scale=1.0 / 128.0, offset=-1)(inputs)
  else:
      nn = inputs

  mask     = Lambda(lambda x: tf.cast(x <= 0., tf.float32))(inputs)
  mask_inv = Lambda(lambda x: tf.cast(x >  0., tf.float32))(inputs)

  nn = keras.layers.ReLU()(nn)

  nn_2d = Lambda(lambda x: tf.reduce_sum(x[..., 0], -1, True))(nn)

  out_channel = _make_divisible(first_conv_filter, 8)
  nn_2d = conv2d_no_bias(nn_2d, out_channel, 3, strides=first_strides, padding="same", use_torch_padding=is_torch_mode, name="stem_2d_")
  nn_2d = batchnorm_with_activation(nn_2d, use_torch_eps=is_torch_mode, name="stem_2d_")

  pre_out = out_channel
  global_block_id_2d = 0
  total_blocks = sum(depthes)
  nn_sum_2d = []
  for id, (expand, out_channel, depth, stride, se, kernel_size) in enumerate(zip(expands, out_channels, depthes, strides, use_ses, kernel_sizes)):
    out = _make_divisible(out_channel, 8)
    is_fused = True if se == 0 else False
    for block_id in range(depth):
        stride = stride if block_id == 0 else 1
        shortcut = True if out == pre_out and stride == 1 else False
        name = "2d_stack_{}_block{}_".format(id, block_id)
        block_drop_rate = drop_connect_rate * global_block_id_2d / total_blocks

        nn_2d = MBConv(nn_2d, out, stride, expand, shortcut, kernel_size, block_drop_rate, se, is_fused, is_torch_mode, name=name)

        nn_sum_2d.append(nn_2d)
        pre_out = out
        global_block_id_2d += 1

  nn = conv3d_no_bias(nn, out_channel, 3, strides=first_strides, padding="same", use_torch_padding=is_torch_mode, name="stem_")
  nn = batchnorm_with_activation(nn, use_torch_eps=is_torch_mode, name="stem_")

  pre_out = out_channel
  global_block_id = 0
  total_blocks = sum(depthes)
  nn_sum = []
  for id, (expand, out_channel, depth, stride, se, kernel_size) in enumerate(zip(expands, out_channels, depthes, strides, use_ses, kernel_sizes)):
    out = _make_divisible(out_channel, 8)
    is_fused = True if se == 0 else False
    for block_id in range(depth):
        stride = stride if block_id == 0 else 1
        shortcut = True if out == pre_out and stride == 1 else False
        name = "stack_{}_block{}_".format(id, block_id)
        block_drop_rate = drop_connect_rate * global_block_id / total_blocks

        nn = MBConv_3d(nn, out, stride, expand, shortcut, kernel_size, block_drop_rate, se, is_fused, is_torch_mode, name=name)

        nn_2d_to_sum = Lambda(lambda x: tf.expand_dims(x, -1))(nn_sum_2d[global_block_id])
        nn_2d_to_sum = keras.layers.Conv3D(16, 1)(nn_2d_to_sum)
        nn_2d_to_sum = keras.layers.Permute((1,2,4,3))(nn_2d_to_sum)

        nn = Add()([nn, nn_2d_to_sum])
        nn = batchnorm_with_activation(nn, name=f'add_2d_{global_block_id}_')

        nn_sum.append(nn)
        pre_out = out
        global_block_id += 1

  global_block_id -= 1

  do_stride = None
  for id, (expand, out_channel, depth, stride_, se, kernel_size) in reversed(list(enumerate(zip(expands, out_channels, depthes, strides, use_ses, kernel_sizes)))):
    out = _make_divisible(out_channel, 8)
    is_fused = True if se == 0 else False
    for block_id in reversed(range(depth)):
      stride = stride_ if block_id == 0 else 1
      if stride != 1:
        do_stride = False
        num_stride = stride
        stride = 1
      if do_stride == True:
        stride = num_stride
        do_stride = None
      elif do_stride == False:
        do_stride = True
      shortcut = True if out == pre_out and stride == 1 else False
      name = "stack_{}_block{}_transposed_".format(id, block_id)
      block_drop_rate = drop_connect_rate * global_block_id / total_blocks

      nn = MBConv_Transpose_3d(nn, out, stride, expand, shortcut, kernel_size, block_drop_rate, se, is_fused, is_torch_mode, name=name)

      nn = Add()([nn, conv3d_no_bias(nn_sum[global_block_id], None, 1, name=f'1x1_unet_{global_block_id}_')])
      nn = batchnorm_with_activation(nn, name=f'bn_1x1_unet_{global_block_id}_')

      pre_out = out
      global_block_id -= 1

  nn = conv3d_no_bias_transpose(nn, out_channel, 3, strides=first_strides, padding="same", use_torch_padding=is_torch_mode, name="end_stem_1")
  nn = batchnorm_with_activation(nn, use_torch_eps=is_torch_mode, name="end_stem_")

  nn = conv3d_no_bias_transpose(nn, 1, 3, strides=1, padding="same", use_torch_padding=is_torch_mode, name="end_stem_2")

  nn = keras.layers.ReLU(max_value=255., negative_slope=0., threshold=.5)(nn)

  nn = Add()([Multiply()([nn, mask]), Multiply()([inputs, mask_inv])])

  model = Model(inputs=inputs, outputs=nn, name=model_name)
  return model

def EfficientNetV2_3D(model_type):
  def wrap_eff(input_shape=(256,256,16,1)):
    model = efficientNetV2_3D(model_type, input_shape)
    return model
  return wrap_eff


if __name__ == '__main__':
  from sci.generator import Generator

  input_shape=(256, 256, 16, 1)
  input_tensor=None
  num_classes=1000
  dropout=0.4
  classifier_activation="softmax"
  pretrained=None
  model_type="t"
  model_name="EfficientNetV2L"

  first_strides=2
  is_torch_mode=False
  drop_connect_rate=0
  include_preprocessing=False

  data_test = Generator('test_v2', batchsize=1, is_custom_processing=True)
  inputs = data_test[0][0]
