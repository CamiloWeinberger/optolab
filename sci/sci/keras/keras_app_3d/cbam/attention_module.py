from tensorflow.keras import layers
from tensorflow.keras import backend as K

def cbam_block(cbam_feature, ratio=8):
  """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
  As described in https://arxiv.org/abs/1807.06521.
  """

  cbam_feature = channel_attention(cbam_feature, ratio)
  cbam_feature = spatial_attention(cbam_feature)
  return cbam_feature

def channel_attention(input_feature, ratio=8):

  channel_axis = -1
  channel = input_feature.shape[channel_axis]

  shared_layer_one = layers.Dense(channel//ratio,
               activation='relu',
               kernel_initializer='he_normal',
               use_bias=True,
               bias_initializer='zeros')
  shared_layer_two = layers.Dense(channel,
               kernel_initializer='he_normal',
               use_bias=True,
               bias_initializer='zeros')

  avg_pool = layers.GlobalAveragePooling3D()(input_feature)
  avg_pool = layers.Reshape((1,1,1,channel))(avg_pool)
  avg_pool = shared_layer_one(avg_pool)
  avg_pool = layers.ReLU(max_value=6., negative_slope=3.)(avg_pool)
  avg_pool = shared_layer_two(avg_pool)
  avg_pool = layers.ReLU(max_value=3., negative_slope=3.)(avg_pool)


  max_pool = layers.GlobalMaxPooling3D()(input_feature)
  max_pool = layers.Reshape((1,1,1,channel))(max_pool)
  max_pool = shared_layer_one(max_pool)
  max_pool = layers.ReLU(max_value=3., negative_slope=3.)(max_pool)
  max_pool = shared_layer_two(max_pool)
  max_pool = layers.ReLU(max_value=3., negative_slope=3.)(max_pool)

  cbam_feature = layers.Add()([avg_pool,max_pool])
  cbam_feature = layers.ReLU()(cbam_feature)
  cbam_feature = layers.Activation('sigmoid')(cbam_feature)

  return layers.Multiply()([input_feature, cbam_feature])

def spatial_attention(input_feature):
  kernel_size = 7

  cbam_feature = input_feature

  avg_pool = layers.Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))(cbam_feature)
  max_pool = layers.Lambda(lambda x: K.max (x, axis=-1, keepdims=True))(cbam_feature)
  concat = layers.Concatenate(-1)([avg_pool, max_pool])
  assert concat.shape[-1] == 2
  cbam_feature = layers.Conv3D(filters=1,
          kernel_size=kernel_size,
          strides=1,
          padding='same',
          activation='relu',
          kernel_initializer='he_normal',
          use_bias=False)(concat)

  cbam_feature = layers.Activation('sigmoid')(cbam_feature)

  return layers.Multiply()([input_feature, cbam_feature])