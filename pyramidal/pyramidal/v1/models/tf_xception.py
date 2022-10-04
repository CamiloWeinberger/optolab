from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Reshape, Input

def xception(input_shape, outputs, dropout=0.5):
  inputs = Input(input_shape)
  base_model = Xception(include_top=False, input_tensor=inputs, weights=None)
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dropout(dropout)(x)
  x = Dense(outputs)(x)
  model = Model(inputs=inputs, outputs=x, name='xception')
  return model

if __name__ == '__main__':
  model = xception((134, 134, 1), 199)
  model.summary()
