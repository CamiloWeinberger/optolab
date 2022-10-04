import os
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('--datapath', default='Datasetx30', type=str, help='Datasetx30 or DatasetsBase25')
ap.add_argument('--load_best_model', default=0, type=int, help='load best model: 1 or 0')
ap.add_argument('--use_tf_data', default=0, type=int, help='load best model: 1 or 0')
ap.add_argument('--model', default='light', type=str, help='must be eff3d or light or light_cbam')
ap.add_argument('--model_type', default='lower', type=str, help='tiny, lower, high, t or s')
ap.add_argument('--sliding_window_approach', default=0, type=int, help='0, 1, 2, 3, ... ; 0 = dont use sliding window')
ap.add_argument('--use_gpus', default='all', type=str, help='all or by id: 0,1,2 or 0 or 1 or 0,1,2,3,4,5,6,7')
args = vars(ap.parse_args())

if args['use_gpus'] != 'all':
  if args['use_gpus'][-1] == ',':
    args['use_gpus'] = args['use_gpus'][:-1]
  if args['use_gpus'][0] == ',':
    args['use_gpus'] = args['use_gpus'][1:]
  os.environ['CUDA_VISIBLE_DEVICES'] = args['use_gpus']

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger

from sci.keras_app_3d.efficientnet_v2 import EfficientNetV2_3D
from sci.keras_app_3d.Lightweight import Lightweight
from sci.keras_app_3d.Lightweightcbam import LightweightCBAM
from sci.generator import Generator, DatasetFromSequenceClass
from sci.sliding_window import SlidingWindowApproach
from sci.keras.callbacks import Metrics

from tensorflow_addons.optimizers import LazyAdam
from sci.losses import mae, ssim_loss_metric

models = {'eff3d': EfficientNetV2_3D, 'light': Lightweight, 'light_cbam': LightweightCBAM}
model_type = ['ultratiny64', 'ultratiny', 'tiny', 'ultralower', 'lower', 'lowerloop', 'high', 't', 's']

assert args['model'] in models
assert args['model_type'] in model_type
assert args['datapath'] in ['Datasetx30', 'DatasetsBase25']

epochs = 60
load_best_model = args['load_best_model'] > 0

if not os.path.exists('models'):
  os.mkdir(f'{os.path.dirname(__file__)}/models')

strategy = tf.distribute.MirroredStrategy()
print('\n\nModel Name: {}_{}\n\nNumber of devices: {}\n\n'.format(args['model'], args['model_type'] + '_sw' * int(args['sliding_window_approach'] > 0), strategy.num_replicas_in_sync))
batchsize = strategy.num_replicas_in_sync

with strategy.scope():
  shape_data   = Generator('val', datapath=args['datapath'], batchsize=1)[0][0][0].shape
  if args['sliding_window_approach'] == 0:
    model = models[args['model']](args['model_type'])(shape_data)
  elif args['sliding_window_approach'] > 0:
    sw = SlidingWindowApproach(args['sliding_window_approach'])

    inputs, multi_img = sw.Header(shape_data)
    multi_img_shape = multi_img.shape[2:]
    backbone = models[args['model']](args['model_type'])(multi_img_shape)

    end_backbone = sw.HandleBackbone(backbone, multi_img)
    output = sw.Tail(end_backbone)

    model = Model(inputs=inputs, outputs=output, name=backbone.name + '_sw')

  else:
    raise Exception('sliding window arg must be positive!!')

  print(model.name)

  if args['use_tf_data'] > 0:
    data_train = Generator(datapath=args['datapath'], batchsize=1)
    data_val   = Generator('val', datapath=args['datapath'], batchsize=1)
    data_test  = Generator('test_v2', datapath=args['datapath'], batchsize=1)

    training_steps   = len(data_train)
    validation_steps = len(data_val)

    training = DatasetFromSequenceClass(data_train, training_steps, epochs, batchsize, dims=[256, 256, 16, 1])
    validation = DatasetFromSequenceClass(data_val, validation_steps, epochs, batchsize, dims=[256, 256, 16, 1])
  else:
    data_train = Generator(batchsize=batchsize, datapath=args['datapath'])
    data_val   = Generator('val', datapath=args['datapath'], batchsize=batchsize)
    data_test  = Generator('test_v2', datapath=args['datapath'], batchsize=batchsize)

  assert len(data_train) != 0 and len(data_val) != 0 and len(data_test) != 0

  optimizer = LazyAdam(1e-3)
  model.compile(optimizer, loss=mae, metrics=[ssim_loss_metric])

  if load_best_model:
    from glob import glob
    def key(value):
      return float(value.split('-_')[-1].replace('.h5', ''))
    path_weights = sorted(glob(f'./models/{model.name}*.h5'), key=key)
    if len(path_weights) > 0:
      path_weight = path_weights[-1]
      model.load_weights(path_weight)

  callbacks = [ModelCheckpoint(f'./models/{model.name}-_{{epoch:03d}}-_{{loss:.4f}}-_{{val_loss:.4f}}.h5',
                               save_best_only=True, save_weights_only=True, mode='min', verbose=1),
               Metrics(data_test),
               ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=3, min_lr=1e-8, verbose=1),
               CSVLogger(f'training_{model.name}.log'),
              ]

if args['use_tf_data'] > 0:
  model.fit(training,
      steps_per_epoch=training_steps,
      validation_data=validation,
      validation_steps=validation_steps,
      epochs=epochs,
      callbacks=callbacks,
      verbose=1)
else:
  model.fit(data_train, validation_data=data_val, epochs=epochs, callbacks=callbacks)

