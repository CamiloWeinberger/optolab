import os
import argparse
import numpy as np

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('--resume', default=0, type=int, help='load best model: 1 or 0')
ap.add_argument('--batchsize', default=64, type=int, help='batch size')
ap.add_argument('--use_gpus', default='all', type=str, help='all or by id: 0,1,2 or 0 or 1 or 0,1,2,3,4,5,6,7')
ap.add_argument('--epochs', default=100, type=int, help='number of epochs')
ap.add_argument('--lr', default=1e-3, type=float, help='learning rate')
ap.add_argument('--normalize_head', default='none', type=str, help='normalize head: none, mean_std, mean_std_min_max or min_max')
ap.add_argument('--normalize_tail', default='none', type=str, help='normalize tail: none, mean_std, mean_std_min_max or min_max')
args = vars(ap.parse_args())

if args['use_gpus'] != 'all':
  if args['use_gpus'][-1] == ',':
    args['use_gpus'] = args['use_gpus'][:-1]
  if args['use_gpus'][0] == ',':
    args['use_gpus'] = args['use_gpus'][1:]
  os.environ['CUDA_VISIBLE_DEVICES'] = args['use_gpus']

import tensorflow as tf
from pyramidal.models.tf_xception import xception
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from pyramidal.generator_tf import Generator

input_shape, outputs = (134, 134, 1), 199

if not os.path.exists('saved_models'):
  os.mkdir(f'{os.path.dirname(__file__)}/saved_models')

strategy = tf.distribute.MirroredStrategy()
print('\n\nNumber of devices: {}\n\n'.format(strategy.num_replicas_in_sync))
batchsize = args['batchsize'] * strategy.num_replicas_in_sync

with strategy.scope():
  model = xception((134, 134, 1), 199)
model.compile(optimizer=Adam(lr=args['lr']), loss='mse', metrics=['mse', 'mae'])

checkpoint = ModelCheckpoint(f'{os.path.dirname(__file__)}/saved_models/{model.name}-_{{epoch:03d}}-_{{val_loss:.4f}}.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
lr_plateau = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(.2), verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=1e-10)

callbacks = [checkpoint, lr_plateau]

train = Generator('train', batchsize=strategy.num_replicas_in_sync, normalize_head=args['normalize_head'], normalize_tail=args['normalize_tail'])
values_normalize_head = train.values_normalize_head
values_normalize_tail = train.values_normalize_tail
val   = Generator('val', batchsize=strategy.num_replicas_in_sync, normalize_head=args['normalize_head'], values_normalize_head=values_normalize_head, normalize_tail=args['normalize_tail'], values_normalize_tail=values_normalize_tail)

model.fit_generator(train, epochs=args['epochs'], callbacks=callbacks, validation_data=val)

model.save(f'{os.path.dirname(__file__)}/saved_models/{model.name}_final_epoch.h5')
