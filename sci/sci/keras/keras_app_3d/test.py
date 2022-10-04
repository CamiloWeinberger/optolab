import os

import numpy as np
from tqdm import tqdm

from scipy.io import savemat
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity   as compare_ssim


is_main_process = False

def get_test(model, data_test):
  data_test.return_name = True
  outputs     = [model(val_data[0]).numpy() for val_data in tqdm(data_test, desc='Obteniendo predicciones...')]

  all_psnr     = 0
  all_ssim     = 0
  length_data  = 0
  for num, [cube_ins, cube_outs] in tqdm(enumerate(zip(data_test, outputs)), desc='Extrayendo metricas...', total=len(data_test)):
    length_data += cube_outs.shape[0]
    name_cube = os.path.basename(cube_ins[-1][0])

    cube_in  = np.int32(cube_ins[1][0, ..., 0])
    cube_out = np.int32(cube_outs[0, ..., 0])

    psnr = []
    ssim = []
    for channel in range(cube_in.shape[-1]):
      psnr += [compare_psnr(cube_in[..., channel], cube_out[..., channel], data_range=255.)]
      ssim += [compare_ssim(cube_in[..., channel], cube_out[..., channel], data_range=255.)]

    all_psnr += np.mean(psnr)
    all_ssim += np.mean(ssim)

    if name_cube.find('chart.mat') != -1 and is_main_process:
      if args['save_results']:
        savemat(f'{home_path}/Escritorio/results/{model.name}_{os.path.basename(weight)}/{name_cube}', {'orig': cube_in, 'pred': cube_out})
      print(f'psnr of chart: {psnr}')

  all_psnr_ = all_psnr / length_data
  all_ssim_ = all_ssim / length_data

  print(f'\nmodel: {model.name}\n===> PSNR: {all_psnr_:.4f}\n===> SSIM: {all_ssim_ :.6f}\n')

if __name__ == '__main__':
  import argparse
  from glob import glob

  # construct the argument parse and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument('--datapath', default='Datasetx30', type=str, help='Datasetx30 or DatasetsBase25')
  ap.add_argument('--use_gpus', default='all', type=str, help='all or by id: 0,1,2 or 0 or 1 or 0,1,2,3,4,5,6,7')
  ap.add_argument('--save_results', default=1, type=int, help='save results')
  args = vars(ap.parse_args())

  args['save_results'] = args['save_results'] > 0

  if args['use_gpus'] != 'all':
    if args['use_gpus'][-1] == ',':
      args['use_gpus'] = args['use_gpus'][:-1]
    if args['use_gpus'][0] == ',':
      args['use_gpus'] = args['use_gpus'][1:]
    os.environ['CUDA_VISIBLE_DEVICES'] = args['use_gpus']

  import tensorflow as tf
  from os import path as osp
  from sci.sliding_window import SlidingWindowApproach
  from tensorflow.keras.models import Model
  from sci.generator import Generator
  from sci.keras_app_3d.Lightweight import Lightweight
  from sci.keras_app_3d.efficientnet_v2 import EfficientNetV2_3D

  from tensorflow.keras import backend as K

  is_main_process = True

  assert args['datapath'] in ['Datasetx30', 'DatasetsBase25']

  models = [EfficientNetV2_3D, Lightweight]
  models = {model.__name__: model for model in models}
  num2name_model = {num: name_model for num, name_model in enumerate(models.keys())}
  nums_models = [num for num, _ in enumerate(models.keys())]

  path_weights = glob(f'{os.path.dirname(__file__)}/models/*')

  strategy = tf.distribute.MirroredStrategy()
  print('\n\nNumber of devices: {}\n\n'.format(strategy.num_replicas_in_sync))

  for weight in reversed(path_weights):

    name_model = osp.basename(weight).split('-_')[0]

    use_sw = False
    if name_model.find('_sw') != -1:
      use_sw = True
      name_model = name_model[:-3]

    model_type = name_model.split('_')[-1]
    name_model = name_model[:-1 - len(model_type)]

    with strategy.scope():
      shape_data   = Generator('val', datapath=args['datapath'], batchsize=1)[0][0][0].shape
      if use_sw:
        sw = SlidingWindowApproach(args['sliding_window_approach'])

        inputs, multi_img = sw.Header(shape_data)
        multi_img_shape = multi_img.shape[2:]
        backbone = models[args['model']](args['model_type'])(multi_img_shape)
        end_backbone = sw.HandleBackbone(backbone, multi_img)
        output = sw.Tail(end_backbone)

        model = Model(inputs=inputs, outputs=output, name=backbone.name + '_sw')
      else:
        model = models[name_model](model_type)(shape_data)
      model.load_weights(weight)

    if args['save_results']:
      home_path = os.path.expanduser('~')
      if not os.path.exists(f'{home_path}/results'):
        os.mkdir(f'{home_path}/results')
      if not os.path.exists(f'{home_path}/results/{model.name}_{os.path.basename(weight)}'):
        os.mkdir(f'{home_path}/results/{model.name}_{os.path.basename(weight)}')

    data_test   = Generator('test_v2', datapath=args['datapath'], batchsize=1, return_name=True)
    get_test(model, data_test)
    # print(f'\ndataset with {len(data_test)} cubes')

    K.clear_session()



'''model: EfficientNetV2_3D_s
===> PSNR: 36.6479
===> SSIM: 0.973537

model: EfficientNetV2_3D_s
===> PSNR: 36.6578
===> SSIM: 0.973985

model: Lightweight_lower
===> PSNR: 36.8243
===> SSIM: 0.953660

model: revsci
===> PSNR: 38.77
===> SSIM: 0.97
===> SAM : 0.11

'''

