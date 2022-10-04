import os
import numpy as np
from glob import glob
from scipy.io import loadmat, savemat
from pyramidal.datapath import datapath
from pyramidal.generators.generator_pt import Generator
from pyramidal.models.pt_wfnet import WFNet
from pyramidal.models.pt_gc_vit import GC_VIT_Lightning, GC_VIT_Lightning_xxtiny
from pytorch_lightning import Trainer
from tqdm import tqdm

import argparse
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('--model', default='GC_VIT', type=str, help='Model to train')
ap.add_argument('--path_model', default='GC_VIT-v=00-epoch=149-val_loss=2.39.pt', type=str, help='Path to model')
ap.add_argument('--use_gpus', default='0', type=str, help='all or by id: 0,1,2 or 0 or 1 or 0,1,2,3,4,5,6,7')
ap.add_argument('--normalize_head', default='none', type=str, help='normalize head: none, mean_std, mean_std_min_max or min_max')
ap.add_argument('--normalize_tail', default='none', type=str, help='normalize tail: none, mean_std, mean_std_min_max or min_max')
args = vars(ap.parse_args())

if args['use_gpus'] != 'all':
  if args['use_gpus'][-1] == ',':
    args['use_gpus'] = args['use_gpus'][:-1]
  if args['use_gpus'][0] == ',':
    args['use_gpus'] = args['use_gpus'][1:]
  os.environ['CUDA_VISIBLE_DEVICES'] = args['use_gpus']
else:
  import torch
  args['use_gpus'] = ','.join(str(v) for v in range(torch.cuda.device_count()))
  os.environ['CUDA_VISIBLE_DEVICES'] = args['use_gpus']
num_gpus = len(args['use_gpus'].split(','))

import torch

def main():
  if args['model'] == 'WFNet':
    model_to_use = WFNet
    is_gcvit = False
    name_model = 'WFNet'
  elif args['model'] == 'GC_VIT' != -1:
    model_to_use = GC_VIT_Lightning
    is_gcvit = True
    name_model = 'GC_VIT'
  elif args['model'] == 'GC_VIT_XXTINY' != -1:
    model_to_use = GC_VIT_Lightning_xxtiny
    is_gcvit = True
    name_model = 'GC_VIT_XXTINY'

  paths = np.array(sorted(glob(f'{datapath[:-4]}_test/*.mat', recursive=False)))
  assert len(paths) == 29, f'len(paths) = {len(paths)}'
  data_flow = Generator(dataset='train', batchsize=1, normalize_head='mean_std', is_gcvit=is_gcvit)

  # show dir where test is saved
  print(f'\n\nDir where test is saved:\n{os.path.join(os.path.dirname(paths[0]), "tested", name_model)}\n\n')

  path_weight = f"{os.path.dirname(os.path.abspath(__file__))}/saved_models/{args['path_model']}"
  if path_weight.endswith('.pt'):
    model = torch.jit.load(path_weight)
  else:
    model = model_to_use.load_from_checkpoint(path_weight)
  print(f'Loaded model from {path_weight}')
  # mode eval and cuda if available
  model.eval()
  model.cuda()

  len_paths = len(glob(f'{datapath}/test/*.npy'))
  pbar = tqdm(paths, total=len_paths)
  for path in paths:
    pbar.desc = f'Processing {path}'
    mat = loadmat(path)
    if 'X_s' in mat and 'Y_kl' in mat:
      X = np.moveaxis(mat['X_s'], -1, 0)
      Y = mat['Y_kl']
      prediction = []
      for index, [x, y] in enumerate(zip(X, Y)):
        x_norm, _ = data_flow.__getitem__(0, x[None, ..., None], 0.)
        x_norm = x_norm.cuda()
        y_pred = model(x_norm)
        prediction.append(y_pred.detach().cpu().numpy())
        pbar.set_description(f"Predicting {index+1}/{len(X)}")
        pbar.update(1)

      prediction = np.array(prediction)
      # save mat file with prediction and another variables in tested folder
      save_path = os.path.join(os.path.dirname(path), 'tested', name_model, os.path.basename(path))
      os.makedirs(os.path.dirname(save_path), exist_ok=True)
      mat['Y_kl_pred'] = prediction
      savemat(save_path, mat)

    else:
      print(f'{path} not found')

if __name__ == '__main__':
  main()