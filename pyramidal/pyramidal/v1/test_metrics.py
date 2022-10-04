import os
import numpy as np
from glob import glob
from scipy.io import loadmat, savemat
from pyramidal.datapath import datapath
from pyramidal.generators.generator_pt import Generator, data_loader
from pyramidal.models.pt_wfnet import WFNet
from pyramidal.models.pt_gc_vit import GC_VIT_Lightning, GC_VIT_Lightning_xxtiny
from pytorch_lightning import Trainer
from pytorch_lightning.strategies.ddp import DDPStrategy
import pandas as pd

import argparse
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('--model', default='GC_VIT_XXTINY', type=str, help='Model to train')
ap.add_argument('--path_model', default='GC_VIT_XXTINY-v=02-epoch=143-val_loss=2.34.ckpt', type=str, help='Path to model')
ap.add_argument('--normalize_head', default='mean_std', type=str, help='normalize head: none, mean_std, mean_std_min_max or min_max')
ap.add_argument('--normalize_tail', default='none', type=str, help='normalize tail: none, mean_std, mean_std_min_max or min_max')
args = vars(ap.parse_args())


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

  path_weight = f"{os.path.dirname(os.path.abspath(__file__))}/saved_models/{args['path_model']}"
  model = model_to_use.load_from_checkpoint(path_weight)
  print(f'Loaded model from {path_weight}')

  num_devices_to_use = torch.cuda.device_count()
  assert num_devices_to_use > 0, 'No GPU found'
  if num_devices_to_use > 5:
    num_devices_to_use = 5
  elif num_devices_to_use > 1:
    num_devices_to_use = 2
  else:
    num_devices_to_use = 1

  trainer = Trainer(accelerator='gpu', devices=num_devices_to_use)

  train = Generator('train', batchsize=1,
                    normalize_head=args['normalize_head'],
                    normalize_tail=args['normalize_tail'],
                    is_gcvit=is_gcvit,
                    )

  values_normalize_head = train.values_normalize_head
  values_normalize_tail = train.values_normalize_tail

  val   = Generator('val',   batchsize=1,
                    normalize_head=args['normalize_head'],
                    values_normalize_head=values_normalize_head,
                    normalize_tail=args['normalize_tail'],
                    values_normalize_tail=values_normalize_tail,
                    is_gcvit=is_gcvit,
                    )

  test = Generator('test', batchsize=1,
                  normalize_head=args['normalize_head'],
                  values_normalize_head=values_normalize_head,
                  normalize_tail=args['normalize_tail'],
                  values_normalize_tail=values_normalize_tail,
                  is_gcvit=is_gcvit,
                  )

  train_data_loader = data_loader(train, batchsize=10, shuffle=False,
                                  num_workers=1)
  val_data_loader   = data_loader(val,   batchsize=10, shuffle=False,
                                  num_workers=1)
  test_data_loader  = data_loader(test,  batchsize=10, shuffle=False,
                                  num_workers=1)

  # do test with trainer.test(model, dataloaders) and save results in a file MAT and CSV
  dir_path_save = f'{datapath[:-4]}_test/'
  for dataloader, name_dataloader in zip([train_data_loader, val_data_loader, test_data_loader], ['train', 'val', 'test']):
    print(f'Running {name_dataloader}...')
    results = trainer.test(model, dataloaders=dataloader)
    print(f'Finished {name_dataloader}!')

    # save results
    path_save = f'{os.path.dirname(dir_path_save)}/tested/{name_model}/{name_dataloader}.mat'
    if not os.path.exists(os.path.dirname(path_save)):
      os.makedirs(os.path.dirname(path_save))
    print(f'Saving results in {path_save}')
    savemat(path_save, results[0])

    # save results in CSV
    path_save = f'{os.path.dirname(dir_path_save)}/tested/{name_model}/{name_dataloader}.csv'
    if not os.path.exists(os.path.dirname(path_save)):
      os.makedirs(os.path.dirname(path_save))
    print(f'Saving results in {path_save}')
    # values of results are in the form of a list of dicts
    dict_csv = {k: [v] for k, v in results[0].items()}
    df = pd.DataFrame(dict_csv)
    df.to_csv(path_save, index=False)



if __name__ == '__main__':
  main()