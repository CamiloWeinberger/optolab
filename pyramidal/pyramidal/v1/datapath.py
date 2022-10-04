import os
import numpy as np

if os.path.expanduser('~').find('mvaldi') != -1:
  datapath = '/home/mlflow/Datasets_phase2pyr_May_2022_npy'
elif os.path.expanduser('~').find('optolab') != -1:
  datapath = '/home/optolab/Desktop/Camilo/Datasets_phase2pyr_May_2022_npy'
elif os.path.expanduser('~').find('bizon') != -1:
  datapath = '/home/bizon/Documents/matias/Camilo/Datasets_phase2pyr_May_2022_npy'
else:
  raise Exception('Unknown user')

if not os.path.exists(f'{datapath}/') or not os.path.exists(f'{datapath}/trainval/') or not os.path.exists(f'{datapath}/test/'):
  # generate data, .mat files to npy files
  from glob import glob
  from scipy.io import loadmat
  from tqdm import tqdm

  paths = np.array(sorted(glob(f'{datapath[:-4]}/*')))

  def save_npy(paths, dataset):
    if not os.path.exists(f'{datapath}/{dataset}/'):
      os.makedirs(f'{datapath}/{dataset}/')
    for path in tqdm(paths):
      mat = loadmat(path)
      if 'X_s' in mat and 'Y_kl' in mat:
        X = np.moveaxis(mat['X_s'], -1, 0)
        Y = mat['Y_kl']
        name = os.path.basename(path)[:-4]
        for index, [x, y] in enumerate(zip(X, Y)):
          np.save(f'{datapath}/{dataset}/{name}{index:03d}.npy', np.array([x, y], dtype=object))
      else:
        print(f'{path} not found')

  # genereate train/val/test
  save_npy(paths, 'trainval') # train

  # generate test
  paths = np.array(sorted(glob(f'{datapath[:-4]}_test/*.mat')))
  save_npy(paths, 'test') # test

"""generate data split for the training/val in same directory dataset"""
if os.path.exists(f'{os.path.dirname(os.path.abspath(__file__))}/parameters/data_split_npy.npy'):
  data_split = np.load(f'{os.path.dirname(os.path.abspath(__file__))}/parameters/data_split_npy.npy', allow_pickle=True)
else:
  #create data split with train_test_split from sklearn (train:val = 0.8:0.2)
  from sklearn.model_selection import train_test_split
  import numpy as np
  from glob import glob
  data_split = train_test_split(np.arange(len(sorted(glob(f'{datapath}/trainval/*')))), test_size=0.2, random_state=42)
  if not os.path.exists(f'{os.path.dirname(os.path.abspath(__file__))}/parameters'):
    os.mkdir(f'{os.path.dirname(os.path.abspath(__file__))}/parameters')
  np.save(f'{os.path.dirname(os.path.abspath(__file__))}/parameters/data_split_npy.npy', data_split)