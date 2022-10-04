from typing import Optional

import os
from os.path import dirname, basename, join, exists
import numpy as np
from glob import glob
from tqdm import tqdm
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from pyramidal.v2.Generators.generator import Generator
from pyramidal.v2.Utils.handle_datapath import handle_datapath

class DataModule(pl.LightningDataModule):
  def __init__(
    self,
    datapath:              str,
    name_dataset:          str,
    data_train:            Optional[str]      = 'train',
    data_val:              Optional[str]      = 'val',
    data_test:             Optional[str]      = 'test',
    data_predict:          Optional[str]      = 'test',
    type_input:            Optional[str]      = 'image',
    type_output:           Optional[str]      = 'regression',
    normalize_head:        Optional[str]      = 'mean_std',
    normalize_tail:        Optional[str]      = 'none',
    custom_transform_head: Optional[callable] = None,
    custom_transform_tail: Optional[callable] = None,
    func_load_data:        Optional[callable] = lambda x: np.load(x, allow_pickle=True),
    batch_size:            Optional[int]      = 16,
    extension:             Optional[str]      = 'npy',
    ) -> None:
    super().__init__()
    self.name_dataset          = name_dataset
    self.data_train            = data_train
    self.data_val              = data_val
    self.data_test             = data_test
    self.data_predict          = data_predict
    self.type_input            = type_input
    self.type_output           = type_output
    self.normalize_head        = normalize_head
    self.normalize_tail        = normalize_tail
    self.custom_transform_head = custom_transform_head
    self.custom_transform_tail = custom_transform_tail
    self.func_load_data        = func_load_data
    self.batch_size            = batch_size
    self.extension             = extension

    self.datapath = handle_datapath(datapath)


  def prepare_data(self):
    r"""Download data if needed, and do any other preprocessing.
    This is called only from a single GPU in distributed training
    (so do not use it to assign state (self.x = y)).

    You can also convert .mat to .npy here

    Paths to use:
      self.datapath     = '/path/to/data' (str) (required)
      self.data_train   = 'train'         (str) (optional)
      self.data_val     = 'val'           (str) (optional)
      self.data_test    = 'test'          (str) (optional)
      self.data_predict = 'predict'       (str) (optional)

    Where to save:
      self.datapath/self.data_train
      self.datapath/self.data_val
      self.datapath/self.data_test
      self.datapath/self.data_predict
    """
    import numpy as np

    datapath = self.datapath
    target_key = 'Y_kl'
    multiplier = 1.
    if datapath.find('500') != -1:
      target_key = 'Y_z'
      multiplier = 700 / (2 * np.pi)

    if not os.path.exists(f'{datapath}/{self.data_train}/') or not os.path.exists(f'{datapath}/{self.data_val}/') or not os.path.exists(f'{datapath}/{self.data_test}/'):
      # generate data, .mat files to npy files
      from scipy.io import loadmat
      from tqdm import tqdm
      from shutil import move

      paths = np.array(sorted(glob(f'{datapath[:-4]}/*')))

      def save_npy(paths, dataset):
        if not os.path.exists(f'{datapath}/{dataset}/'):
          os.makedirs(f'{datapath}/{dataset}/')
        for path in tqdm(paths):
          mat = loadmat(path)
          if 'X_s' in mat and target_key in mat:
            X = np.moveaxis(mat['X_s'], -1, 0)
            Y = mat[target_key] * multiplier
            name = os.path.basename(path)[:-4]
            for index, [x, y] in enumerate(zip(X, Y)):
              np.save(f'{datapath}/{dataset}/{name}{index:03d}.npy', np.array([x[..., None], y], dtype=object))
          else:
            print(f'{path} not found')

      # genereate train/val/test
      save_npy(paths, 'trainval') # train
      from sklearn.model_selection import train_test_split
      paths_trainval = sorted(glob(f'{datapath}/trainval/*'))
      data_split = train_test_split(np.arange(len(paths_trainval)), test_size=0.2, random_state=42)
      if not os.path.exists(f'{datapath}/{self.data_train}/'):
        os.makedirs(f'{datapath}/{self.data_train}/')
      if not os.path.exists(f'{datapath}/{self.data_val}/'):
        os.makedirs(f'{datapath}/{self.data_val}/')
      for indexes, dataset in zip(data_split, [self.data_train, self.data_val]):
        for index in indexes:
          move(paths_trainval[index], f'{datapath}/{dataset}/{basename(paths_trainval[index])}')


      # generate test
      paths = np.array(sorted(glob(f'{datapath[:-4]}_test/*.mat')))
      save_npy(paths, self.data_test) # test


  def setup(self, stage: Optional[str] = None):
    # Assign train/val datasets for use in dataloaders
    if stage == 'fit' or stage is None:
      self.datamodule_train = Generator(
        self.datapath,
        self.data_train,
        self.extension,
        self.type_input,
        self.type_output,
        normalize_head=self.normalize_head,
        norm_by_channel_head=True,
        normalize_tail=self.normalize_tail,
        norm_by_channel_tail=True,
        custom_transform_head=self.custom_transform_head,
        custom_transform_tail=self.custom_transform_tail,
        func_load_data=self.func_load_data,
        name_dataset=self.name_dataset)
      self.datamodule_val   = Generator(
        self.datapath,
        self.data_val,
        self.extension,
        self.type_input,
        self.type_output,
        normalize_head=self.normalize_head,
        norm_by_channel_head=True,
        normalize_tail=self.normalize_tail,
        norm_by_channel_tail=True,
        custom_transform_head=self.custom_transform_head,
        custom_transform_tail=self.custom_transform_tail,
        func_load_data=self.func_load_data,
        name_dataset=self.name_dataset)

    # Assign test dataset for use in dataloader(s)
    if stage == 'test' or stage is None:
      self.datamodule_test = Generator(
        self.datapath,
        self.data_test,
        self.extension,
        self.type_input,
        self.type_output,
        normalize_head=self.normalize_head,
        norm_by_channel_head=True,
        normalize_tail=self.normalize_tail,
        norm_by_channel_tail=True,
        custom_transform_head=self.custom_transform_head,
        custom_transform_tail=self.custom_transform_tail,
        func_load_data=self.func_load_data,
        name_dataset=self.name_dataset)

    if stage == 'predict' or stage is None:
      self.datamodule_predict = Generator(
        self.datapath,
        self.data_predict,
        self.extension,
        self.type_input,
        self.type_output,
        normalize_head=self.normalize_head,
        norm_by_channel_head=True,
        normalize_tail=self.normalize_tail,
        norm_by_channel_tail=True,
        custom_transform_head=self.custom_transform_head,
        custom_transform_tail=self.custom_transform_tail,
        func_load_data=self.func_load_data,
        name_dataset=self.name_dataset)

  def train_dataloader(self):
    return DataLoader(self.datamodule_train, batch_size=self.batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)

  def val_dataloader(self):
    return DataLoader(self.datamodule_val, batch_size=self.batch_size, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)

  def test_dataloader(self):
    return DataLoader(self.datamodule_test, batch_size=self.batch_size, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)

  def predict_dataloader(self):
    return DataLoader(self.datamodule_predict, batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=True)

if __name__ == '__main__':
  print('Testing DataModule')
  # permute from (H, W, S, C) to (C, S, H, W)
  datadir = ''
  transform = lambda x: x.permute(3, 2, 0, 1)
  dm = DataModule(datadir, 'test', normalize_head='none', custom_transform_head=transform, custom_transform_tail=transform)
  dm.prepare_data()
  dm.setup(stage='predict')
  dataloader = iter(dm.predict_dataloader())
  for X in next(dataloader):
    print(X.shape)

  # test normalize = mean_std
  print('Testing DataModule with normalize = mean_std')
  dm = DataModule(datadir, 'test', normalize_head='mean_std', custom_transform_head=transform, custom_transform_tail=transform)
  dm.prepare_data()
  dm.setup(stage='fit')
  dm.setup(stage='predict')

  dataloader = iter(dm.predict_dataloader())
  for X in next(dataloader):
    print(X.shape)