from typing import Optional

import os
from os.path import dirname, basename, join, exists
import numpy as np
from glob import glob
from tqdm import tqdm
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from sci.pytorch.Generators.generator import Generator

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
    self.datapath              = datapath
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

    if self.datapath[-1] != '/':
      self.datapath += '/'

    self.paths_mat = glob(join(self.datapath, '*', '*.mat'))

    user = os.environ.get('USER')
    if user.find('mvaldi') != -1:
      data_save_data = f'/media/mvaldi-pucv-low/data/alejandro/{basename(dirname(self.datapath))}'
    elif user == 'optolab':
      data_save_data = f'/Storage1/Matias/alejandro/{basename(dirname(self.datapath))}'
    elif user.find('bizon') != -1:
      data_save_data = f'/home/bizon/Documents/matias/alejandro/{basename(dirname(self.datapath))}'
    else:
      raise Exception('Unknown user')

    self.datapath = data_save_data


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
    if len(glob(join(self.datapath, '*', '*'))) > 10300:
      return

    from scipy.io import loadmat

    for path_mat in tqdm(self.paths_mat, desc='Converting .mat to .npy'):
      path_npy = join(self.datapath, basename(dirname(path_mat)),
                      basename(path_mat).replace('.mat', f'.{self.extension}'))
      if not exists(dirname(path_npy)):
        os.makedirs(dirname(path_npy), exist_ok=True)
      data = loadmat(path_mat)
      if 'mask' in data and 'orig' in data:
        mask = data['mask']
        Y = data['orig']
        X = Y * mask

        np.save(path_npy, np.array([X[..., None], Y[..., None]]))


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
  datadir = '/Storage1/Matias/Datasetx30/'
  transform = lambda x: x.permute(3, 2, 0, 1)
  dm = DataModule(datadir, 'test', type_input='3d', type_output='3d', normalize_head='none', custom_transform_head=transform, custom_transform_tail=transform)
  dm.prepare_data()
  dm.setup(stage='predict')
  dataloader = iter(dm.predict_dataloader())
  for X in next(dataloader):
    print(X.shape)

  # test normalize = mean_std
  print('Testing DataModule with normalize = mean_std')
  dm = DataModule(datadir, 'test', type_input='3d', type_output='3d', normalize_head='mean_std', custom_transform_head=transform, custom_transform_tail=transform)
  dm.prepare_data()
  dm.setup(stage='fit')
  dm.setup(stage='predict')

  dataloader = iter(dm.predict_dataloader())
  for X in next(dataloader):
    print(X.shape)