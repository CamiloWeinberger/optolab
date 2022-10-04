from typing import Optional

import os
from os.path import join
import numpy as np
from tqdm import tqdm
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from pyramidal.pytorch.Generators.generator import Generator
from pyramidal.pytorch.Utils.datapath import datapath

class DataModule(pl.LightningDataModule):
  def __init__(
    self,
    datavariant:    str,
    data_train:     Optional[str]      = 'train',
    data_val:       Optional[str]      = 'val',
    data_test:      Optional[str]      = 'test',
    data_predict:   Optional[str]      = 'test',
    normalize_head: Optional[str]      = 'mean_std',
    normalize_tail: Optional[str]      = 'none',
    func_load_data: Optional[callable] = lambda x: np.load(x, allow_pickle=True),
    batch_size:     Optional[int]      = 16,
    extension:      Optional[str]      = 'npy',
    is_half:        Optional[bool]     = True,
    ) -> None:
    r"""
    Args:
      datavariant:    str, name of the dataset
      data_train:     str, name of the training dataset
      data_val:       str, name of the validation dataset
      data_test:      str, name of the testing dataset
      data_predict:   str, name of the prediction dataset
      normalize_head: str, normalization method for the head
      normalize_tail: str, normalization method for the tail
      func_load_data: callable, function to load the data
      batch_size:     int, batch size
      extension:      str, extension of the data
      is_half:        bool, whether to use half precision
    """
    super().__init__()
    self.datavariant    = datavariant
    self.data_train     = data_train
    self.data_val       = data_val
    self.data_test      = data_test
    self.data_predict   = data_predict
    self.normalize_head = normalize_head
    self.normalize_tail = normalize_tail
    self.func_load_data = func_load_data
    self.batch_size     = batch_size
    self.extension      = extension
    self.is_half        = is_half


  def prepare_data(self):
    r"""Download data if needed, and do any other preprocessing.
    This is called only from a single GPU in distributed training
    (so do not use it to assign state (self.x = y)).

    You can also convert .mat to .npy here

    Paths to use:
      self.datavariant  = '/path/to/data' (str) (required)
      self.data_train   = 'train'         (str) (optional)
      self.data_val     = 'val'           (str) (optional)
      self.data_test    = 'test'          (str) (optional)
      self.data_predict = 'predict'       (str) (optional)

    Where to save:
      self.datavariant/self.data_train
      self.datavariant/self.data_val
      self.datavariant/self.data_test
      self.datavariant/self.data_predict
    """
    from glob import glob
    if (len(glob(join(datapath, f'{self.data_train}', '*'))) > 50000 and
        len(glob(join(datapath, f'{self.data_val}', '*'))) > 9000 and
        len(glob(join(datapath, f'{self.data_test}', '*'))) > 9000):
       return


    # Download data
    import pysftp
    from shutil import move, rmtree, copyfile
    from os.path import basename
    with open(glob('*.server')[0], 'r') as f:
      host = f.readline().strip()
    os.makedirs(datapath, exist_ok=True)
    remote_dir = 'database/Camilo/Datasets_phase2pyr_May_2022_500modes'
    if os.path.exists(join(datapath, remote_dir.split('/')[0])):
      rmtree(join(datapath, remote_dir.split('/')[0]))
    # files = sftp.listdir_attr("database/Camilo")
    # for f in files:
    #   print(f)
    print('Downloading data...')
    with pysftp.Connection(host, username='optolab', password='seldon') as sftp:
      # download from directory /database/camilo/Datasets_phase2pyr_May_2022_500modes
      sftp.get_r(remote_dir, datapath)

    # Move files to correct location
    for f in tqdm(glob(join(datapath, remote_dir, '*')), desc='Moving files to correct location'):
      move(f, join(datapath, basename(f)))

    rmtree(join(datapath, remote_dir.split('/')[0]))

    # Compile .mat and convert to .npy
    from scipy.io import loadmat

    os.makedirs(f'{datapath}/trainval', exist_ok=True)
    for f in tqdm(glob(join(datapath, '*.mat')), desc='Converting .mat to .npy'):
      mat = loadmat(f)
      # depermute mat['X_s'] from H, W, B to B, H, W
      X = mat['X_s'].transpose(2, 0, 1)
      for index in range(mat['Y_z'].shape[0]):
        data = np.array([X[index], mat['Y_z'][index]], dtype=object)
        np.save(join(datapath, 'trainval', f'{basename(f)}_{index:05d}.npy'), data)

    # Split trainval into self.data_train and self.data_val
    from sklearn.model_selection import train_test_split
    os.makedirs(f'{datapath}/{self.data_train}', exist_ok=True)
    os.makedirs(f'{datapath}/{self.data_val}', exist_ok=True)
    os.makedirs(f'{datapath}/{self.data_test}', exist_ok=True)

    # random state for reproducibility
    train, val = train_test_split(glob(join(datapath, 'trainval', '*')), test_size=0.2, random_state=42)
    for f in tqdm(train, desc='Moving train files'):
      move(f, join(datapath, self.data_train, basename(f)))
    for f in tqdm(val, desc='Moving val files'):
      move(f, join(datapath, self.data_val, basename(f)))

    # Copy val data to self.data_test
    for f in tqdm(glob(join(datapath, self.data_val, '*')), desc='Copying test files'):
      copyfile(f, join(datapath, self.data_test, basename(f)))

    # Remove trainval
    rmtree(join(datapath, 'trainval'))

  def setup(self, stage: Optional[str] = None):
    # Assign train/val datasets for use in dataloaders
    if stage == 'fit' or stage is None:
      self.datamodule_train = Generator(
        self.datavariant,
        self.data_train,
        self.extension,
        normalize_head=self.normalize_head,
        norm_by_channel_head=False,
        normalize_tail=self.normalize_tail,
        norm_by_channel_tail=False,
        func_load_data=self.func_load_data,
        is_half=self.is_half,
        )
      self.datamodule_val   = Generator(
        self.datavariant,
        self.data_val,
        self.extension,
        normalize_head=self.normalize_head,
        norm_by_channel_head=False,
        normalize_tail=self.normalize_tail,
        norm_by_channel_tail=False,
        func_load_data=self.func_load_data,
        is_half=self.is_half,
        )

    # Assign test dataset for use in dataloader(s)
    if stage == 'test' or stage is None:
      self.datamodule_test = Generator(
        self.datavariant,
        self.data_test,
        self.extension,
        normalize_head=self.normalize_head,
        norm_by_channel_head=False,
        normalize_tail=self.normalize_tail,
        norm_by_channel_tail=False,
        func_load_data=self.func_load_data,
        is_half=self.is_half,
        return_name=True,
        )

    if stage == 'predict' or stage is None:
      self.datamodule_predict = Generator(
        self.datavariant,
        self.data_predict,
        self.extension,
        normalize_head=self.normalize_head,
        norm_by_channel_head=False,
        normalize_tail=self.normalize_tail,
        norm_by_channel_tail=False,
        func_load_data=self.func_load_data,
        is_half=self.is_half,
        is_train=False,
        )

  def train_dataloader(self):
    return DataLoader(self.datamodule_train,   batch_size=self.batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)

  def val_dataloader(self):
    return DataLoader(self.datamodule_val,     batch_size=self.batch_size, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)

  def test_dataloader(self):
    return DataLoader(self.datamodule_test,    batch_size=self.batch_size, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)

  def predict_dataloader(self):
    return DataLoader(self.datamodule_predict, batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=True)