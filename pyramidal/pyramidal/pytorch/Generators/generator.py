from typing import Optional, List, Tuple
import os
import numpy as np
from tqdm import tqdm
from glob import glob
from os.path import join
from os.path import dirname

import torch
from torch.utils.data import Dataset

class Generator(Dataset):
  def __init__(
    self,
    datavariant:           Optional[str]      = '',
    dataset:               Optional[str]      = 'train',
    extension:             Optional[str]      = 'npy',
    normalize_head:        Optional[str]      = 'none',
    norm_by_channel_head:  Optional[bool]     = False,
    normalize_tail:        Optional[str]      = 'none',
    norm_by_channel_tail:  Optional[bool]     = False,
    func_load_data:        Optional[callable] = lambda x: np.load(x, allow_pickle=True),
    return_name:           Optional[bool]     = False,
    is_half:               Optional[bool]     = True,
    is_train:              Optional[bool]     = True,
    ) -> None:
    r"""
    Initialize the generator.

    Args:
      datavariant:           Path to the data.
      extension:             Extension of the files without the dot.
      normalize_head:        Normalization to use for the head (inputs).
      norm_by_channel_head:  Normalize by channel for the head (inputs).
      normalize_tail:        Normalization to use for the tail (outputs).
      norm_by_channel_tail:  Normalize by channel for the tail (outputs).
      return_name:           Whether to return the name of the file.
    """
    if is_train:
      from pyramidal.pytorch.Utils.datapath import datapath
      assert dataset in ['train', 'val', 'test']
      assert normalize_head in ['none', 'mean_std'], 'normalize_head must be one of: none, mean_std'
      assert normalize_tail in ['none', 'mean_std'], 'normalize_tail must be one of: none, mean_std'
      self.partition = np.array(sorted(glob(join(datapath, f'{dataset}/*.{extension}'))))
      assert len(self.partition) > 0, f'No data found in {dataset}/*.{extension}'

    self.is_train = 'save' if is_train else 'inference'

    self.extension      = extension
    self.func_load_data = func_load_data
    self.datavariant   = datavariant

    self.return_name    = return_name
    self.dataset        = dataset

    self.normalize_head = normalize_head
    self.normalize_tail = normalize_tail

    self.values_normalize_head = self._do_normalize(normalize_head, 'head', norm_by_channel_head)
    self.values_normalize_tail = self._do_normalize(normalize_tail, 'tail', norm_by_channel_tail)

    if self.normalize_head == 'mean_std':
      self.transform_head = lambda x: torch.tensor((x - self.values_normalize_head[0]) / self.values_normalize_head[1])
      self.detransform_head = lambda x: x * self.values_normalize_head[1] + self.values_normalize_head[0]
    else:
      self.transform_head = lambda x: torch.tensor(x)
      self.detransform_head = lambda x: x

    if self.normalize_tail == 'mean_std':
      self.transform_tail = lambda x: torch.tensor((x - self.values_normalize_tail[0]) / self.values_normalize_tail[1])
      self.detransform_tail = lambda x: x * self.values_normalize_tail[1] + self.values_normalize_tail[0]
    else:
      self.transform_tail = lambda x: torch.tensor(x)
      self.detransform_tail = lambda x: x

    self.is_half = is_half

  def denormalize(self, x: np.ndarray, section: str) -> np.ndarray:
    r"""
    Denormalize the data.

    Args:
      x: Data to denormalize.
      section: Section of the data to denormalize.

    Returns:
      The denormalized data.
    """
    if section == 'head':
      return self.detransform_head(x)
    elif section == 'tail':
      return self.detransform_tail(x)
    else:
      raise ValueError(f'Unknown section {section}')


  def _do_normalize(self, normalize:str, name:str, norm_by_channel:bool):
    r"""
    Do the normalization.

    Args:
      normalize:        Normalization to use.
      values_normalize: Values to use for the normalization.
      name:             Name of the normalization.

    Returns:
      The values to use for the normalization.

    Raises:
      ValueError: If the normalization is not implemented.
    """
    index = 0 if name == 'head' else 1
    name_by_channel = 'ch' if norm_by_channel else ''
    name_file = f'/tmp/mlflow/{self.is_train}/parameters/{self.datavariant}_{normalize}_{name}_{name_by_channel}.{self.extension}'
    os.makedirs(dirname(name_file), exist_ok=True)

    # check if self.values_normalize_{head,tail} is already computed
    if not isinstance(getattr(self, f'values_normalize_{name}', None), type(None)):
      return getattr(self, f'values_normalize_{name}')

    if normalize == 'none':
      values_normalize = np.array([])
    elif normalize == 'mean_std':
      if self.dataset == 'train':
        if os.path.exists(name_file):
          values_normalize = np.load(name_file)
        else:
          lenght = len(self.partition)
          path   = self.partition[0]
          if norm_by_channel:
            x                = self.process_path(path)[index]
            init             = np.zeros_like(np.mean(x, axis=tuple(range(x.ndim-1))), dtype=np.float32)
            values_normalize = np.array([init, init])
            for path in tqdm(self.partition, desc=f'{normalize}_{name}'):
              x = self.process_path(path)[index]
              if normalize == 'mean_std':
                values_normalize[0] += np.mean(x, axis=tuple(range(x.ndim-1)))
                values_normalize[1] += np.std(x, axis=tuple(range(x.ndim-1)))
          else:
            values_normalize = np.array([0, 0], dtype=np.float32)
            for path in tqdm(self.partition, desc=f'{normalize}_{name}'):
              x = self.process_path(path)[index]
              if normalize == 'mean_std':
                values_normalize[0] += np.mean(x)
                values_normalize[1] += np.std(x)

          values_normalize /= lenght
          np.save(name_file, values_normalize)
      else:
        try:
          values_normalize = np.load(name_file)
        except FileNotFoundError:
          raise ValueError(f'You need to compute the normalization for the {name} of the {self.datavariant} dataset'
                           f'Solution: set normalize_{name} to "mean_std" and dataset to "train"')
    else:
      raise ValueError(f'normalize_{name} must be one of: none, mean_std')

    return values_normalize

  def __len__(self):
    r"""
    Get the length of the dataset.

    Returns:
      The length of the dataset.
    """
    return len(self.partition)

  def process_path(self, path: str) -> List[np.ndarray]:
    r"""
    Process the path.

    Args:
      path: Path to the data.

    Returns:
      The processed data.

    Raises:
      ValueError: If the function to load the data is not implemented or if the function to load the data cannot load the data.
    """
    x, y = self.func_load_data(path)
    y = y[:int(self.datavariant)]
    return x, y

  def flow(self, x: np.ndarray, y: Optional[np.ndarray]=None) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    This function allow pass the data to the normalization function.

    Args:
      x: Inputs to the model.
      y: Label of the data.

    Returns:
      The data flowed to the model.
    """
    return self.__getitem__(0, x, y)

  def __getitem__(self, index, x=None, y=None):
    r"""
    Get the data.

    Args:
      index: Index of the data.
      x:     Inputs to the model.
      y:     Label of the data.

    Returns:
      The data.

    Raises:
      ValueError: If the function to load the data is not implemented or if the function to load the data cannot load the data.
    """
    if x is None:
      path = self.partition[index]
      x, y = self.process_path(path)

    x = self.transform_head(x)
    # cast to float or half
    if self.is_half:
      x = x.half()
    else:
      x = x.float()

    if y is not None:
      y = self.transform_tail(y)
      # cast to float or half
      if self.is_half:
        y = y.half()
      else:
        y = y.float()

      y = y * 700 / (2 * np.pi)

      if self.return_name:
        return x, y, path, self.denormalize
      return x, y
    return x