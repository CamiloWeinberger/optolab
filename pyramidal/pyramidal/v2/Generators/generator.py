from typing import Optional, List, Tuple

import torch
from torchvision import transforms
from torch.utils.data import Dataset

import os
import numpy as np
from tqdm import tqdm
from glob import glob

class Generator(Dataset):
  _values_normalize_head = None
  _values_normalize_tail = None
  def __init__(
    self,
    datapath:              Optional[str]              = '',
    dataset:               Optional[str]              = 'train',
    extension:             Optional[str]              = 'npy',
    type_input:            Optional[str]              = 'image',
    type_output:           Optional[str]              = 'regression',
    normalize_head:        Optional[str]              = 'none',
    norm_by_channel_head:  Optional[bool]             = False,
    normalize_tail:        Optional[str]              = 'none',
    norm_by_channel_tail:  Optional[bool]             = False,
    custom_transform_head: Optional[callable]         = None,
    custom_transform_tail: Optional[callable]         = None,
    func_load_data:        Optional[callable]         = lambda x: np.load(x, allow_pickle=True),
    name_dataset:          Optional[str]              = 'dataset',
    return_name:           Optional[bool]             = False
    ) -> None:
    r"""
    Initialize the generator.

    Args:
      datapath:              Path to the data.
      dataset:               Dataset to use.
      extension:             Extension of the files without the dot.
      type_input:            Type of input.
      type_output:           Type of output.
      normalize_head:        Normalization to use for the head (inputs).
      norm_by_channel_head:  Normalize by channel for the head (inputs).
      normalize_tail:        Normalization to use for the tail (outputs).
      norm_by_channel_tail:  Normalize by channel for the tail (outputs).
      custom_transform_head: Custom transform to use.
      custom_transform_tail: Custom transform to use.
      name_dataset:          Name of the dataset.
      return_name:           Whether to return the name of the file.
    """

    assert dataset in ['train', 'val', 'test']
    assert type_input  in ['image', 'regression', 'classification', '3d'], 'type_input must be one of: image, regression, classification, 3d'
    assert type_output in ['image', 'regression', 'classification', '3d'], 'type_output must be one of: image, regression, classification, 3d'
    assert normalize_head in ['none', 'mean_std'], 'normalize_head must be one of: none, mean_std'
    assert normalize_tail in ['none', 'mean_std'], 'normalize_tail must be one of: none, mean_std'
    self.partition = np.array(sorted(glob(f'{datapath}/{dataset}/*.{extension}')))
    assert len(self.partition) > 0, f'No data found in {datapath}/{dataset}/*.{extension}'

    self.extension      = extension
    self.func_load_data = func_load_data
    self.name_dataset   = name_dataset

    self.return_name    = return_name
    self.dataset        = dataset

    self.normalize_head = normalize_head
    self.normalize_tail = normalize_tail

    self.values_normalize_head = self._do_normalize(normalize_head, 'head', norm_by_channel_head)
    self.values_normalize_tail = self._do_normalize(normalize_tail, 'tail', norm_by_channel_tail)

    self.type_input = type_input
    self.type_output = type_output

    if type_input == 'image':
      if self.normalize_head == 'mean_std':
        self.transform_head = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize(mean=self.values_normalize_head[0],
                              std=self.values_normalize_head[1]),
          ])
      else :
        self.transform_head = transforms.Compose([
          transforms.ToTensor(),
          ])
    elif type_input in ['regression', 'classification', '3d']:
      if self.normalize_head == 'mean_std':
        self.transform_head = lambda x: torch.tensor((x - self.values_normalize_head[0]) / self.values_normalize_head[1])
      else:
        self.transform_head = lambda x: torch.tensor(x)
    else:
      raise NotImplementedError(f'{type_input} is not implemented')

    if type_output == 'image':
      if self.normalize_tail == 'mean_std':
        self.transform_tail = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize(mean=self.values_normalize_tail[0],
                              std=self.values_normalize_tail[1]),
          ])
      else :
        self.transform_tail = transforms.Compose([
          transforms.ToTensor(),
          ])
    elif type_output in ['regression', 'classification', '3d']:
      if self.normalize_tail == 'mean_std':
        self.transform_tail = lambda x: torch.tensor((x - self.values_normalize_tail[0]) / self.values_normalize_tail[1])
      else:
        self.transform_tail = lambda x: torch.tensor(x)
    else:
      raise NotImplementedError(f'{type_output} is not implemented')

    self.custom_transform_head = custom_transform_head
    self.custom_transform_tail = custom_transform_tail


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
    name_file = f'{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/parameters/{self.name_dataset}_{normalize}_{name}_{name_by_channel}.{self.extension}'
    if not os.path.exists(os.path.dirname(name_file)):
      os.mkdir(os.path.dirname(name_file))

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
            init             = np.zeros_like(np.mean(x, axis=tuple(range(x.ndim-1))))
            values_normalize = np.array([init, init])
            for path in tqdm(self.partition, desc=f'{normalize}_{name}'):
              x = self.process_path(path)[index]
              if normalize == 'mean_std':
                values_normalize[0] += np.mean(x, axis=tuple(range(x.ndim-1)))
                values_normalize[1] += np.std(x, axis=tuple(range(x.ndim-1)))
          else:
            values_normalize = np.array([0, 0])
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
          raise ValueError(f'You need to compute the normalization for the {name} of the {self.name_dataset} dataset'
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
    if self.custom_transform_head is not None:
      x = self.custom_transform_head(x)
    # cast to float32 in pytorch
    x = x.half()

    if y is not None:
      y = self.transform_tail(y)
      if self.custom_transform_tail is not None:
        y = self.custom_transform_tail(y)
      y = y.half()

      if self.return_name:
        return x, y, path
      return x, y
    return x