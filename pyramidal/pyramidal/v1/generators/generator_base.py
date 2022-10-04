import os
import numpy as np
from tqdm import tqdm
from glob import glob
from numpy import newaxis
from pyramidal.v1.datapath import datapath, data_split

class GeneratorBase:
  def __init__(self, dataset:str='train', batchsize=1, normalize_head='none', values_normalize_head=[], normalize_tail='none', values_normalize_tail=[], is_gcvit=False, return_name=False):
    assert dataset in ['train', 'val', 'test']
    self.is_gcvit    = is_gcvit
    self.batchsize   = batchsize
    self.return_name = return_name
    self.dataset     = dataset
    if dataset in ['train', 'val']:
      train_index, val_index = data_split
      if dataset == 'train':
        index = train_index
      else:
        index = val_index
      self.partition = np.array(sorted(glob(f'{datapath}/trainval/*.npy')))[index]
    else:
      self.partition = np.array(sorted(glob(f'{datapath}/test/*.npy')))

    self.normalize_head = normalize_head
    self.normalize_tail = normalize_tail

    self.values_normalize_head = self.do_normalize(normalize_head, values_normalize_head, 'head')
    self.values_normalize_tail = self.do_normalize(normalize_tail, values_normalize_tail, 'tail')

    self.on_epoch_end()

  def do_normalize(self, normalize, values_normalize, name):
    normalize = normalize
    index = 0 if name == 'head' else 1
    NoneType = type(None)

    if normalize == 'none':
      values_normalize = []
    elif normalize in ['mean_std', 'mean_std_min_max', 'min_max']:
      values_normalize = values_normalize
      if self.dataset == 'train':
        if os.path.exists(f'{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/parameters/{normalize}_{name}.npy'):
          values_normalize = np.load(f'{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/parameters/{normalize}_{name}.npy')
        else:
          lenght = len(self.partition)
          values_normalize = [0, 0, 0, 0]
          for path in tqdm(self.partition, desc=f'{normalize}_{name}'):
            x = self._process_path(path)[index]
            if isinstance(x, NoneType):
              continue
            if normalize == 'mean_std':
              values_normalize[0] += np.mean(x)
              values_normalize[1] += np.std(x)
            elif normalize == 'mean_std_min_max':
              values_normalize[0] += np.mean(x)
              values_normalize[1] += np.std(x)
              values_normalize[2] += np.min(x)
              values_normalize[3] += np.max(x)
            elif normalize == 'min_max':
              values_normalize[0] += np.min(x)
              values_normalize[1] += np.max(x)
          values_normalize = [x / lenght for x in values_normalize]
          np.save(f'{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/parameters/{normalize}_{name}.npy', values_normalize)
    else:
      raise ValueError(f'normalize_{name} must be one of: none, mean_std, min_max')

    return values_normalize

  def _process_path(self, pyramidal_path):
    X, Y = np.load(pyramidal_path, allow_pickle=True)
    return X[..., newaxis], Y

  def on_epoch_end(self):
    range_data = np.arange(len(self.partition))
    if self.dataset == 'train':
      np.random.shuffle(range_data)
    out_list = []
    l = []
    for lis in range_data:
      if len(l) == self.batchsize:
        out_list.append(l)
        l = []
      l.append(lis)
    if len(l) > 0:
      out_list.append(l)

    self.indexed_data = np.array(out_list)

  def __len__(self):
    return len(self.indexed_data)

  def process_path(self, path):
    while True:
      input_model, label = self._process_path(path)
      if input_model is None:
        if os.path.exists('error.log'):
          with open('error.log', 'r') as f:
            lines = f.readlines()
          if path not in lines:
            with open('error.log', 'a') as f:
              f.write(path + '\n')
        else:
          with open('error.log', 'w') as f:
            f.write(path + '\n')
        path = np.random.choice(self.partition)
      else:
        break

    return input_model, label