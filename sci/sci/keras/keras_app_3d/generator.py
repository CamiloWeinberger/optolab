#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 14:34:03 2022

@author: mvaldipucv
"""

import numpy as np
from glob import glob
import scipy.io as scio
from numpy import newaxis
from tensorflow import keras
import tensorflow as tf

class Generator(keras.utils.Sequence):
  def __init__(self, dataset='train', datapath='Datasetx30', batchsize=1, is_1d=False, return_name=False, is_custom_processing=False, need_mask=False):
    assert dataset in ['train', 'val', 'test_v1', 'test_v2']
    self.batchsize   = batchsize
    self.dataset     = dataset
    self.is_1d       = is_1d
    self.return_name = return_name
    self.is_custom_processing = is_custom_processing
    self.need_mask = need_mask

    self.partition = []

    for cubes_path in glob(f'/Storage1/Matias/{datapath}/{dataset}/*'):
      self.partition.append(cubes_path)

    self.partition = np.array(self.partition)

    self.on_epoch_end()

  def process_path(self, cubes):
    mat  = scio.loadmat(cubes)
    if 'mask' in mat:
      # meas = mat['meas']
      mask = mat['mask']
      orig = mat['orig']

      # mask_s        = np.sum(mask, axis=-1)
      # index         = np.where(mask_s == 0)
      # mask_s[index] = 1
      # mask_s        = mask_s.astype(np.float32)

      # meas_re = meas / mask_s
      # meas_re = meas_re[..., None]

      input_model = orig * mask


      # if self.is_custom_processing:
      #   input_model = mask * meas_re + (mask*-1 + 1) * np.ones_like(meas_re) * -1 * 255
      # else:
      #   input_model = mask * meas_re

      input_model = input_model[newaxis, ..., newaxis]
      label       = orig       [newaxis, ..., newaxis]

      if self.need_mask:
        mask = mask[newaxis, ..., newaxis]
        return [input_model, mask], label

      return input_model, label
    else:
      print(cubes)
      return None, None

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


  def __getitem__(self, index):
    indexs = self.indexed_data[index]
    paths  = self.partition[indexs]
    inputs_model = []
    masks_inputs = []
    labels       = []
    for path in paths:
      if self.need_mask:
        [input_model, mask], label = self.process_path(path)
      else:
        input_model, label = self.process_path(path)
      if input_model is None:
        continue
      inputs_model.append(input_model)
      labels.append(label)
      if self.need_mask:
        masks_inputs.append(mask)
    inputs_model = np.float32(np.concatenate(inputs_model, axis=0))
    labels       = np.float32(np.concatenate(labels, axis=0))

    # inputs_model = inputs_model / 255
    # labels       = labels / 255

    if self.need_mask:
      masks_inputs = np.float32(np.concatenate(masks_inputs, axis=0))
      if self.return_name:
        return [inputs_model, masks_inputs], labels, paths
      return [inputs_model, masks_inputs], labels

    if self.return_name:
      return inputs_model, labels, paths
    return inputs_model, labels

  def __call__(self):
    for [s1, s2], l in self:
      yield {"inputs_img": s1, "inputs_mask": s2}, l

def DatasetFromSequenceClass(sequenceClass, stepsPerEpoch, nEpochs, batchSize, dims=[256,256,16,1], data_type=tf.float32, label_type=tf.float32):
    # eager execution wrapper
    def DatasetFromSequenceClassEagerContext(func):
        def DatasetFromSequenceClassEagerContextWrapper(batchIndexTensor):
            # Use a tf.py_function to prevent auto-graph from compiling the method
            tensors = tf.py_function(
                func,
                inp=[batchIndexTensor],
                Tout=[data_type, label_type]
            )

            # set the shape of the tensors - assuming channels last
            tensors[0].set_shape([batchSize] + dims)   # [samples, height, width, nChannels]
            tensors[1].set_shape([batchSize] + dims) # [samples, height, width, nClasses for one hot]
            return tensors
        return DatasetFromSequenceClassEagerContextWrapper

    # TF dataset wrapper that indexes our sequence class
    @DatasetFromSequenceClassEagerContext
    def LoadBatchFromSequenceClass(batchIndexTensor):
        # get our index as numpy value - we can use .numpy() because we have wrapped our function
        batchIndex = batchIndexTensor.numpy()

        # zero-based index for what batch of data to load; i.e. goes to 0 at stepsPerEpoch and starts cound over
        zeroBatch = batchIndex % stepsPerEpoch

        # load data
        data, labels = sequenceClass[zeroBatch]

        # convert to tensors and return
        return tf.convert_to_tensor(data), tf.convert_to_tensor(labels)

    # create our data set for how many total steps of training we have
    dataset = tf.data.Dataset.range(stepsPerEpoch*nEpochs)

    # return dataset using map to load our batches of data, use TF to specify number of parallel calls
    return dataset.map(LoadBatchFromSequenceClass, num_parallel_calls=tf.data.experimental.AUTOTUNE)




if __name__ == '__main__':
  from tqdm import tqdm
  index = 0

  data_train = Generator(datapath='DatasetsBase25', batchsize=1, is_custom_processing=False)
  data_val   = Generator('val', datapath='DatasetsBase25', batchsize=1, is_custom_processing=False)
  data_test  = Generator('test_v2', datapath='DatasetsBase25', batchsize=1, is_custom_processing=False)

  self = data_train

  # load our data as tensorflow datasets
  # training = DatasetFromSequenceClass(data_train, len(data_train), 40, 8, dims=[256,256,1,16])
