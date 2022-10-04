import os
from shutil import rmtree
from scipy.io import savemat
from os.path import basename
from tqdm import tqdm
import numpy as np
from skimage import io
import time

import mlflow
from mlflow.models.signature import infer_signature

import torch
from pytorch_lightning import LightningModule
from pyramidal.pytorch.Generators.DataModule import DataModule
from pytorch_lightning.utilities.rank_zero import rank_zero_only

@rank_zero_only
def test(model: LightningModule, datamodule: DataModule, mlflow_obj: mlflow) -> None:
  r"""
  Get metrics from the model and log them to mlflow, and test the model on the test set.

  Args:
    model:      Model to test.
    datamodule: Data to test the model.
  """
  # TEST
  model.eval()
  model = model.cuda()
  # speed test
  with torch.no_grad():
    # Infer the model signature
    [x, y] = next(iter(datamodule.val_dataloader()))
    x = x.float().cuda()

    signature = infer_signature(x[0:1].detach().cpu().numpy(), model(x[0:1]).detach().cpu().numpy())
    mlflow_obj.pytorch.log_model(model, 'model', signature=signature)

    # test
    delta = 30 # [s]
    start = time.time()
    iters = 0
    while True:
      model(x[0:1])
      iters += 1
      if (time.time() - start) > delta:
        end = time.time()
        break

    # log to mlflow
    mlflow_obj.log_metric('speed s', (end - start) / iters)
    # mlflow.log_metric('speed ms', 1000 * (end - start) / iters)
    mlflow_obj.log_metric('speed Hz', iters / (end - start))
    mlflow_obj.log_metric('speed KHz', iters / (end - start) / 1000)

    # log artifact some inputs and outputs for visualization in the UI
    rmtree('/tmp/mlflow/save/details/', ignore_errors=True)
    os.makedirs('/tmp/mlflow/save/details/', exist_ok=True)

    datamodule.setup('test')

    indexs = np.random.choice(range(len(datamodule.datamodule_test)), 9, False)

    for index in indexs:
      x, y, path, denorm = datamodule.datamodule_test[index]
      x = denorm(x, 'head')
      x_img = (x - x.min()) / (x.max() - x.min())
      x_img_numpy = x_img.detach().cpu().numpy()
      io.imsave(f'/tmp/mlflow/save/details/{basename(path)[:-4]}.png', x_img_numpy)
    # add y to txt file for visualization in the UI
    with open('/tmp/mlflow/save/details/y.txt', 'w') as f:
      f.write(str(y.detach().cpu().numpy()))

    mlflow.log_artifact('/tmp/mlflow/save/details/', 'details')

    datamodule.setup('test')

    path_to_save = f'/tmp/optolab/camilo/results/'
    os.makedirs(path_to_save, exist_ok=True)

    for batch in tqdm(datamodule.datamodule_test, desc='Testing'):
      x, y, path, denorm_fn = batch
      x = x.float().cuda()
      y = y.float().cuda()

      y_hat = model(x.unsqueeze(0))

      y = denorm_fn(y, 'tail')
      y_hat = denorm_fn(y_hat, 'tail')

      y_hat = y_hat.cpu().detach().numpy()[0]
      y = y.cpu().detach().numpy()

      savemat(f'{path_to_save}/{basename(path)[:-4]}.mat', {'predict': y_hat, 'orig': y})


    print('Uploading results to mlflow...')
    mlflow_obj.log_artifact(path_to_save, 'results')