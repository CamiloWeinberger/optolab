import os
import sys
import re
from glob import glob
from shutil import rmtree
from os.path import basename, exists
import numpy as np

import mlflow
import mlflow.pytorch
from pyramidal.pytorch.Utils.mlflow_utils import get_tracking_uri


def load_model_from_mlflow(*, project='pyramidal', datavariant='baseline', modelName=None, what_choice='best'):
  """
  Load a model from mlflow
  This function is used to load a model from mlflow

  Steps:
  1. Choose the model to load (best or last)
  2. Select the run_id chosen
  3. Download the code of the model from mlflow
  4. Load the model from the code and set it to local variable
  5. Load the model with the weights from mlflow
  """
  assert what_choice in ['best', 'last']
  # set tracking uri
  mlflow_tracking_uri = get_tracking_uri()
  mlflow.set_tracking_uri(mlflow_tracking_uri)
  # for this example we use random run id from mlflow
  # get random run id
  name = f'{project} {datavariant}'
  exp = mlflow.get_experiment_by_name(f'{project} {datavariant}')
  # get all metrics for all runs
  order_by = 'metrics.loss_val ASC' if what_choice == 'best' else 'metrics.end_time DESC'
  # search by model name and order by loss
  filter_string = ''
  if modelName != None:
    filter_string = f'tags.modelName = "{modelName}"'
  metrics = mlflow.search_runs(
    experiment_ids=exp.experiment_id,
    filter_string=filter_string,
    order_by=[order_by],
    max_results=1,
  )
  run_id = metrics.iloc[0].run_id

  # download artifact
  artifact_uri = f'runs:/{run_id}/parameters'
  if exists(f'/tmp/mlflow/inference/{name}/parameters'):
    rmtree(f'/tmp/mlflow/inference/{name}/parameters')
  mlflow.artifacts.download_artifacts(artifact_uri, dst_path=f'/tmp/mlflow/inference/{name}/')
  paths_model = sorted(glob(f'/tmp/mlflow/inference/{name}/parameters/*.py'))
  # get modelName
  modelName = [basename(path).split('.')[0] for path in paths_model if not basename(path).startswith('_') and basename(path).find('generator') == -1][0]

  # import model code from artifact
  sys.path.append(f'/tmp/mlflow/inference/{name}/parameters')
  for path in paths_model:
    with open(path) as f:
      lines = f.readlines()
      lines_to_write = []
      for line in lines:
        # remove pyramidal.*.Models. from all imports
        if line.find(f'{project}.') != -1 and line.find('.Models.') != -1:
          # remove with re pattern pyramidal.*.Models.
          line = re.sub(f'{project}.*.Models.', '', line)
        # remove pyramidal.*.Utils. from all imports
        elif line.find(f'{project}.') != -1 and line.find('.Utils.') != -1:
          # remove with re pattern pyramidal.*.Utils.
          line = re.sub(f'{project}.*.Utils.', '', line)
        lines_to_write.append(line)
    with open(path, 'w') as f:
      f.writelines(lines_to_write)
  # add to variable the model class
  locals()[modelName] = getattr(__import__(modelName), modelName)

  # Add mnist_model variable to local variables
  model = mlflow.pytorch.load_model(f'runs:/{run_id}/model')
  # configure data flow for inference
  values_normalize_head = None
  values_normalize_tail = None
  if metrics.iloc[0]['params.normalize_head'] != 'none':
    file_npy = glob(f'/tmp/mlflow/inference/{name}/parameters/parameters/*head*.npy')[0]
    values_normalize_head = np.load(file_npy)
  if metrics.iloc[0]['params.normalize_tail'] != 'none':
    file_npy = glob(f'/tmp/mlflow/inference/{name}/parameters/parameters/*tail*.npy')[0]
    values_normalize_tail = np.load(file_npy)
  from generator import Generator
  Generator.values_normalize_head = values_normalize_head
  Generator.values_normalize_tail = values_normalize_tail
  gen = Generator(
    normalize_head=metrics.iloc[0]['params.normalize_head'],
    normalize_tail=metrics.iloc[0]['params.normalize_tail'],
    is_train=False
  )
  return model, gen.flow



if __name__ == '__main__':
  import os
  os.chdir('testTemplate')
  load_model_from_mlflow(what_choice='last')