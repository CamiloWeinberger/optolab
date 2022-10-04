import os
from os.path import join, dirname

# set logger critical
import logging
logging.getLogger().setLevel(logging.CRITICAL)
# ignore warnings
import warnings
warnings.filterwarnings("ignore")

import mlflow

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import MLFlowLogger

from pyramidal.pytorch.Generators.DataModule import DataModule
from pyramidal.pytorch.Models import load_model
from pyramidal.pytorch.Utils.mlflow_utils import get_tracking_uri
from pyramidal.pytorch.test import test
from pyramidal.pytorch.Utils.download_artifacts_mlflow import compile_download_artifacts_mlflow

def main(strategy_var=None, resume=None):
  import argparse
  parser = argparse.ArgumentParser(description='PyTorch Training for pyramidal')
  app = parser.add_argument_group('Global')
  app.add_argument('--datavariant',    default='54',          type=str,    help='Variant of the data to use')
  app.add_argument('--model',          default='wfnet',       type=str,    help='Model to use')
  app.add_argument('--resume',         default=False, action='store_true', help='Resume training')
  app.add_argument('--batchsize',      default=64,            type=int,    help='Batch size')
  app.add_argument('--use_swa',        default=False, action='store_true', help='Use SWA')
  app.add_argument('--is_float',       default=False, action='store_true', help='Use 32 precision')
  app.add_argument('--normalize_head', default='mean_std',    type=str,    help='normalize head: none, mean_std')
  app.add_argument('--normalize_tail', default='none',        type=str,    help='normalize tail: none, mean_std')

  parser = Trainer.add_argparse_args(parser)
  # set default values
  strategy_var = strategy_var if strategy_var is not None else 'ddp'
  strategy_var = DDPStrategy(find_unused_parameters=False) if strategy_var == 'ddp' else strategy_var
  parser.set_defaults(
    accelerator= 'gpu',
    devices    = -1,
    strategy   = strategy_var,
    num_workers= os.cpu_count() // 2, # more will increase your CPU memory consumption
    max_epochs = 1,
    precision  = 16, #why: https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html
  )

  temp_args, _ = parser.parse_known_args()
  model_to_use = load_model(temp_args.model)
  parser = model_to_use.add_model_specific_args(parser)

  args = parser.parse_args()

  args.resume = resume if resume != None else args.resume

  if args.resume:
    from pyramidal.pytorch.inference import load_model_from_mlflow
    model, _ = load_model_from_mlflow(project='pyramidal', datavariant=args.datavariant, modelName=args.model, what_choice='best')
  else:
    model = model_to_use(**vars(args))

  model.prepare_weights()

  checkpoint_callback = ModelCheckpoint(
    mode='min',
    save_top_k=1,
    monitor='loss_val',
  )
  tqdm_callback = TQDMProgressBar(refresh_rate=5)

  dm = DataModule(
    args.datavariant,
    batch_size=args.batchsize,
    normalize_head=args.normalize_head,
    normalize_tail=args.normalize_tail,
    is_half=not args.is_float,
  )
  dm.prepare_data()
  dm.setup('fit')

  # set tracking uri
  import time
  mlflow_tracking_uri = get_tracking_uri()
  mlflow.set_tracking_uri(mlflow_tracking_uri)
  # set experiment name
  mlflow.set_experiment(f'pyramidal {args.datavariant}')
  run_id = None
  save_run_id = True
  if os.path.exists('/tmp/mlflow/current_id.txt'):
    with open('/tmp/mlflow/current_id.txt', 'r') as f:
      line = f.readlines()
      if len(line) > 0:
        line = line[0].strip()
    if len(line) > 0:
      start, run_id_tmp = line.split(';;')
      if (time.time() - float(start)) < 60:
        save_run_id = False
        run_id = run_id_tmp

  with mlflow.start_run(run_id=run_id) as run:
    mlflow_uri = mlflow.get_tracking_uri()
    exp_id = run.info.experiment_id
    exp_name = mlflow.get_experiment(exp_id).name

    mlf_logger = MLFlowLogger(experiment_name=exp_name, tracking_uri=mlflow_uri)
    mlf_logger._run_id = run.info.run_id

    if save_run_id:
      with open('/tmp/mlflow/current_id.txt', 'w') as f:
        f.write(f'{time.time()};;{run.info.run_id}')

    locals()[args.model] = model_to_use

    trainer = Trainer.from_argparse_args(args,
      logger=mlf_logger,
      callbacks=[tqdm_callback, checkpoint_callback],
    )

    mlflow.pytorch.autolog()

    # add tag modelName
    mlflow.set_tag('modelName', args.model)

    # log code for load model
    mlflow.log_artifact(join(dirname(__file__), 'Models', args.model+'.py'), 'parameters')
    mlflow.log_artifact(join(dirname(__file__), 'Models', '_base.py'), 'parameters')
    mlflow.log_artifact(join(dirname(__file__), 'Models', '_losses.py'), 'parameters')
    mlflow.log_artifact(join(dirname(__file__), 'Utils', '_parser_utils.py'), 'parameters')
    mlflow.log_artifact(join(dirname(__file__), 'Generators', 'generator.py'), 'parameters')

    # log artifact params
    mlflow.log_artifact('/tmp/mlflow/save/parameters/', 'parameters')

    # log code for download artifacts
    compile_download_artifacts_mlflow(mlflow_tracking_uri, 'pyramidal', run.info.run_id, args.datavariant, mlflow)

    # Train the model ⚡
    trainer.fit(model, dm)


    # load the best model
    model = model_to_use.wrap_load_from_checkpoint(checkpoint_callback.best_model_path)

    # to gpu
    test(model, dm, mlflow)



if __name__ == '__main__':
  main()