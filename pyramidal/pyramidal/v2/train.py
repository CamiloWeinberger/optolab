import argparse
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--datapath',       default='',            type=str,    help='Path to the data')
parser.add_argument('--model',          default='wfnet',       type=str,    help='Model to use')
parser.add_argument('--resume',         default=False, action='store_true', help='Resume training')
parser.add_argument('--batchsize',      default=2,             type=int,    help='Batch size')
parser.add_argument('--devices',        default="-1",          type=str,    help='GPU to use (default: -1 mean use all available GPU)')
parser.add_argument('--epochs',         default=60,            type=int,    help='Number of epochs')
parser.add_argument('--normalize_head', default='mean_std',    type=str,    help='normalize head: none, mean_std')
parser.add_argument('--normalize_tail', default='none',        type=str,    help='normalize tail: none, mean_std')
parser.add_argument('--strategy',       default='ddp',         type=str,    help='Strategy to use')
parser.add_argument('--use_swa',        default=False, action='store_true', help='Use Stochastic Weight Averaging')
args = parser.parse_args()

if args.devices == '-1':
  args.devices = int(args.devices)
else:
  args.devices = [int(x) for x in args.devices.split(',')]

import os
from os.path import join, dirname
from glob import glob

from pytorch_lightning import Trainer
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import StochasticWeightAveraging
from pytorch_lightning.loggers.mlflow import MLFlowLogger
import torchvision.transforms as T

from pyramidal.v2.Callbacks.MlFlowCheckpoint import MLFlowModelCheckpoint
from pyramidal.v2.Generators.DataModule import DataModule
from pyramidal.v2.Models import load_model

def main():
  tqdm_callback = TQDMProgressBar()

  database = '199'
  if args.datapath == '500modes':
    database = '500'

  mlflow = MLFlowLogger(experiment_name=f'pyramidal {database} {args.model}')

  checkpoint_callback = MLFlowModelCheckpoint(
    mlflow,
    dirpath='./saved_models/',
    filename=f'{args.model}-{{epoch}}-{{loss_val:.2f}}',
    mode='min',
    save_top_k=1,
    monitor='loss_val',
  )

  callbacks = [checkpoint_callback, tqdm_callback]
  if args.use_swa:
    swa = StochasticWeightAveraging(1e-4)
    callbacks.append(swa)

  trainer = Trainer(accelerator='gpu', devices=args.devices,
                    max_epochs=args.epochs,
                    num_sanity_val_steps=2,
                    callbacks=callbacks,
                    logger=mlflow,
                    strategy = DDPStrategy(find_unused_parameters=False) if args.strategy == 'ddp' else args.strategy,
                    precision=16,
                    amp_backend="native",
                    )

  model_to_use = load_model(args.model)

  if args.resume:
    all_models_path = glob(f'{join(dirname(dirname(dirname(__file__))), "saved_models", args.model)}*')
    def sorted_path_by_val_loss(path):
      return float(path.split('=')[-1][:-5])
    all_models_path = sorted(all_models_path, key=sorted_path_by_val_loss)
    resume_from_checkpoint = all_models_path[0]
    if os.path.isdir(resume_from_checkpoint):
      resume_from_checkpoint = join(resume_from_checkpoint, 'checkpoint', 'mp_rank_00_model_states.pt')
      model_to_use.load_model_is_deepspeed = True
    model_to_use.use_class_args = True
    model_to_use.datapath_class = args.datapath
    model = model_to_use.load_from_checkpoint(resume_from_checkpoint)
    if model_to_use.load_model_is_deepspeed:
      resume_from_checkpoint = dirname(dirname(resume_from_checkpoint))
    print(f'Loaded model from {os.path.basename(resume_from_checkpoint)}')
  else:
    model = model_to_use(datapath=args.datapath)

  transform = lambda x: T.Resize((224, 224))(x)
  dm = DataModule(
    args.datapath,
    os.path.basename(args.datapath),
    batch_size=args.batchsize,
    type_input='image',
    type_output='regression',
    normalize_head=args.normalize_head,
    normalize_tail=args.normalize_tail,
    custom_transform_head=transform,
    # custom_transform_tail=transform
    )

  if False:
    x = next(iter(dm.train_dataloader()))
    # create a random batch of input data
    # x = torch.rand(1, 1, 224, 224)
    model.train()
    model.forward(x)

  trainer.fit(model, dm)

if __name__ == '__main__':
  main()