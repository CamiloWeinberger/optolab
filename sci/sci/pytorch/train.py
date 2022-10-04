import argparse
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--datapath',  default='/Storage1/Matias/Datasetx30', type=str,    help='Model to use')
parser.add_argument('--model',     default='RegUNetTv2tiny', type=str,    help='Model to use')
parser.add_argument('--resume',    default=False, action='store_true',    help='Resume training')
parser.add_argument('--batchsize', default=2,     type=int,               help='Batch size')
parser.add_argument('--devices',   default="-1",  type=str,               help='GPU to use (default: -1 mean use all available GPU)')
parser.add_argument('--epochs',    default=60,    type=int,               help='Number of epochs')
parser.add_argument('--strategy',  default='ddp', type=str,               help='Strategy to use')
parser.add_argument('--use_swa',   default=False, action='store_true',    help='Strategy to use')
parser.add_argument('--is_float',  default=False, action='store_true',    help='Use 32 precision')
args = parser.parse_args()

if args.devices == '-1':
  args.devices = int(args.devices)
else:
  args.devices = [int(x) for x in args.devices.split(',')]


import os
from os.path import join, basename, dirname
from glob import glob

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.strategies.deepspeed import DeepSpeedStrategy
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import StochasticWeightAveraging

from sci.pytorch.Generators.DataModule import DataModule
from sci.pytorch.Models import load_model

def main():
  if args.model == 'sci3d':
    args.batchsize = 1

  version_num = 0
  if not os.path.exists('lightning_logs'):
    os.mkdir('lightning_logs')
  if len(glob('lightning_logs/version_*')) > 0:
    def key(value):
      return int(value.split('_')[-1])
    version_num = int(sorted(glob('lightning_logs/version_*'), key=key)[-1].split('_')[-1]) + 1

  checkpoint_callback = ModelCheckpoint(dirpath='./saved_models/',
                                        filename=f'{args.model}-v={version_num:02d}-{{epoch}}-{{loss_val:.2f}}',
                                        mode='min',
                                        save_top_k=1,
                                        monitor='loss_val',
                                        )
  tqdm_callback = TQDMProgressBar()

  callbacks = [checkpoint_callback, tqdm_callback]
  if args.use_swa:
    swa = StochasticWeightAveraging(1e-4)
    callbacks.append(swa)

  trainer = Trainer(accelerator='gpu', devices=args.devices,
                    max_epochs=args.epochs,
                    num_sanity_val_steps=2,
                    callbacks=callbacks,
                    strategy = DDPStrategy(find_unused_parameters=False) if args.strategy == 'ddp' else args.strategy,
                    precision=16,
                    amp_backend="native",
                    )

  model_to_use = load_model(args.model)
  model_to_use.is_half = not args.is_float

  if args.resume:
    all_models_path = glob(f'{join(dirname(dirname(dirname(__file__))), "saved_models", args.model)}*')
    def sorted_path_by_val_loss(path):
      return float(path.split('=')[-1][:-5])
    all_models_path = sorted(all_models_path, key=sorted_path_by_val_loss)
    resume_from_checkpoint = all_models_path[0]
    if os.path.isdir(resume_from_checkpoint):
      resume_from_checkpoint = join(resume_from_checkpoint, 'checkpoint', 'mp_rank_00_model_states.pt')
      model_to_use.load_model_is_deepspeed = True
    model = model_to_use.load_from_checkpoint(resume_from_checkpoint)
    if model_to_use.load_model_is_deepspeed:
      resume_from_checkpoint = dirname(dirname(resume_from_checkpoint))
    print(f'Loaded model from {os.path.basename(resume_from_checkpoint)}')
  else:
    model = model_to_use()

  transform = lambda x: x.permute(3, 2, 0, 1)
  dm = DataModule(
    args.datapath,
    os.path.basename(args.datapath),
    batch_size=args.batchsize,
    type_input='3d',
    type_output='3d',
    normalize_head='none',
    normalize_tail='none',
    custom_transform_head=transform,
    custom_transform_tail=transform
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