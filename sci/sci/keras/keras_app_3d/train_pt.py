import os
import argparse
from glob import glob
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('--model', default='lightweight', help='Model to use')
ap.add_argument('--resume', default=False, action='store_true', help='Resume training')
ap.add_argument('--batchsize', default=1, type=int, help='Batch size')
ap.add_argument('--use_gpus', default='all', type=str, help='all or by id: 0,1,2 or 0 or 1 or 0,1,2,3,4,5,6,7')
ap.add_argument('--epochs', default=100, type=int, help='Number of epochs')
ap.add_argument('--normalize_head', default='none', type=str, help='normalize head: none, mean_std, mean_std_min_max or min_max')
ap.add_argument('--normalize_tail', default='none', type=str, help='normalize tail: none, mean_std, mean_std_min_max or min_max')
args = vars(ap.parse_args())

if args['use_gpus'] != 'all':
  if args['use_gpus'][-1] == ',':
    args['use_gpus'] = args['use_gpus'][:-1]
  if args['use_gpus'][0] == ',':
    args['use_gpus'] = args['use_gpus'][1:]
  os.environ['CUDA_VISIBLE_DEVICES'] = args['use_gpus']
else:
  import torch
  args['use_gpus'] = ','.join(str(v) for v in range(torch.cuda.device_count()))
  os.environ['CUDA_VISIBLE_DEVICES'] = args['use_gpus']

from pytorch_lightning import Trainer
from sci.generators.DataModule import SciDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar

def main():
  num_gpus = len(args['use_gpus'].split(','))
  version_num = 0
  if len(glob('lightning_logs/version_*')) > 0:
    version_num = int(sorted(glob('lightning_logs/version_*'))[-1].split('_')[-1]) + 1

  if args['model'] == 'WFNet':
    model_to_use = WFNet
    is_gcvit = False
    name_model = 'WFNet'
  elif args['model'] == 'GC_VIT' != -1:
    model_to_use = GC_VIT_Lightning
    is_gcvit = True
    name_model = 'GC_VIT'
  elif args['model'] == 'GC_VIT_3d':
    model_to_use = GC_VIT_3D_Lightning
    is_gcvit = True
    name_model = 'GC_VIT_3d'


  checkpoint_callback = ModelCheckpoint(dirpath='./saved_models/',
                                        filename=f'{name_model}-v={version_num:02d}-{{epoch}}-{{val_loss:.2f}}',
                                        mode='min',
                                        save_top_k=1,
                                        monitor='val_loss',
                                        )
  tqdm_callback = TQDMProgressBar()

  trainer = Trainer(accelerator='gpu', devices=num_gpus,
                    max_epochs=args['epochs'],
                    num_sanity_val_steps=2,
                    callbacks=[checkpoint_callback, tqdm_callback],
                    amp_backend='native',
                    profiler='simple',
                    )

  if args['resume']:
    all_models_path = glob(f'{os.path.join(os.path.dirname(__file__), "saved_models", name_model)}*')
    def sorted_path_by_val_loss(path):
      return float(path.split('=')[-1][:-5])
    all_models_path = sorted(all_models_path, key=sorted_path_by_val_loss)
    resume_from_checkpoint = all_models_path[0]
    print(f'Loading model from {resume_from_checkpoint}')
    model = model_to_use.load_from_checkpoint(resume_from_checkpoint)
  else:
    model = model_to_use()

  trainer.fit(model, SciDataModule(batch_size=args['batchsize'],))

if __name__ == '__main__':
  main()