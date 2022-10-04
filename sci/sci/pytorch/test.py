import argparse
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--datapath',       default='',            type=str,    help='Model to use')
parser.add_argument('--model',          default='RegUNetT',      type=str,    help='Model to use')
parser.add_argument('--batchsize',      default=16,            type=int,    help='Batch size')
parser.add_argument('--devices',        default=-1,            type=int,    help='GPU to use (default: -1 mean use all available GPU)')
parser.add_argument('--another_test',   default=True,  action='store_false', help='Do another test what you writed')
parser.add_argument('--save_test',      default=True,  action='store_false', help='Do another test what you writed')
args = parser.parse_args()

import os
from os.path import dirname, join, basename
from glob import glob
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.strategies.ddp import DDPStrategy

from sci.pytorch.Generators.DataModule import DataModule
from sci.pytorch.Models import load_model

def main():
  user = os.environ.get('USER')
  if user.find('mvaldi') != -1:
    datapath = f'/media/mvaldi-pucv-low/data/alejandro/Datasetx30'
  elif user == 'optolab':
    datapath = f'/Storage1/Matias/alejandro/Datasetx30'
  elif user.find('bizon') != -1:
    datapath = f'/home/bizon/Documents/matias/alejandro/Datasetx30'
  else:
    raise Exception('Unknown user')

  args.datapath = datapath

  if args.devices != -2:
    trainer = Trainer(accelerator='gpu', devices=args.devices,
                      num_sanity_val_steps=2,
                      profiler='simple',
                      strategy = DDPStrategy(find_unused_parameters=False)
                      )

  model_to_load = load_model(args.model)

  all_models_path = glob(f'{join(dirname(dirname(dirname(__file__))), "saved_models", args.model)}*')
  def sorted_path_by_val_loss(path):
    return float(path.split('=')[-1][:-5])
  all_models_path = sorted(all_models_path, key=sorted_path_by_val_loss)
  resume_from_checkpoint = all_models_path[0]
  if os.path.isdir(resume_from_checkpoint):
    resume_from_checkpoint = join(resume_from_checkpoint, 'checkpoint', 'mp_rank_00_model_states.pt')
    model_to_load.load_model_is_deepspeed = True
  model = model_to_load.load_from_checkpoint(resume_from_checkpoint)
  print(f'Loaded model from {resume_from_checkpoint}')
  model.is_half = False

  if not args.another_test:

    transform = lambda x: x.permute(3, 2, 0, 1)
    dm = DataModule(args.datapath, os.path.dirname(args.datapath),
                    args.batchsize,
                    custom_transform_head=transform,
                    custom_transform_tail=transform,
                    name_dataset='Datasetx30',
                    )
    trainer.test(model, datamodule=dm)

  else:
    # mode eval and cuda if available
    model.eval()
    # set to cuda if available else choice another device (multiple gpu)
    if args.devices != -2:
      if args.devices == -1:
        device = 0
      else:
        device = args.devices
      device = torch.device('cuda:{}'.format(device) if torch.cuda.is_available() else 'cpu')
      model.to(device)
    else:
      device = torch.device('cpu')


    from sci.pytorch.Generators.generator import Generator
    import numpy as np
    from tqdm import tqdm
    from scipy.io import savemat
    from skimage.metrics import peak_signal_noise_ratio as compare_psnr
    from skimage.metrics import structural_similarity   as compare_ssim

    if args.save_test:
      if not os.path.exists(f'{datapath}/results_test/{args.model}'):
        os.makedirs(f'{datapath}/results_test/{args.model}')
      print(f'\nCreated {datapath}/results_test/{args.model}\n')

    transform = lambda x: x.permute(3, 2, 0, 1)
    gen = Generator(
      dataset='test',
      type_input='3d',
      type_output='3d',
      name_dataset='Datasetx30',
      custom_transform_head=transform,
      custom_transform_tail=transform,
      )
    flow = gen.flow
    denormalize = gen.undo_normalize

    paths = glob(os.path.join(args.datapath, 'test', '*'))
    pbar  = tqdm(paths)
    all_psnr = 0
    all_ssim = 0
    for path in pbar:
      x, y  = np.load(path)
      y     = np.float32(y)[..., 0]
      image = flow(x)[None]
      image = image.to(device)
      image = image.float()
      with torch.no_grad():
        y_hat = model(image)
        # depermute from (B C S H W) to (B H W S C)
        y_hat = y_hat.permute(0, 3, 4, 2, 1)

      y_hat = y_hat[0].cpu().detach().numpy()
      y_hat = denormalize(y_hat, 'tail')[..., 0]

      if args.save_test:
        savemat(f'{datapath}/results_test/{args.model}/{basename(path)[:-4]}.mat', {'predict': y_hat, 'orig': y})

      psnr = []
      ssim = []
      for channel in range(y.shape[-1]):
        psnr += [compare_psnr(y[..., channel], y_hat[..., channel], data_range=255.)]
        ssim += [compare_ssim(y[..., channel], y_hat[..., channel], data_range=255.)]

      all_psnr += np.mean(psnr)
      all_ssim += np.mean(ssim)



  all_psnr_ = all_psnr / len(paths)
  all_ssim_ = all_ssim / len(paths)

  print(f'\nmodel: {args.model}\n===> PSNR: {all_psnr_:.4f}\n===> SSIM: {all_ssim_ :.6f}\n')

if __name__ == '__main__':
  main()