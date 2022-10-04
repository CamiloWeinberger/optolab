import os
import torch
from tqdm import tqdm
from pyramidal.models.pt_wfnet import WFNet
from pyramidal.models.pt_gc_vit import GC_VIT_Lightning, GC_VIT_Lightning_xxtiny

import argparse
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('--model', default='GC_VIT', type=str, help='Model to use: WFNet or GC_VIT')
ap.add_argument('--path_model', default='GC_VIT-v=00-epoch=149-val_loss=2.39.ckpt', type=str, help='Path to model')
args = vars(ap.parse_args())

if args['model'] == 'WFNet':
  model_to_use = WFNet
  is_gcvit = False
  name_model = 'WFNet'
elif args['model'] == 'GC_VIT' != -1:
  model_to_use = GC_VIT_Lightning
  is_gcvit = True
  name_model = 'GC_VIT'
elif args['model'] == 'GC_VIT_XXTINY' != -1:
  model_to_use = GC_VIT_Lightning_xxtiny
  is_gcvit = True
  name_model = 'GC_VIT_XXTINY'


path_weight = f"{os.path.dirname(os.path.abspath(__file__))}/saved_models/{args['path_model']}"
model = model_to_use.load_from_checkpoint(path_weight)
print(f'Loaded model from {path_weight}')

script = model.to_torchscript()

# save for use in production environment
torch.jit.save(script, f"{os.path.dirname(os.path.abspath(__file__))}/saved_models/{args['path_model'][0:-4]}pt")