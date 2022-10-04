import os
import argparse
from glob import glob
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('--model', default='GC_VIT_base', help='Model to use')
ap.add_argument('--resume', default=False, action='store_true', help='Resume training')
ap.add_argument('--batchsize', default=1, type=int, help='batch size')
ap.add_argument('--devices', default=-1, type=int, help='GPU to use (default: -1 mean use all available GPU)')
ap.add_argument('--epochs', default=100, type=int, help='number of epochs')
ap.add_argument('--lr', default=1e-3, type=float, help='learning rate')
ap.add_argument('--normalize_head', default='none', type=str, help='normalize head: none, mean_std, mean_std_min_max or min_max')
ap.add_argument('--normalize_tail', default='none', type=str, help='normalize tail: none, mean_std, mean_std_min_max or min_max')
ap.add_argument('--strategy', default='ddp', type=str, help='strategy to use')
args = vars(ap.parse_args())



from pytorch_lightning import Trainer
from pyramidal.models.pt_wfnet import WFNet
from pyramidal.generators.generator_pt import Generator, data_loader
from pyramidal.models.pt_gc_vit import GC_VIT_Lightning, GC_VIT_Lightning_xxtiny, GC_VIT_Lightning_xxtiny_half
# from pyramidal.models.pt_gc_vit_3d import GC_VIT_3D_Lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.strategies.ddp import DDPStrategy

def main():
  version_num = 0
  if len(glob('lightning_logs/version_*')) > 0:
    version_num = int(sorted(glob('lightning_logs/version_*'))[-1].split('_')[-1]) + 1

  if args['model'] == 'WFNet':
    model_to_use = WFNet
    is_gcvit = False
    name_model = 'WFNet'
  elif args['model'] == 'GC_VIT_base' != -1:
    model_to_use = GC_VIT_Lightning
    is_gcvit = True
    name_model = 'GC_VIT'
  elif args['model'] == 'GC_VIT_XXTINY' != -1:
    model_to_use = GC_VIT_Lightning_xxtiny
    is_gcvit = True
    name_model = 'GC_VIT_XXTINY'
  elif args['model'] == 'GC_VIT_XXTINY_half' != -1:
    model_to_use = GC_VIT_Lightning_xxtiny_half
    is_gcvit = True
    name_model = 'GC_VIT_XXTINY_half'

  checkpoint_callback = ModelCheckpoint(dirpath='./saved_models/',
                                        filename=f'{name_model}-v={version_num:02d}-{{epoch}}-{{val_loss:.2f}}',
                                        mode='min',
                                        save_top_k=1,
                                        monitor='val_loss',
                                        )
  tqdm_callback = TQDMProgressBar()

  trainer = Trainer(accelerator='gpu', devices=args['devices'],
                    max_epochs=args['epochs'],
                    num_sanity_val_steps=2,
                    callbacks=[checkpoint_callback, tqdm_callback],
                    amp_backend='native',
                    strategy = DDPStrategy(find_unused_parameters=False) if args['strategy'] == 'ddp' else args['strategy'],
                    precision=16,
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


  train = Generator('train', batchsize=1,
                    normalize_head=args['normalize_head'],
                    normalize_tail=args['normalize_tail'],
                    is_gcvit=is_gcvit,
                    )

  values_normalize_head = train.values_normalize_head
  values_normalize_tail = train.values_normalize_tail

  val   = Generator('val',   batchsize=1,
                    normalize_head=args['normalize_head'],
                    values_normalize_head=values_normalize_head,
                    normalize_tail=args['normalize_tail'],
                    values_normalize_tail=values_normalize_tail,
                    is_gcvit=is_gcvit,
                    )


  train_data_loader = data_loader(train, batchsize=args['batchsize'], shuffle=True,
                                  num_workers=len(os.sched_getaffinity(0)))
  val_data_loader   = data_loader(val,   batchsize=args['batchsize'], shuffle=False,
                                  num_workers=len(os.sched_getaffinity(0)))


  if not is_gcvit:
    x = next(iter(train_data_loader))[0]
    # create a random batch of input data
    # x = torch.rand(1, 1, 224, 224)
    model.train()
    model.forward(x)

  trainer.fit(model, train_data_loader, val_data_loader)

if __name__ == '__main__':
  main()