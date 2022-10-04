import torch

from pyramidal.v2.Models.gcvit import gcvit

class gcvit_xxtiny_half(gcvit):
  def __init__(self, datapath=''):
    self.multiply_filters = 0.5
    super().__init__('gc_vit_xxtiny', datapath=datapath)

if __name__ == '__main__':
  model = gcvit_xxtiny_half()
  x = torch.randn(1, 3, 224, 224)
  y = model(x)
  print(y.shape)