import torch
from torch import nn

from sci.pytorch.Models.gcvit.gcvit import GCViT, _to_channel_first, PatchEmbed
from sci.pytorch.Models.regnet.regnet import ResBottleneckBlock_Reversed, AnyStage

class GCVit_URegnet_model(GCViT):
  use_norm = False
  use_path = False
  use_head = False
  use_drop = False

  mode = 'concat'
  bn = False
  attn_transformer = True
  mult_filters = 1.0

  def __init__(self,):
    super().__init__(depths=[3, 4, 19, 5],
                    num_heads=[2, 4, 8, 16],
                    window_size=[8, 8, 16, 8],
                    dim=64,
                    mlp_ratio=3,
                    drop_path_rate=0.0)
    mode = self.mode
    attn_transformer = self.attn_transformer
    assert mode in ['concat', 'add', 'add&concat', 'concat&add']
    if mode.find('&') != -1:
      layers, last_layer = mode.split('&')
      mult_channels = [2 if layers == 'concat' else 1, 2 if last_layer == 'concat' else 1]
    else:
      mult_channels = 2 if mode == 'concat' else 1
      mult_channels = [mult_channels, mult_channels]
    self.mode = mode

    self.patch_embed = PatchEmbed(in_chans=1, dim=64, use_skip=True, use_revsci=True)
    self.conv2 = nn.Sequential(
      nn.ConvTranspose3d(64*mult_channels[-1], 32, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1),
                          output_padding=(0, 1, 1)),
      nn.LeakyReLU(inplace=True),
      nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
      nn.LeakyReLU(inplace=True),
      nn.Conv3d(32, 16, kernel_size=1, stride=1),
      nn.LeakyReLU(inplace=True),
      nn.Conv3d(16, 1, kernel_size=3, stride=1, padding=1),
    )

    parameters = [(96,                      64, 2, 13, ResBottleneckBlock_Reversed, 1,  32, True, self.bn),
                  (128 * mult_channels[0],  96, 2, 10, ResBottleneckBlock_Reversed, 1,  64, True, self.bn),
                  (256 * mult_channels[0], 128, 2,  9, ResBottleneckBlock_Reversed, 1,  64, True, self.bn),
                  (512 * mult_channels[0], 256, 2,  8, ResBottleneckBlock_Reversed, 1, 128, True, self.bn),
                  (512,                    512, 1,  6, ResBottleneckBlock_Reversed, 1, 128, True, self.bn)]
    self.connexions = [1, 1, 1, 2]
    self.reverse_stages = nn.ModuleList()
    for index, (w_in, w_out, stride, d, block_fun_r, bm, gw, se_r, use_bn) in enumerate(reversed(parameters)):
      self.reverse_stages.append(AnyStage(w_in, w_out, stride, d, block_fun_r, bm, gw, se_r, use_bn, attn_transformer))



  def forward_features(self, x, skip_connections=[]):

    for level in self.levels:
      x = level(x)
      skip_connections.append(_to_channel_first(x))

    x = _to_channel_first(x)
    return x, skip_connections

  def forward(self, x):
    x, skip_connections = self.patch_embed(x)
    # skip_connections.append(x)
    x, skip_connections = self.forward_features(x, skip_connections)
    skip_connections.pop(-1)

    index_stages = 0
    for index, (skip, conexion_index) in enumerate(zip(list(reversed(skip_connections)), self.connexions)):
      stage_r_list = self.reverse_stages[index_stages:index_stages+conexion_index]
      index_stages += conexion_index
      for stage_r in stage_r_list:
        x = stage_r(x)
      if self.mode == 'concat':
        x = torch.cat([x, skip], dim=1)
      elif self.mode == 'add':
        x = x + skip
      elif self.mode.find('&') != -1:
        layer, last_layer = self.mode.split('&')
        if index == len(skip_connections) - 1:
          if last_layer == 'concat':
            x = torch.cat([x, skip], dim=1)
          else:
            x = x + skip
        else:
          if layer == 'concat':
            x = torch.cat([x, skip], dim=1)
          else:
            x = x + skip
      else:
        raise ValueError('Unknown mode')

    x = self.conv2(x)

    return x

if __name__ == '__main__':
  model = GCVit_URegnet_model()
  x = torch.randn(1, 1, 16, 256, 256) * 255
  y = model(x)
  print(y.shape)