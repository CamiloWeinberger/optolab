import torch
import torch.nn as nn
import math
import numpy as np


class ResStemIN(nn.Module):
  """ResNet stem for ImageNet: 7x7, BN, ReLU, MaxPool."""

  def __init__(self, w_in, w_out):
    super(ResStemIN, self).__init__()
    self.conv = nn.Conv3d(w_in, w_out, 7, stride=2, padding=3, bias=False)
    self.bn   = nn.BatchNorm3d(w_out)
    self.relu = nn.LeakyReLU(inplace=True)
    self.pool = nn.MaxPool3d(3, stride=(2,2,1), padding=1)

  def forward(self, x):
    for layer in self.children():
      x = layer(x)
    return x

class ResStemOut(nn.Module):
  """ResNet stem for ImageNet: 7x7, BN, ReLU, MaxPool.
  Up image """

  def __init__(self, w_in, w_out):
    super(ResStemOut, self).__init__()
    self.conv = nn.ConvTranspose3d(w_in, w_out, 7, stride=2, padding=3, bias=False)
    self.bn   = nn.BatchNorm3d(w_out)
    self.relu = nn.LeakyReLU(inplace=True)
    self.pool = nn.MaxUnpool3d(3, stride=(2,2,1), padding=1)

  def forward(self, x):
    for layer in self.children():
      x = layer(x)
    return x


class SimpleStemIN(nn.Module):
  """Simple stem for ImageNet: 3x3, BN, ReLU."""

  def __init__(self, in_w, out_w):
    super(SimpleStemIN, self).__init__()
    self.conv = nn.Conv3d(in_w, out_w, 3, stride=(1,2,2), padding=1, bias=False)
    self.bn   = nn.BatchNorm3d(out_w)
    self.relu = nn.LeakyReLU(inplace=True)

  def forward(self, x):
    for layer in self.children():
      x = layer(x)
    return x


class SE(nn.Module):
  """Squeeze-and-Excitation (SE) block: AvgPool, FC, ReLU, FC, Sigmoid."""

  def __init__(self, w_in, w_se, attn_transformer=False):
    super(SE, self).__init__()
    self.avg_pool = nn.AdaptiveAvgPool3d((16, 1, 1))
    self.attn_transformer = attn_transformer
    self.sigmoid = nn.Sigmoid()
    if not attn_transformer:
      self.f_ex = nn.Sequential(
        nn.Linear(w_in, w_se),
        nn.LeakyReLU(inplace=True),
        nn.Linear(w_se, w_in),
      )
    else:
      self.f_ex = nn.TransformerEncoder(nn.TransformerEncoderLayer(w_in, 4), 2)

  def forward(self, x):
    avg = self.avg_pool(x)
    # flatten
    avg  = avg.view(avg.size(0), 16, -1)
    # transform
    attn = self.f_ex(avg)
    attn = self.sigmoid(attn)
    attn = attn.view(attn.size(0), attn.size(-1), attn.size(1), 1, 1)
    return x * attn


class BottleneckTransform(nn.Module):
  """Bottlenect transformation: 1x1, 3x3 [+SE], 1x1"""

  def __init__(self, w_in, w_out, stride, bm, gw, se_r, use_bn, attn_transformer=False):
    super(BottleneckTransform, self).__init__()
    if w_out > w_in:
      w_b = int(round(w_out * bm))
    else:
      w_b = int(round(w_in * bm))
    g = w_b // gw if w_b // gw > 0 else 1
    self.a      = nn.Conv3d(w_in, w_b, 1, stride=1, padding=0, bias=False)
    if use_bn:
      self.a_bn   = nn.BatchNorm3d(w_b)
    self.a_relu = nn.LeakyReLU(inplace=True)
    self.b      = nn.Conv3d(w_b, w_b, 3, stride=(1, stride, stride), padding=1, groups=g, bias=False)
    if use_bn:
      self.b_bn   = nn.BatchNorm3d(w_b)
    self.b_relu = nn.LeakyReLU(inplace=True)
    if se_r:
      w_se    = int(round(w_in * se_r))
      self.se = SE(w_b, w_se, attn_transformer)
    self.c    = nn.Conv3d(w_b, w_out, 1, stride=1, padding=0, bias=False)
    if use_bn:
      self.c_bn = nn.BatchNorm3d(w_out)
      self.c_bn.final_bn = True

  def forward(self, x):
    for layer in self.children():
      x = layer(x)
    return x

class BottleneckTransform_Reversed(nn.Module):
  """Bottlenect transformation reverse: 1x1, 3x3 [+SE], 1x1"""

  def __init__(self, w_in, w_out, stride, bm, gw, se_r, use_bn, use_attn_transformer):
    super().__init__()
    if w_out > w_in:
      w_b = int(round(w_out * bm))
    else:
      w_b = int(round(w_in * bm))
    g = w_b // gw if w_b // gw > 0 else 1
    self.a      = nn.Conv3d(w_in, w_b, 1, stride=1, padding=0, bias=False)
    if use_bn:
      self.a_bn   = nn.BatchNorm3d(w_b)
    self.a_relu = nn.LeakyReLU(inplace=True)
    if stride == 1:
      self.b    = nn.Conv3d(w_b, w_b, 3, stride=(1, stride, stride), padding=1, groups=g, bias=False)
    else:
      self.b    = nn.ConvTranspose3d(w_b, w_b, kernel_size=(1, 3, 3), stride=(1, stride, stride), padding=(0, 1, 1), output_padding=(0, 1, 1), groups=g, bias=False)
    if use_bn:
      self.b_bn   = nn.BatchNorm3d(w_b)
    self.b_relu = nn.LeakyReLU(inplace=True)
    if se_r:
      w_se    = int(round(w_in * se_r))
      self.se = SE(w_b, w_se, use_attn_transformer)
    self.c    = nn.Conv3d(w_b, w_out, 1, stride=1, padding=0, bias=False)
    if use_bn:
      self.c_bn = nn.BatchNorm3d(w_out)
      self.c_bn.final_bn = True

  def forward(self, x):
    for layer in self.children():
      x = layer(x)
    return x


class ResBottleneckBlock(nn.Module):
  """Residual bottleneck block: x + F(x), F = bottleneck transform"""

  def __init__(self, w_in, w_out, stride, bm=1.0, gw=1, se_r=None, use_bn=True, attn_transformer=False):
    super(ResBottleneckBlock, self).__init__()
    # Use skip connection with projection if shape changes
    self.proj_block = (w_in != w_out) or (stride != 1)
    if self.proj_block:
        self.proj = nn.Conv3d(w_in, w_out, 1, stride=(1, stride, stride), padding=0, bias=False)
        if use_bn:
          self.bn = nn.BatchNorm3d(w_out)
        else:
          self.bn = nn.Identity()
    self.f    = BottleneckTransform(w_in, w_out, stride, bm, gw, se_r, use_bn, attn_transformer)
    self.relu = nn.LeakyReLU(inplace=True)

  def forward(self, x):
    if self.proj_block:
      x = self.bn(self.proj(x)) + self.f(x)
    else:
      x = x + self.f(x)
    x = self.relu(x)
    return x

class ResBottleneckBlock_Reversed(nn.Module):
  """Residual bottleneck block: x + F(x), F = bottleneck transform reversed"""

  def __init__(self, w_in, w_out, stride, bm=1.0, gw=1, se_r=None, use_bn=True, use_attn_transformer=False):
    super(ResBottleneckBlock_Reversed, self).__init__()
    # Use skip connection with projection if shape changes
    self.proj_block = (w_in != w_out) or (stride != 1)
    if self.proj_block:
      self.proj = nn.ConvTranspose3d(w_in, w_out, 1, stride=(1, stride, stride), padding=0, output_padding=(0, 1, 1), bias=False)
      if use_bn:
        self.bn = nn.BatchNorm3d(w_out)
      else:
        self.bn = nn.Identity()
    self.f    = BottleneckTransform_Reversed(w_in, w_out, stride, bm, gw, se_r, use_bn, use_attn_transformer)
    self.relu = nn.LeakyReLU(inplace=True)

  def forward(self, x):
    if self.proj_block:
      fn = self.f(x)
      x = self.bn(self.proj(x)) + fn
    else:
      x = x + self.f(x)
    x = self.relu(x)
    return x



class AnyHead(nn.Module):
  """AnyNet head: AvgPool, 1x1."""

  def __init__(self, w_in, nc):
    super(AnyHead, self).__init__()
    self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
    self.fc = nn.Linear(w_in, nc, bias=True)

  def forward(self, x):
    x = self.avg_pool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    return x


class AnyStage(nn.Module):
  """AnyNet stage (sequence of blocks w/ the same output shape)."""

  def __init__(self, w_in, w_out, stride, d, block_fun, bm, gw, se_r, use_bn, use_attn_transformer):
    super(AnyStage, self).__init__()
    for i in range(d):
      b_stride = stride if i == 0 else 1
      b_w_in   = w_in if i == 0 else w_out
      name     = "b{}".format(i + 1)
      self.add_module(name, block_fun(b_w_in, w_out, b_stride, bm, gw, se_r, use_bn, use_attn_transformer))

  def forward(self, x):
    for block in self.children():
      x = block(x)
    return x

class AnyNet(nn.Module):
  """AnyNet model."""

  def __init__(self, **kwargs):
    super(AnyNet, self).__init__()
    if kwargs:
      self._construct(
          stem_w=kwargs["stem_w"],
          ds=kwargs["ds"],
          ws=kwargs["ws"],
          ss=kwargs["ss"],
          bms=kwargs["bms"],
          gws=kwargs["gws"],
          se_r=kwargs["se_r"],
          nc=kwargs["nc"],
      )
    for m in self.modules():
      if isinstance(m, nn.Conv3d):
        # Note that there is no bias due to BN
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(mean=0.0, std=math.sqrt(2.0 / fan_out))
      elif isinstance(m, nn.BatchNorm3d):
        zero_init_gamma = hasattr(m, "final_bn") and m.final_bn
        m.weight.data.fill_(0.0 if zero_init_gamma else 1.0)
        m.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        m.weight.data.normal_(mean=0.0, std=0.01)
        m.bias.data.zero_()

  def _construct(self, stem_w, ds, ws, ss, bms, gws, se_r, nc):
    # Generate dummy bot muls and gs for models that do not use them
    bms = bms if bms else [None for _d in ds]
    gws = gws if gws else [None for _d in ds]
    stage_params = list(zip(ds, ws, ss, bms, gws))
    self.stem = SimpleStemIN(1, stem_w)
    prev_w = stem_w
    self.stages = nn.ModuleList()
    for i, (d, w, s, bm, gw) in enumerate(stage_params):
      # name = "s{}".format(i + 1)
      self.stages.append(AnyStage(prev_w, w, s, d, ResBottleneckBlock, bm, gw, se_r))
      # self.add_module(name, AnyStage(prev_w, w, s, d, ResBottleneckBlock, bm, gw, se_r))
      prev_w = w
    self.head = AnyHead(w_in=prev_w, nc=nc)

  def forward(self, x):
    x = self.stem(x)
    shapes_stage = [x.shape]
    for stage in self.stages:
      x = stage(x)
      shapes_stage.append(x.shape)
    x = self.head(x)
    return x

class RegUNet_model(nn.Module):
  bn = True
  mode = 'concat'
  attn_transformer = False
  mult_filters = 1.
  def __init__(self):
    super().__init__()
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

    self.conv1 = nn.Sequential(
      nn.Conv3d(1, 16, kernel_size=5, stride=1, padding=2),
      nn.LeakyReLU(inplace=True),
      nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
      nn.LeakyReLU(inplace=True),
      nn.Conv3d(32, 32, kernel_size=1, stride=1),
      nn.LeakyReLU(inplace=True),
      nn.Conv3d(32, 64, kernel_size=3, stride=(1, 2, 2), padding=1),
      nn.LeakyReLU(inplace=True),
    )
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

    parameters = [(64,                          int(224  * self.mult_filters), 2,  2, ResBottleneckBlock, ResBottleneckBlock_Reversed, 1, 112, True, self.bn),
                 (int(224 * self.mult_filters), int(448  * self.mult_filters), 2,  5, ResBottleneckBlock, ResBottleneckBlock_Reversed, 1, 112, True, self.bn),
                 (int(448 * self.mult_filters), int(896  * self.mult_filters), 2, 11, ResBottleneckBlock, ResBottleneckBlock_Reversed, 1, 112, True, self.bn),
                 (int(896 * self.mult_filters), int(2240 * self.mult_filters), 2,  1, ResBottleneckBlock, ResBottleneckBlock_Reversed, 1, 112, True, self.bn)]
    self.stages = nn.ModuleList()
    self.reverse_stages = nn.ModuleList()
    for index, (w_in, w_out, stride, d, block_fun, block_fun_r, bm, gw, se_r, use_bn) in enumerate(parameters):
      self.stages.append(AnyStage(w_in, w_out, stride, d, block_fun, bm, gw, se_r, use_bn, attn_transformer))
      if index == len(parameters) - 1:
        self.reverse_stages.append(AnyStage(w_out, w_in, stride, d, block_fun_r, bm, gw, se_r, use_bn, attn_transformer))
      else:
        self.reverse_stages.append(AnyStage(w_out * mult_channels[0], w_in, stride, d, block_fun_r, bm, gw, se_r, use_bn, attn_transformer))


  def forward(self, x):
    x = self.conv1(x)
    shapes = [x.shape]
    skip_connections = [x]
    for index, stage in enumerate(self.stages):
      x = stage(x)
      shapes.append(x.shape)
      if index != len(self.stages) - 1:
        skip_connections.append(x)

    for index, (skip, stage_r) in enumerate(zip(reversed(skip_connections), reversed(self.reverse_stages))):
      x = stage_r(x)
      if self.mode == 'concat':
        x = torch.cat([x, skip], dim=1)
      elif self.mode == 'add':
        x = x + skip
      elif self.mode.find('&') != -1:
        layer, last_layer = self.mode.split('&')
        if index == len(self.stages) - 1:
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
  x = torch.randn(1, 1, 16, 256, 256)

  class RegUNetT(RegUNet_model):
    def __init__(self):
      self.bn = False
      self.mode = 'concat'
      self.attn_transformer = True
      super().__init__()
  model = RegUNetT()
  y = model(x)
  print(y.shape)

  # model = RegUNet_model('concat')
  # y = model(x)
  # print(y.shape)

  # del model
  # torch.cuda.empty_cache()

  # model = RegUNet_model('add')
  # y = model(x)
  # print(y.shape)

