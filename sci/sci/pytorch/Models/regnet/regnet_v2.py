import torch
import torch.nn as nn
import math
import numpy as np

class SE(nn.Module):
  """Squeeze-and-Excitation (SE) block: AvgPool, FC, ReLU, FC, Sigmoid."""

  def __init__(self, w_in, w_se, attn_transformer=False, activation=nn.LeakyReLU(inplace=True), multi_head_num=4, ):
    super(SE, self).__init__()
    self.avg_pool = nn.AdaptiveAvgPool3d((16, 1, 1))
    self.attn_transformer = attn_transformer
    self.sigmoid = nn.Sigmoid()
    if not attn_transformer:
      self.f_ex = nn.Sequential(
        nn.Linear(w_in, w_se),
        activation,
        nn.Linear(w_se, w_in),
      )
    else:
      self.f_ex = nn.TransformerEncoder(nn.TransformerEncoderLayer(w_in, multi_head_num, w_se, batch_first=True, norm_first=True), multi_head_num // 2)

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

  def __init__(self, w_in, w_out, stride, bm, gw, se_r, use_bn, attn_transformer=False, activation=nn.ReLU(inplace=True), multi_head_num=4):
    super(BottleneckTransform, self).__init__()
    if w_out > w_in:
      w_b = int(round(w_out * bm))
    else:
      w_b = int(round(w_in * bm))
    g = w_b // gw if w_b // gw > 0 else 1
    self.a      = nn.Conv3d(w_in, w_b, 1, stride=1, padding=0, bias=False)
    if use_bn:
      self.a_bn   = nn.BatchNorm3d(w_b)
    else:
      self.a_bn   = nn.Identity()
    self.a_relu = activation
    self.b      = nn.Conv3d(w_b, w_b, 3, stride=(1, stride, stride), padding=1, groups=g, bias=False)
    if use_bn:
      self.b_bn   = nn.BatchNorm3d(w_b)
    else:
      self.b_bn   = nn.Identity()
    self.b_relu = activation
    if se_r:
      w_se    = int(round(w_in * se_r))
      self.se = SE(w_b, w_se, attn_transformer, multi_head_num)
    else:
      self.se = nn.Identity()
    self.c    = nn.Conv3d(w_b, w_out, 1, stride=1, padding=0, bias=False)
    if use_bn:
      self.c_bn = nn.BatchNorm3d(w_out)
      self.c_bn.final_bn = True
    else:
      self.c_bn = nn.Identity()

  def forward(self, x):
    x = self.a(x)
    x = self.a_bn(x)
    x = self.a_relu(x)
    x = self.b(x)
    x = self.b_bn(x)
    x = self.b_relu(x)
    x = self.se(x)
    x = self.c(x)
    x = self.c_bn(x)

    return x

class BottleneckTransform_Reversed(nn.Module):
  """Bottlenect transformation reverse: 1x1, 3x3 [+SE], 1x1"""

  def __init__(self, w_in, w_out, stride, bm, gw, se_r, use_bn, use_attn_transformer, activation=nn.ReLU(inplace=True), multi_head_num=4):
    super().__init__()
    if w_out > w_in:
      w_b = int(round(w_out * bm))
    else:
      w_b = int(round(w_in * bm))
    g = w_b // gw if w_b // gw > 0 else 1
    self.a      = nn.Conv3d(w_in, w_b, 1, stride=1, padding=0, bias=False)
    if use_bn:
      self.a_bn   = nn.BatchNorm3d(w_b)
    self.a_relu = activation
    if stride == 1:
      self.b    = nn.Conv3d(w_b, w_b, 3, stride=(1, stride, stride), padding=1, groups=g, bias=False)
    else:
      self.b    = nn.ConvTranspose3d(w_b, w_b, kernel_size=(1, 3, 3), stride=(1, stride, stride), padding=(0, 1, 1), output_padding=(0, 1, 1), groups=g, bias=False)
    if use_bn:
      self.b_bn   = nn.BatchNorm3d(w_b)
    self.b_relu = activation
    if se_r:
      w_se    = int(round(w_in * se_r))
      self.se = SE(w_b, w_se, use_attn_transformer, multi_head_num)
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

  def __init__(self, w_in, w_out, stride, bm=1.0, gw=1, se_r=None, use_bn=True, attn_transformer=False, activation=nn.ReLU(inplace=True), multi_head_num=4):
    super(ResBottleneckBlock, self).__init__()
    # Use skip connection with projection if shape changes
    self.proj_block = (w_in != w_out) or (stride != 1)
    if self.proj_block:
        self.proj = nn.Conv3d(w_in, w_out, 1, stride=(1, stride, stride), padding=0, bias=False)
        if use_bn:
          self.bn = nn.BatchNorm3d(w_out)
        else:
          self.bn = nn.Identity()
    self.f    = BottleneckTransform(w_in, w_out, stride, bm, gw, se_r, use_bn, attn_transformer, activation, multi_head_num)
    self.relu = activation

  def forward(self, x):
    if self.proj_block:
      x = (self.bn(self.proj(x)) + self.f(x)) / 2
    else:
      x = (x + self.f(x)) / 2
    x = self.relu(x)
    return x

class ResBottleneckBlock_Reversed(nn.Module):
  """Residual bottleneck block: x + F(x), F = bottleneck transform reversed"""

  def __init__(self, w_in, w_out, stride, bm=1.0, gw=1, se_r=None, use_bn=True, use_attn_transformer=False, activation=nn.ReLU(inplace=True), multi_head_num=4):
    super(ResBottleneckBlock_Reversed, self).__init__()
    # Use skip connection with projection if shape changes
    self.proj_block = (w_in != w_out) or (stride != 1)
    if self.proj_block:
      self.proj = nn.ConvTranspose3d(w_in, w_out, 1, stride=(1, stride, stride), padding=0, output_padding=(0, 1, 1), bias=False)
      if use_bn:
        self.bn = nn.BatchNorm3d(w_out)
      else:
        self.bn = nn.Identity()
    self.f    = BottleneckTransform_Reversed(w_in, w_out, stride, bm, gw, se_r, use_bn, use_attn_transformer, activation, multi_head_num)
    self.relu = activation

  def forward(self, x):
    if self.proj_block:
      fn = self.f(x)
      x = (self.bn(self.proj(x)) + fn) / 2
    else:
      x = (x + self.f(x)) / 2
    x = self.relu(x)
    return x

class AnyStage(nn.Module):
  """AnyNet stage (sequence of blocks w/ the same output shape)."""

  def __init__(self, w_in, w_out, stride, d, block_fun, bm, gw, se_r, use_bn, use_attn_transformer, activation, multi_head_num):
    super(AnyStage, self).__init__()
    for i in range(d):
      b_stride = stride if i == 0 else 1
      b_w_in   = w_in if i == 0 else w_out
      name     = "b{}".format(i + 1)
      self.add_module(name, block_fun(b_w_in, w_out, b_stride, bm, gw, se_r, use_bn, use_attn_transformer, activation, multi_head_num))

  def forward(self, x):
    for block in self.children():
      x = block(x)
    return x

class RegUNet_model(nn.Module):
  bn = False
  mode = 'concat'
  attn_transformer = True
  mult_filters = 1.
  activation_fn = 'GELU'
  loops = [4, 6, 10, 6]
  filters = [224, 448, 896, 2240]
  groups = 112
  multi_head_num = 4
  se_r = 2.
  def __init__(self):
    super().__init__()
    kwargs_activation = {'inplace': True} if self.activation_fn != 'GELU' else None
    if self.activation_fn == 'GELU':
      self.activation = getattr(nn, self.activation_fn)()
    else:
      self.activation = getattr(nn, self.activation_fn)(**kwargs_activation)

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
      self.activation,
      nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
      self.activation,
      nn.Conv3d(32, 32, kernel_size=1, stride=1),
      self.activation,
      nn.Conv3d(32, 64, kernel_size=3, stride=(1, 2, 2), padding=1),
      self.activation,
    )
    self.conv2 = nn.Sequential(
      nn.ConvTranspose3d(64*mult_channels[-1], 32, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1),
                          output_padding=(0, 1, 1)),
      self.activation,
      nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
      self.activation,
      nn.Conv3d(32, 16, kernel_size=1, stride=1),
      self.activation,
      nn.Conv3d(16, 1, kernel_size=3, stride=1, padding=1),
    )

    parameters = [(64,                                      int(self.filters[0] * self.mult_filters), 2, self.loops[0], ResBottleneckBlock, ResBottleneckBlock_Reversed, 1, int(self.groups  * self.mult_filters), self.se_r, self.bn),
                 (int(self.filters[0] * self.mult_filters), int(self.filters[1] * self.mult_filters), 2, self.loops[1], ResBottleneckBlock, ResBottleneckBlock_Reversed, 1, int(self.groups  * self.mult_filters), self.se_r, self.bn),
                 (int(self.filters[1] * self.mult_filters), int(self.filters[2] * self.mult_filters), 2, self.loops[2], ResBottleneckBlock, ResBottleneckBlock_Reversed, 1, int(self.groups  * self.mult_filters), self.se_r, self.bn),
                 (int(self.filters[2] * self.mult_filters), int(self.filters[3] * self.mult_filters), 2, self.loops[3], ResBottleneckBlock, ResBottleneckBlock_Reversed, 1, int(self.groups  * self.mult_filters), self.se_r, self.bn)]
    self.stages = nn.ModuleList()
    self.reverse_stages = nn.ModuleList()
    for index, (w_in, w_out, stride, d, block_fun, block_fun_r, bm, gw, se_r, use_bn) in enumerate(parameters):
      self.stages.append(AnyStage(w_in, w_out, stride, d, block_fun, bm, gw, se_r, use_bn, attn_transformer, self.activation, self.multi_head_num))
      if index == len(parameters) - 1:
        self.reverse_stages.append(AnyStage(w_out, w_in, stride, d, block_fun_r, bm, gw, se_r, use_bn, attn_transformer, self.activation, self.multi_head_num))
      else:
        self.reverse_stages.append(AnyStage(w_out * mult_channels[0], w_in, stride, d, block_fun_r, bm, gw, se_r, use_bn, attn_transformer, self.activation, self.multi_head_num))


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
        x = (x + skip) / 2
      elif self.mode.find('&') != -1:
        layer, last_layer = self.mode.split('&')
        if index == len(self.stages) - 1:
          if last_layer == 'concat':
            x = torch.cat([x, skip], dim=1)
          else:
            x = (x + skip) / 2
        else:
          if layer == 'concat':
            x = torch.cat([x, skip], dim=1)
          else:
            x = (x + skip) / 2
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
      self.mult_filters = .25
      self.activation_fn = 'GELU'
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

