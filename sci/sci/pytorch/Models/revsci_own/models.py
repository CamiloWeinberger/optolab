from sci.pytorch.Models.revsci_own.my_tools import *

class re_3dcnn_tiny(nn.Module):
  def __init__(self,):
    super().__init__()
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
      nn.ConvTranspose3d(64, 32, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1),
                          output_padding=(0, 1, 1)),
      nn.LeakyReLU(inplace=True),
      nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
      nn.LeakyReLU(inplace=True),
      nn.Conv3d(32, 16, kernel_size=1, stride=1),
      nn.LeakyReLU(inplace=True),
      nn.Conv3d(16, 1, kernel_size=3, stride=1, padding=1),
    )

  def forward(self, data):
    out = self.conv1(data)

    out = self.conv2(out)

    return out

class re_3dcnn(nn.Module):

  def __init__(self):
    super(re_3dcnn, self).__init__()
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
      nn.ConvTranspose3d(64, 32, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1),
                          output_padding=(0, 1, 1)),
      nn.LeakyReLU(inplace=True),
      nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
      nn.LeakyReLU(inplace=True),
      nn.Conv3d(32, 16, kernel_size=1, stride=1),
      nn.LeakyReLU(inplace=True),
      nn.Conv3d(16, 1, kernel_size=3, stride=1, padding=1),
    )

    self.layers = nn.ModuleList()
    for _ in range(18):
      self.layers.append(rev_3d_part1(64, 2))

  def forward(self, data):
    out = self.conv1(data)

    for layer in self.layers:
      out = layer(out)

    out = self.conv2(out)

    return out

class re_3dcnn1(nn.Module):
  def __init__(self):
    super().__init__()
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
      nn.ConvTranspose3d(64, 32, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1),
                          output_padding=(0, 1, 1)),
      nn.LeakyReLU(inplace=True),
      nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
      nn.LeakyReLU(inplace=True),
      nn.Conv3d(32, 16, kernel_size=1, stride=1),
      nn.LeakyReLU(inplace=True),
      nn.Conv3d(16, 1, kernel_size=3, stride=1, padding=1),
    )

    self.layers = nn.ModuleList()
    for _ in range(18):
      self.layers.append(rev_3d_part(32))

  def forward(self, meas_re):

    batch_size = meas_re.shape[0]
    mask = self.mask.to(meas_re.device)
    maskt = mask.expand([batch_size, 8, 256, 256])
    maskt = maskt.mul(meas_re)
    data = meas_re + maskt
    out = self.conv1(torch.unsqueeze(data, 1))

    for layer in self.layers:
        out = layer(out)

    out = self.conv2(out)

    return out
