import os
from os.path import basename, dirname, join
from glob import glob
from scipy.io import loadmat

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sci.pytorch.Models._base import ModelBase

from sci.pytorch.Models.Losses import MAELoss

def A_operator(z, Phi):
    y = torch.sum(Phi * z, 1, keepdim=True)
    return y


def At_operator(z, Phi):
    y = z * Phi
    return y



class InputCvBlock(nn.Module):
    '''(Conv with num_in_frames groups => BN => ReLU) + (Conv => BN => ReLU)'''

    def __init__(self, num_in_frames, out_ch):
        super(InputCvBlock, self).__init__()
        self.interm_ch = 30
        self.convblock = nn.Sequential(
            nn.Conv3d(num_in_frames, num_in_frames * self.interm_ch, \
                      kernel_size=3, padding=1, groups=num_in_frames, bias=True),
            # nn.BatchNorm2d(num_in_frames * self.interm_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(num_in_frames * self.interm_ch, out_ch, kernel_size=3, padding=1, bias=True),
            # nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convblock(x)


class CvBlock(nn.Module):
    '''(Conv2d => BN => ReLU) x 2'''

    def __init__(self, in_ch, out_ch):
        super(CvBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=True),
            # nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=True),
            # nn.BatchNorm2d(out_ch),
            # nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convblock(x) + x


class DownBlock(nn.Module):
    '''Downscale + (Conv2d => BN => ReLU)*2'''

    def __init__(self, in_ch, out_ch):
        super(DownBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, stride=(1, 2, 2), bias=True),
            # nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            CvBlock(out_ch, out_ch)
        )

    def forward(self, x):
        return self.convblock(x)


class UpBlock(nn.Module):
    '''(Conv2d => BN => ReLU)*2 + Upscale'''

    def __init__(self, in_ch, out_ch):
        super(UpBlock, self).__init__()
        self.convblock = nn.Sequential(
            CvBlock(in_ch, in_ch),
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=True)
        )

    def forward(self, x):
        return F.interpolate(self.convblock(x), scale_factor=(1, 2, 2))


class OutputCvBlock(nn.Module):
    '''Conv2d => BN => ReLU => Conv2d'''

    def __init__(self, in_ch, out_ch):
        super(OutputCvBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, kernel_size=3, padding=1, bias=True),
            # nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=True)
        )

    def forward(self, x):
        return self.convblock(x)


class BasicBlock(torch.nn.Module):
    def __init__(self, args, layer):
        super(BasicBlock, self).__init__()

        self.eta_step = nn.Parameter(torch.Tensor([0.01]))
        # define head module

        self.n_channels = args.n_channels

        self.chs_lyr0 = 32 // 2
        self.chs_lyr1 = 64 // 2
        self.chs_lyr2 = 128 // 2
        self.k = 3

        self.inc = InputCvBlock(num_in_frames=args.n_channels + 1, out_ch=self.chs_lyr0)
        self.downc0 = DownBlock(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1)
        self.downc1 = DownBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr2)
        if layer == 0:
            self.upc2 = UpBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr1)
            self.upc1 = UpBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr0)
            self.outc = OutputCvBlock(in_ch=self.chs_lyr0, out_ch=1)
        else:
            self.upc2 = UpBlock(in_ch=self.chs_lyr2 * 2, out_ch=self.chs_lyr1)
            self.upc1 = UpBlock(in_ch=self.chs_lyr1 * 2, out_ch=self.chs_lyr0)
            self.outc = OutputCvBlock(in_ch=self.chs_lyr0 * 2, out_ch=1)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.unfold = nn.Unfold(kernel_size=3, stride=1, padding=1)
        self.generate_kernel1 = nn.Conv2d(in_channels=1, out_channels=self.chs_lyr2 * self.k * self.k, kernel_size=5, padding=1, stride=4, bias=None)
        self.generate_kernel2 = nn.Conv2d(in_channels=1, out_channels=self.chs_lyr1 * self.k * self.k, kernel_size=3, padding=1, stride=2, bias=None)
        self.generate_kernel3 = nn.Conv2d(in_channels=1, out_channels=self.chs_lyr0 * self.k * self.k, kernel_size=3, padding=1, stride=1, bias=None)



    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, v, y, y1, mask, Phisum, h_pre):

        yb = A_operator(v, mask)
        y1 = y1 + (y - yb)
        # x = v + At_operator((y1-yb)/Phisum, mask)
        x = v + At_operator((y1 - yb) / (Phisum + self.eta_step), mask)  ## torch.Size([1, 8, 1, 128, 128])

        Phisum[Phisum == 0] = 1  # torch.Size([1, 1, 128, 128])
        y_ = y / Phisum  # torch.Size([1, 1, 1, 128, 128])
        y_ = y_.expand_as(v)  # torch.Size([1, 8, 1, 128, 128])

        x_d0, x_d1, x_d2, x_d3, x_d4 = x.shape

        # # Spatial Denoising
        v = x.contiguous().view(x_d0, x_d2, x_d1, x_d3, x_d4)  # torch.Size([1, 1, 8, 128, 128])
        y_ = y_.contiguous().view(x_d0, x_d2, x_d1, x_d3, x_d4)  # torch.Size([1, 1, 8, 128, 128])

        h_curr = []

        _xcat = torch.cat((v, y_), dim=1)
        x0 = self.inc(_xcat)  # torch.Size([1, 32, 8, 128, 128])
        x1 = self.downc0(x0)  # torch.Size([1, 64, 8, 64, 64])
        x2 = self.downc1(x1)  # torch.Size([1, 128, 8, 32, 32])
        h_curr.append(x2)

        y_ = y_[:, :, 0, :, :]
        if len(h_pre) == 0:
            x2 = self.upc2(x2)
            h_curr.append(x2)
            x1 = self.upc1(x1 + x2)  # torch.Size([1, 32, 8, 128, 128])
            h_curr.append(x1)
            x = self.outc(x0 + x1)
        else:
            N, xC, xD, xH, xW = h_pre[0].size()
            kernel = self.relu(self.generate_kernel1(y_)).reshape([N, xC, self.k ** 2, xH, xW])
            similarity = []
            for i in range(xD):
                unfold_hpre = self.unfold(h_pre[0][:, :, i, :, :]).reshape([N, xC, -1, xH, xW])
                sim = self.sigmoid((kernel * unfold_hpre).sum(2))
                similarity.append(sim)
            similarity = torch.stack(similarity, dim=2) #torch.Size([1, 128, 8, 32, 32])
            h_pre[0] = similarity * h_pre[0]

            x2 = self.upc2(torch.cat((x2, h_pre[0]), dim=1))
            h_curr.append(x2)

            N, xC, xD, xH, xW = h_pre[1].size()
            kernel = self.relu(self.generate_kernel2(y_)).reshape([N, xC, self.k ** 2, xH, xW])
            similarity = []
            for i in range(xD):
                unfold_hpre = self.unfold(h_pre[1][:, :, i, :, :]).reshape([N, xC, -1, xH, xW])
                sim = self.sigmoid((kernel * unfold_hpre).sum(2))
                similarity.append(sim)
            similarity = torch.stack(similarity, dim=2)  # torch.Size([1, 128, 8, 32, 32])
            h_pre[1] = similarity * h_pre[1]
            x1 = self.upc1(torch.cat((x1 + x2, h_pre[1]), dim=1))
            h_curr.append(x1)

            N, xC, xD, xH, xW = h_pre[2].size()
            kernel = self.relu(self.generate_kernel3(y_)).reshape([N, xC, self.k ** 2, xH, xW])
            similarity = []
            for i in range(xD):
                unfold_hpre = self.unfold(h_pre[2][:, :, i, :, :]).reshape([N, xC, -1, xH, xW])
                sim = self.sigmoid((kernel * unfold_hpre).sum(2))
                similarity.append(sim)
            similarity = torch.stack(similarity, dim=2)  # torch.Size([1, 128, 8, 32, 32])
            h_pre[2] = similarity * h_pre[2]

            x = self.outc(torch.cat((x0 + x1, h_pre[2]), dim=1))  # torch.Size([1, 1, 8, 128, 128])

        x = v + x

        v = x.contiguous().view(x_d0, x_d1, x_d2, x_d3, x_d4)  # torch.Size([1, 8, 1, 128, 128])

        return v, y1, h_curr

class Args:
  pass

class sci3d(ModelBase, pl.LightningModule):
    def __init__(self):
        super().__init__()

        args = Args()
        args.layer_num  = 10
        args.n_channels = 1

        self.Phi_scale = nn.Parameter(torch.Tensor([0.01])).float()
        self.args = args
        onelayer = []
        self.LayerNo = args.layer_num

        for i in range(self.LayerNo):
            onelayer.append(BasicBlock(args, i))

        self.fcs = nn.ModuleList(onelayer)

        self.metrics = {}
        self.metrics['mse']  = nn.MSELoss()
        self.metrics['mae']  = MAELoss()
        self.metrics['psnr'] = lambda x, y: 10 * torch.log10(255. / self.metrics['mse'](x, y))

        # put your loss functions here
        self.loss_fn = MAELoss()

        self.lr = 1e-4

        datapath = '/Storage1/Matias/Datasetx30/'

        paths_mat = glob(join(datapath, '*', '*.mat'))

        data = loadmat(paths_mat[0])
        if 'mask' in data and 'orig' in data:
          self.mask = torch.from_numpy(data['mask']).half().unsqueeze(0).unsqueeze(0).permute(4, 0, 1, 2, 3)


    def forward(self, y):
        if self.mask.device.type == 'cpu':
          self.mask = self.mask.to(y.device)

        # batch_size, nframes, ch, w, h = y.shape
        Phisum = torch.sum(self.mask * self.mask, 0, keepdim=True)
        # Phisum[Phisum == 0] = 1

        y = y.permute(2,0,1,3,4)

        y1 = torch.zeros_like(y)
        hidden_state = []
        v = At_operator(y, self.mask)  # torch.Size([16, 1, 3, 64, 64])
        # h = [[None] * 4] * 8 * batch_size
        for i in range(self.LayerNo):
            v, y1, hidden_state = self.fcs[i](v, y, y1, self.mask, Phisum, hidden_state)

        y_pred = v

        y_pred = y_pred.permute(1, 2, 0, 3, 4)

        return y_pred

if __name__ == '__main__':
  model = sci3d()
  x = torch.randn(1, 1, 16, 256, 256)
  y = model(x)
  print(y.shape)
