import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import math
# from torch.autograd import Variable
from collections import OrderedDict


class FishBlock(nn.Module):
    r"""Pre-act residual block, the middle transformations are bottle-necked
        :param in_planes: (int) number of input channels
        :param planes: (int) number of output channels
        :param stride: (int) stride value 
        :param mode: (string) DOWN | UP
        :param k: (int) times of additive
        Forward out resolution = ---- {floor[(input_size -1) / stride] + 1} "DOWN"
                                 ---- {}                                    "UP"         
    """

    expansion = 4
    def __init__(self, in_planes, planes, stride=2, mode='DOWN', k=1, dilation=1):
        super(FishBlock, self).__init__()
        self.mode = mode.upper()
        self.relu = nn.ReLU(inplace=True)
        self.k = k

        btnk_ch = planes // self.expansion

        self.bn1 = nn.BatchNorm2d(in_planes)
        # size unchanged
        self.conv1 = nn.Conv2d(in_planes, btnk_ch, kernel_size=1, bias=False)

        self.bn2 = nn.BatchNorm2d(btnk_ch)

        self.conv2 = nn.Conv2d(btnk_ch, btnk_ch, kernel_size=3, stride=stride, padding=dilation,
                               dilation=dilation, bias=False)

        self.bn3 = nn.BatchNorm2d(btnk_ch)
        self.conv3 = nn.Conv2d(btnk_ch, planes, kernel_size=1, bias=False)

        if mode == 'UP':
            self.downsample = None
        # downsample
        elif in_planes != planes or stride > 1:
            self.downsample = nn.Sequential(
                nn.BatchNorm2d(in_planes),
                self.relu,
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
            )
        else:
            self.downsample = None

    def _pre_act_forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.mode == 'UP':
            residual = self.squeeze_idt(x)
        elif self.downsample is not None:
            residual = self.downsample(residual)

        out += residual

        return out

    def squeeze_idt(self, idt):
        n, c, h, w = idt.size()
        return idt.view(n, c // self.k, self.k, h, w).sum(2)

    def forward(self, x):
        out = self._pre_act_forward(x)
        return out


class Fish(nn.Module):
    r"""Fish tail, body and head
    :params block (string) FishBlock by default
    :params num_cls (int) number of classes
    :params num_down_sample (int) number of DR units
    :params num_up_sample (int) number of UR units
    :params trans_map (tuple/list) index of the horizontal connection layers
    :params network_planes
    :params num_res_blks
    :params num_trans_blks

    """
    def __init__(self, block=FishBlock, num_cls=1000, num_down_sample=5, num_up_sample=3, trans_map=(2, 1, 0, 6, 5, 4),
                 network_planes=None, num_res_blks=None, num_trans_blks=None):
        super(Fish, self).__init__()

        self.block = block
        self.trans_map = trans_map

        # The paper keeps a sample rate of 2 by default
        self.upsample = nn.Upsample(scale_factor=2)
        self.down_sample = nn.MaxPool2d(2, stride=2)
        
        self.num_cls = num_cls
        self.num_down = num_down_sample
        self.num_up = num_up_sample
        self.res1 = self._make_residual_block(3, 64, 3)
        # self.network_planes = network_planes[1:]
        # self.depth = len(self.network_planes)
        # self.num_trans_blks = num_trans_blks
        # self.num_res_blks = num_res_blks
        # self.fish = self._make_fish(network_planes[0])

    def _make_residual_block(self, inplanes, outplanes, nstage, is_up=False, k=1, dilation=1):

        layers = nn.Sequential()

        if is_up:
            layers.add_module('restage0',self.block(inplanes, outplanes, mode='UP', dilation=dilation, k=k))
        else:
            layers.add_module('restage0', self.block(inplanes, outplanes, stride=1))

        for i in range(1, nstage):
            # does not alter the shape of the image
            layers.add_module('restage%d'%i, self.block(outplanes, outplanes, stride=1, dilation=dilation))

        return layers

    def __make_score(self, in_ch, out_ch=1000, has_pool=False):

        bn = nn.BatchNorm2d(in_ch)
        relu = nn.ReLU(inplace=True)
        conv_trans = nn.Conv2d(in_ch, in_ch // 2, kernel_size=1, bias=False)
        bn_out = nn.BatchNorm2d(in_ch // 2)
        conv = nn.Sequential(bn, relu, conv_trans, bn_out, relu)
        if has_pool:
            fc = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_ch // 2, out_ch, kernel_size=1, bias=True))
        else:
            fc = nn.Conv2d(in_ch // 2, out_ch, kernel_size=1, bias=True)
        return [conv, fc]


    def _make_stage(self, is_down_sample, inplanes, outplanes, n_blk, has_trans=True,
                    has_score=False, trans_planes=0, no_sampling=False, num_trans=2, **kwargs):

        sample_block = nn.ModuleList()
        if has_score:
            sample_block.add_module('score', self._make_score(outplanes, outplanes * 2, has_pool=False))

        if no_sampling or is_down_sample:
            res_block = self._make_residual_block(inplanes, outplanes, n_blk, **kwargs)
        else:
            res_block = self._make_residual_block(inplanes, outplanes, n_blk, is_up=True, **kwargs)

        sample_block.append(res_block)

        if has_trans:
            trans_in_planes = self.in_planes if trans_planes == 0 else trans_planes
            sample_block.add_module('trans', self._make_residual_block(trans_in_planes, trans_in_planes, num_trans))

        if not no_sampling and is_down_sample:
            sample_block.add_module('down', self.down_sample)
        elif not no_sampling:  # Up-Sample
            sample_block.add_module('up', self.upsample)

        return sample_block


if __name__ == "__main__":

    # test FishBlock

    test_x = torch.ones(32, 4, 224, 224)
    # print(test_x.shape)

    base = FishBlock(4, 64)
    # print(base)
    # test_y = base(test_x)
    # print(test_y.shape)

    fish = Fish()
    print(fish)

