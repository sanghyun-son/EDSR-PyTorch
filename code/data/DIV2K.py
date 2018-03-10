import os

from data import common
from data import SRData

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data

class DIV2K(SRData.SRData):
    def __init__(self, args, train=True):
        super(DIV2K, self).__init__(args, train)

    def _set_filesystem(self, dir_data):
        self.apath = dir_data + '/DIV2K'
        self.dir_hr = os.path.join(self.apath, 'DIV2K_train_HR')
        self.dir_lr = os.path.join(self.apath, 'DIV2K_train_LR_bicubic')
        self.ext = '.png'

    def _make_filename(self, idx):
        return '{:0>4}'.format(idx)

    def _name_hrfile(self, filename):
        return os.path.join(self.dir_hr, filename + self.ext)

    def _name_hrbin(self):
        return os.path.join(self.apath, '{}_bin_HR.npy'.format(self.split))

    def _name_lrfile(self, filename, scale):
        return os.path.join(
            self.dir_lr,
            'X{}/{}x{}{}'.format(scale, filename, scale, self.ext))

    def _name_lrbin(self, scale):
        return os.path.join(
            self.apath,
            '{}_bin_LR_X{}.npy'.format(self.split, scale))

    def __len__(self):
        if self.train:
            return len(self.images_hr) * self.repeat
        else:
            return len(self.images_hr)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr)
        else:
            return idx

