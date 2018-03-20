import os

from data import common
from data import srdata
from data import imgfolder

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data

class Urban100(imgfolder.ImgFolder):
    def __init__(self, args, train=True):
        super(Urban100, self).__init__(args, train)

    def _set_filesystem(self, dir_data):
        self.apath = dir_data + '/benchmark/Urban100'
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_lr = os.path.join(self.apath, 'LR_bicubic')
        self.ext = '.png'

