import os

from data import common

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data

class Demo(data.Dataset):
    def __init__(self, args, train=False):
        self.args = args
        self.train = False
        self.name = 'MyImage'
        self.scale = args.scale
        self.idx_scale = 0
        apath = '../test'

        self.filelist = []
        for f in os.listdir(apath):
            if f.find('.png') >= 0 or f.find('.jp') >= 0:
                self.filelist.append(os.path.join(apath, f))

    def __getitem__(self, idx):
        filename = os.path.split(self.filelist[idx])[-1]
        filename, _ = os.path.splitext(filename)
        lr = misc.imread(self.filelist[idx])
        lr = common.set_channel([lr], self.args.n_colors)[0]

        return common.np2Tensor([lr], self.args.rgb_range)[0], filename

    def __len__(self):
        return len(self.filelist)

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale
