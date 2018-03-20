import os

from data import common
from data import srdata

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data

class ImgFolder(srdata.SRData):
    def __init__(self, args, train=True):
        self.repeat = args.test_every // (args.n_train // args.batch_size)
        super(ImgFolder, self).__init__(args, train)

    def _scan(self):
        list_hr = []
        list_lr = [[] for _ in self.scale]
        for entry in os.scandir(self.dir_hr):
            filename = os.path.splitext(entry.name)[0]
            list_hr.append(os.path.join(self.dir_hr, filename + self.ext))
            for si, s in enumerate(self.scale):
                list_lr[si].append(os.path.join(
                    self.dir_lr,
                    'X{}/{}x{}{}'.format(s, filename, s, self.ext)
                ))

        return list_hr, list_lr

    def _set_filesystem(self, dir_data):
        raise NotImplementedError

    def __len__(self):
        return len(self.images_hr)

    def _get_index(self, idx):
        return idx

