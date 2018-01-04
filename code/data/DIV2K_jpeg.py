import os
import os.path
import random
import math
import errno

from data import common
from data import DIV2K

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data
from torchvision import transforms

class DIV2K_jpeg(DIV2K.DIV2K):
    def __init__(self, args, train=True):
        self._init_basic(args, train)

        split = 'train'
        dir_HR = 'DIV2K_{}_HR'.format(split)
        dir_LR = [
            'DIV2K_{}_LR_bicubic{}'.format(split, q) for q in args.quality]
        x_scale = ['X{}'.format(s) for s in args.scale]

        if self.args.ext != 'pack':
            self.dir_in = [[
                os.path.join(self.apath, dq, xs) \
                for dq in dir_LR] \
                for xs in x_scale]
            self.dir_tar = os.path.join(self.apath, dir_HR)
        else:
            print('Preparing binary packages...')
            packname = 'pack.pt' if self.train else 'packv.pt'
            name_tar = os.path.join(self.apath, dir_HR, packname)
            print('\tLoading ' + name_tar)
            self.pack_in = []
            self.pack_tar = torch.load(name_tar)
            if self.train:
                self._save_partition(
                    self.pack_tar,
                    os.path.join(self.apath, dir_HR, 'packv.pt'))

            for i, xs in enumerate(x_scale):
                pack_dq = []
                for j, dq in enumerate(dir_LR):
                    name_in = os.path.join(self.apath, dq, xs, packname)
                    print('\tLoading ' + name_in)
                    pack_dq.append(torch.load(name_in))
                    if self.train:
                        self._save_partition(
                            pack_dq[-1],
                            os.path.join(self.apath, dq, xs, 'packv.pt'))
                self.pack_in.append(pack_dq)

    def _load_file(self, idx):
        quality = random.randrange(
            len(self.args.quality)) if self.train else -1

        def _get_filename():
            if self.args.ext == 'png':
                if quality == len(self.args.quality) - 1 or quality == -1:
                    ext = '.png'
                else:
                    ext = '.jpeg'
            elif self.args.ext == 'pt':
                ext = '.pt'

            filename = '{:0>4}'.format(idx)
            name_in = '{}/{}x{}{}'.format(
                self.dir_in[self.idx_scale][quality],
                filename,
                self.scale[self.idx_scale],
                ext)
            name_tar = os.path.join(self.dir_tar, filename + '.png')

            return name_in, name_tar

        if self.args.ext == 'png':
            name_in, name_tar = _get_filename()
            img_in = misc.imread(name_in)
            img_tar = misc.imread(name_tar)
        elif self.args.ext == 'pt':
            name_in, name_tar = _get_filename()
            img_in = torch.load(name_in).numpy()
            img_tar = torch.load(name_tar).numpy()
        elif self.args.ext == 'pack':
            img_in = self.pack_in[self.idx_scale][quality][idx].numpy()
            img_tar = self.pack_tar[idx].numpy()

        return img_in, img_tar

