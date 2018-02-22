import os

from data import common

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data

class SRData(data.Dataset):
    def __init__(self, args, train=True):
        self.args = args
        self.train = train
        self.split = 'train' if train else 'test'
        self.scale = args.scale
        self.idx_scale = 0
        self.repeat = args.test_every // (args.n_train // args.batch_size)

        self._set_filesystem(args.dir_data)
        def _scan():
            list_hr = []
            list_lr = [[] * len(self.scale)]
            idx_begin = 0 if train else args.n_train
            idx_end = args.n_train if train else args.offset_val + args.n_val
            for i in range(idx_begin + 1, idx_end + 1):
                filename = self._make_filename(i)
                list_hr.append(self._name_hrfile(filename))
                for si, s in enumerate(self.scale):
                    list_lr[si].append(self._name_lrfile(filename, s))

            return list_hr, list_lr

        def _load():
            self.images_hr = np.load(self._name_hrbin())
            self.images_lr = [
                np.load(self._name_lrbin(s)) for s in self.scale]

        if args.ext == 'img':
            self.images_hr, self.images_lr = _scan()
        elif args.ext.find('bin') >= 0:
            try:
                if args.ext.find('reset') >= 0:
                    raise IOError
                print('Loading a binary file')
                _load()
            except:
                print('Preparing a binary file')
                list_hr, list_lr = _scan()
                hr = [misc.imread(f) for f in list_hr]
                np.save(self._name_hrbin(), hr)
                del hr
                for si, s in enumerate(self.scale):
                    lr_scale = [misc.imread(f) for f in list_lr[si]]
                    np.save(self._name_lrbin(s), lr_scale)
                    del lr_scale
                _load()
        else:
            print('Please define data type')

    def _set_filesystem(self, dir_data):
        raise NotImplementedError

    def _make_filename(self, idx):
        raise NotImplementedError

    def _name_hrfile(self, filename):
        raise NotImplementedError

    def _name_hrbin(self):
        raise NotImplementedError

    def _name_lrfile(self, filename, scale):
        raise NotImplementedError

    def _name_lrbin(self, scale):
        raise NotImplementedError

    def __getitem__(self, idx):
        img_lr, img_hr = self._load_file(idx)
        img_lr, img_hr = self._get_patch(img_lr, img_hr)
        img_lr, img_hr = common.set_channel(
            img_lr, img_hr, self.args.n_colors)

        return common.np2Tensor(img_lr, img_hr, self.args.rgb_range)

    def __len__(self):
        if self.train:
            return len(self.images_hr)
        else:
            return len(self.images_lr)

    def _get_index(self, idx):
        return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        img_lr = self.images_lr[self.idx_scale][idx]
        img_hr = self.images_hr[idx]
        if self.args.ext == 'img':
            img_lr = misc.imread(img_lr)
            img_hr = misc.imread(img_hr)

        return img_lr, img_hr

    def _get_patch(self, img_lr, img_hr):
        patch_size = self.args.patch_size
        scale = self.scale[self.idx_scale]
        multi_scale = len(self.scale) > 1
        if self.train:
            img_lr, img_hr = common.get_patch(
                img_lr, img_hr, patch_size, scale, multi_scale=multi_scale)
            img_lr, img_hr = common.augment(img_lr, img_hr)
        else:
            ih, iw, c = img_lr.shape
            img_hr = img_hr[0:ih * scale, 0:iw * scale, :]

        return img_lr, img_hr

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale

