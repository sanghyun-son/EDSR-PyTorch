import random

import numpy as np
import skimage.io as sio
import skimage.color as sc
import skimage.transform as st

import torch
from torchvision import transforms

def get_patch(img_in, img_tar, patch_size, scale, multi_scale=False):
    ih, iw, c = img_in.shape
    th, tw = scale * ih, scale * iw

    p = scale if multi_scale else 1
    tp = p * patch_size
    ip = tp // scale

    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)
    tx, ty = scale * ix, scale * iy

    img_in = img_in[iy:iy + ip, ix:ix + ip, :]
    img_tar = img_tar[ty:ty + tp, tx:tx + tp, :]

    return img_in, img_tar

def set_channel(img_in, img_tar, n_channel):
    if img_tar.ndim == 2:
        img_in = np.expand_dims(img_in, axis=2)
        img_tar = np.expand_dims(img_tar, axis=2)

    h, w, c = img_tar.shape

    def _set_channel(img):
        if n_channel == 1 and c == 3:
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channel == 3 and c == 1:
            img = np.concatenate([img] * n_channel, 2)

        return img

    return _set_channel(img_in), _set_channel(img_tar)

def np2Tensor(img_in, img_tar, rgb_range):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        torch_tensor = torch.from_numpy(np_transpose).float()
        torch_tensor.mul_(rgb_range / 255)

        return torch_tensor

    return _np2Tensor(img_in), _np2Tensor(img_tar)

def augment(img_in, img_tar, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)
        
        return img

    return _augment(img_in), _augment(img_tar)

