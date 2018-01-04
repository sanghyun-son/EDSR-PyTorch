import random

import numpy as np
import skimage.io as sio
import skimage.color as sc
import skimage.transform as st

import torch
from torchvision import transforms

def get_patch(img_in, img_tar, args, scale, ix=-1, iy=-1):
    (ih, iw, c) = img_in.shape
    (th, tw) = (scale * ih, scale * iw)

    patch_mult = scale if len(args.scale) > 1 else 1
    tp = patch_mult * args.patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)
    img_in = img_in[iy:iy + ip, ix:ix + ip, :]
    img_tar = img_tar[ty:ty + tp, tx:tx + tp, :]
    info_patch = {
        'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    return img_in, img_tar, info_patch

def set_channel(img_in, img_tar, n_channel):
    (h, w, c) = img_tar.shape
    if n_channel == 1 and c == 3:
        img_in = np.expand_dims(sc.rgb2ycbcr(img_in)[:, :, 0], 2)
        img_tar = np.expand_dims(sc.rgb2ycbcr(img_tar)[:, :, 0], 2)
    elif n_channel == 3 and c == 1:
        img_in = np.concatenate([img_in] * n_channel, 2)
        img_tar = np.concatenate([img_tar] * n_channel, 2)

    return img_in, img_tar

def np2Tensor(img_in, img_tar, rgb_range):
    ts = (2, 0, 1)
    img_mul = rgb_range / 255
    img_in = torch.Tensor(img_in.transpose(ts).astype(float)).mul_(img_mul)
    img_tar = torch.Tensor(img_tar.transpose(ts).astype(float)).mul_(img_mul)

    return img_in, img_tar

def augment(img_in, img_tar, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}

    if random.random() < 0.5 and flip_h:
        img_in = img_in[:, ::-1, :]
        img_tar = img_tar[:, ::-1, :]
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            img_in = img_in[::-1, :, :]
            img_tar = img_tar[::-1, :, :]
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            img_in = img_in.transpose(1, 0, 2)
            img_tar = img_tar.transpose(1, 0, 2)
            info_aug['trans'] = True

    return img_in, img_tar, info_aug
