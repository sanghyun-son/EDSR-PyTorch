import random

import numpy as np
import skimage.io as sio
import skimage.color as sc
import skimage.transform as st

import torch
from torchvision import transforms

def getPatch(imgIn, imgTar, args, scale):
    (ih, iw, c) = imgIn.shape
    (th, tw) = (scale * ih, scale * iw)

    patchMult = scale if len(args.scale) > 1 else 1
    tp = patchMult * args.patchSize
    ip = tp // scale

    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)
    (tx, ty) = (scale * ix, scale * iy)
    imgIn = imgIn[iy:iy + ip, ix:ix + ip, :]
    imgTar = imgTar[ty:ty + tp, tx:tx + tp, :]
    patchInfo = {
        'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    return imgIn, imgTar, patchInfo

def setChannel(imgIn, imgTar, nChannel):
    (h, w, c) = imgTar.shape
    if nChannel == 1 and c == 3:
        imgIn = np.expand_dims(sc.rgb2ycbcr(imgIn)[:, :, 0], 2)
        imgTar = np.expand_dims(sc.rgb2ycbcr(imgTar)[:, :, 0], 2)
    elif nChannel == 3 and c == 1:
        imgIn = np.concatenate([imgIn] * nChannel, 2)
        imgTar = np.concatenate([imgTar] * nChannel, 2)

    return imgIn, imgTar

def np2Tensor(imgIn, imgTar, rgbRange, precision='single'):
    ts = (2, 0, 1)
    mulImg = rgbRange / 255
    imgIn = torch.Tensor(imgIn.transpose(ts).astype(float)).mul_(mulImg)
    imgTar = torch.Tensor(imgTar.transpose(ts).astype(float)).mul_(mulImg)

    return imgIn, imgTar

def augment(imgIn, imgTar, flip=True, rotation=True):
    augInfo = {'hFlip': False, 'vFlip': False, 'trans': False}

    if random.random() < 0.5 and flip:
        imgIn = imgIn[:, ::-1, :]
        imgTar = imgTar[:, ::-1, :]
        augInfo['hFlip'] = True

    if rotation:
        if random.random() < 0.5:
            imgIn = imgIn[::-1, :, :]
            imgTar = imgTar[::-1, :, :]
            augInfo['vFlip'] = True
        if random.random() < 0.5:
            imgIn = imgIn.transpose(1, 0, 2)
            imgTar = imgTar.transpose(1, 0, 2)
            augInfo['trans'] = True

    return imgIn, imgTar, augInfo
