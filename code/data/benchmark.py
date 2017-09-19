from __future__ import print_function

import os
import os.path
import random
import math
import errno

from data import common

import numpy as np
import skimage
import skimage.io as sio
import skimage.color as sc

import torch
import torch.utils.data as data
from torchvision import transforms

class benchmark(data.Dataset):
    def __init__(self, args, setName):
        self.args = args
        self.name = setName
        self.train = False
        self.scale = args.scale
        self.scaleIdx = 0

        apath = args.dataDir + '/benchmark'
        self.ext = '.png'
        dirHR = 'benchmark_test_HR/' + setName
        dirLR = 'benchmark_test_LR/' + setName

        xScale = ['X{}'.format(s) for s in args.scale]
        self.dirIn = [os.path.join(apath, dirLR, xs) for xs in xScale]
        self.dirTar = os.path.join(apath, dirHR)
        self.fileList = []
        for f in os.listdir(self.dirTar):
            if f.endswith(self.ext):
                fileName, fileExt = os.path.splitext(f)
                self.fileList.append(fileName)
    def __getitem__(self, idx):
        scale = self.scale[self.scaleIdx]

        (nameIn, nameTar) = self.getFileName(idx, scale)
        imgIn = sio.imread(nameIn)
        imgTar = sio.imread(nameTar)
        if len(imgIn.shape) == 2:
            imgIn = np.expand_dims(imgIn, 2)
            imgTar = np.expand_dims(imgTar, 2)
        ih, iw, c = imgIn.shape
        imgTar = imgTar[0:ih * scale, 0:iw * scale, :]

        imgIn, imgTar = common.setChannel(imgIn, imgTar, self.args.nChannel)

        return common.np2Tensor(imgIn, imgTar, self.args.rgbRange)

    def __len__(self):
        return len(self.fileList)

    def getFileName(self, idx, scale):
        fileName = self.fileList[idx]
        nameIn = '{}x{}{}'.format(
            fileName, self.scale[self.scaleIdx], self.ext)
        nameIn = os.path.join(self.dirIn[self.scaleIdx], nameIn)
        nameTar = fileName + self.ext
        nameTar = os.path.join(self.dirTar, nameTar)

        return nameIn, nameTar

