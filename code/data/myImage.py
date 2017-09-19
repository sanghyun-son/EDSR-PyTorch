import os
import os.path
import random
import math
import errno

from data import common

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data
from torchvision import transforms

class myImage(data.Dataset):
    def __init__(self, args, train=True):
        self.args = args
        self.train = train
        self.name = 'myImage'
        self.scale = args.scale
        self.scaleIdx = 0
        apath = '../test'

        self.fileList = []
        if not train:
            for f in os.listdir(apath):
                try:
                    fileName = os.path.join(apath, f)
                    misc.imread(fileName)
                    self.fileList.append(fileName)
                except:
                    pass

    def __getitem__(self, idx):
        imgIn = misc.imread(self.fileList[idx])
        if len(imgIn.shape) == 2:
            imgIn = np.expand_dims(imgIn, 2)

        imgIn, imgTar = common.setChannel(imgIn, imgIn, self.args.nChannel)

        return common.np2Tensor(imgIn, imgTar, self.args.rgbRange)

    def __len__(self):
        return len(self.fileList)

    def setScale(self, scaleIdx):
        self.scaleIdx = scaleIdx

