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

class DIV2K(data.Dataset):
    def __init__(self, args, train=True):
        self.args = args
        self.name = 'DIV2K'
        self.train = train
        self.scale = args.scale
        self.scaleIdx = 0

        self.repeat = args.testEvery // (args.nTrain // args.batchSize)

        if args.ext == 'png':
            apath = args.dataDir + '/DIV2K'
            self.ext = '.png'
        else:
            apath = args.dataDir + '/DIV2K_decoded'
            self.ext = '.pt'

        split = 'train'
        dirHR = 'DIV2K_{}_HR'.format(split)
        dirLR = 'DIV2K_{}_LR_bicubic'.format(split)
        xScale = ['X{}'.format(s) for s in args.scale]

        if self.args.ext != 'ptpack':
            self.dirIn = [os.path.join(apath, dirLR, xs) for xs in xScale]
            self.dirTar = os.path.join(apath, dirHR)
        else:
            print('Preparing binary packages...')
            packName = 'pack.pt' if self.train else 'packv.pt'
            nameTar = os.path.join(apath, dirHR, packName)
            print('\tLoading ' + nameTar)
            self.packIn = []
            self.packTar = torch.load(nameTar)
            if self.train:
                self.savePartition(self.packTar, os.path.join(apath, dirHR, 'packv.pt'))
            for i, xs in enumerate(xScale):
                nameIn = os.path.join(apath, dirLR, xs, packName)
                print('\tLoading ' + nameIn)
                self.packIn.append(torch.load(nameIn))
                if self.train:
                    self.savePartition(self.packIn[i],
                        os.path.join(apath, dirLR, xs, 'packv.pt'))

    def __getitem__(self, idx):
        scale = self.scale[self.scaleIdx]
        if self.train:
            idx = (idx % self.args.nTrain) + 1
        else:
            idx = (idx + self.args.valOffset) + 1

        if self.args.ext == 'png':
            nameIn, nameTar = self.getFileName(idx)
            imgIn = sio.imread(nameIn)
            imgTar = sio.imread(nameTar)
        elif self.args.ext == 'pt':
            nameIn, nameTar = self.getFileName(idx)
            imgIn = torch.load(nameIn).numpy()
            imgTar = torch.load(nameTar).numpy()
        elif self.args.ext == 'ptpack':
            imgIn = self.packIn[self.scaleIdx][idx].numpy()
            imgTar = self.packTar[idx].numpy()

        if self.train:
            imgIn, imgTar, pi = common.getPatch(imgIn, imgTar, self.args, scale)
            imgIn, imgTar, ai = common.augment(imgIn, imgTar)
        else:
            (ih, iw, c) = imgIn.shape
            imgTar = imgTar[0:ih * scale, 0:iw * scale, :]

        imgIn, imgTar = common.setChannel(imgIn, imgTar, self.args.nChannel)

        return common.np2Tensor(
            imgIn, imgTar, self.args.rgbRange, self.args.precision)

    def __len__(self):
        if self.train:
            return self.args.nTrain * self.repeat
        else:
            return self.args.nVal

    def setScale(self, scaleIdx):
        self.scaleIdx = scaleIdx

    def getFileName(self, idx):
        fileName = '{:0>4}{}'.format(idx)
        nameIn = '{}x{}{}'.format(
            fileName, self.scale[self.scaleIdx], self.ext)
        nameIn = os.path.join(self.dirIn[self.scaleIdx], nameIn)
        nameTar = fileName + self.ext
        nameTar = os.path.join(self.dirTar, nameTar)

        return nameIn, nameTar

    def savePartition(self, dict, name):
        valDict = {}
        for i in range(self.args.nTrain, self.args.nTrain + self.args.nVal):
            valDict[i + 1] = dict[i + 1]
        torch.save(valDict, name)
