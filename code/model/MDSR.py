from collections import OrderedDict
from model import common

import torch.nn as nn

class MDSR(nn.Module):
    def __init__(self, args):
        super(MDSR, self).__init__()
        nResBlock = args.nResBlock
        nFeat = args.nFeat
        self.args = args

        subMul, addMul = -1 * args.subMean, 1 * args.subMean

        # Submean layer
        self.subMean = common.meanShift(
            args.rgbRange,
            (0.4488, 0.4371, 0.4040),
            subMul)

        # Head convolution for feature extracting
        self.headConv = common.conv3x3(args.nChannel, nFeat)

        # Scale-dependent pre-processing module
        self.preProcess = nn.ModuleList([
            nn.Sequential(
                common.ResBlock(nFeat, kernel_size=5),
                common.ResBlock(nFeat, kernel_size=5)) for _ in args.scale])

        # Main branch
        modules = [common.ResBlock(nFeat) for _ in range(nResBlock)]
        modules.append(common.conv3x3(nFeat, nFeat))
        self.body = nn.Sequential(*modules)

        # Scale-dependent upsampler
        self.upsample = nn.ModuleList([
            common.upsampler(s, nFeat) for s in args.scale])

        # Tail convolution for reconstruction
        self.tailConv = common.conv3x3(nFeat, args.nChannel)

        # Addmean layer
        self.addMean = common.meanShift(
            args.rgbRange,
            (0.4488, 0.4371, 0.4040),
            addMul)

        self.scaleIdx = 0

    def forward(self, x):
        x = self.subMean(x)
        x = self.headConv(x)
        x = self.preProcess[self.scaleIdx](x)
        res = self.body(x)
        res += x
        us = self.upsample[self.scaleIdx](res)
        output = self.tailConv(us)
        output = self.addMean(output)

        return output

    def setScale(self, scaleIdx):
        self.scaleIdx = scaleIdx
