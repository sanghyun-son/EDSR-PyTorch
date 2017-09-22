from model import common

import torch.nn as nn

class EDSR(nn.Module):
    def __init__(self, args):
        super(EDSR, self).__init__()
        nResBlock = args.nResBlock
        nFeat = args.nFeat
        scale = args.scale[0]

        self.args = args

        # Submean layer
        self.subMean = common.meanShift(
            args.rgbRange,
            (0.4488, 0.4371, 0.4040), -1 * args.subMean)

        # Head convolution for feature extracting
        self.headConv = common.conv3x3(args.nChannel, nFeat)

        # Main branch
        modules = [common.ResBlock(nFeat) for _ in range(nResBlock)]
        modules.append(common.conv3x3(nFeat, nFeat))
        self.body = nn.Sequential(*modules)

        # Upsampler
        self.upsample = common.upsampler(scale, nFeat)

        # Tail convolution for reconstruction
        self.tailConv = common.conv3x3(nFeat, args.nChannel)

        # Addmean layer
        self.addMean = common.meanShift(
            args.rgbRange,
            (0.4488, 0.4371, 0.4040), 1 * args.subMean)

    def forward(self, x):
        x = self.subMean(x)
        x = self.headConv(x)

        res = self.body(x)
        res += x

        us = self.upsample(res)
        output = self.tailConv(us)
        output = self.addMean(output)

        return output

