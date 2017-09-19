from collections import OrderedDict

from model import common

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.autograd import Variable

class vgg(nn.Module):
    def __init__(self, args, GANType):
        super(vgg, self).__init__()
        self.args = args

        preTrained = models.vgg19(pretrained=True).features
        vggUnpack = [m for m in preTrained]
        self.vgg54 = nn.Sequential(*vggUnpack[:36])
        self.vgg54.requires_grad = False
        self.mse = nn.MSELoss()
        self.normalizer = common.vggNormalizer(args)

    def forward(self, input, target):
        nInput = self.normalizer(input)
        nTarget = self.normalizer(target)

        inputFeat = self.vgg54(nInput)
        targetFeat = self.vgg54(nTarget)
        targetFeat = targetFeat.detach()
        targetFeat.requires_grad = False
        loss = self.mse(inputFeat, targetFeat)

        return loss
