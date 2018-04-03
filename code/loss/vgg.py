from model import common

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable

class vgg(nn.Module):
    def __init__(self, conv_index, rgb_range):
        super(vgg, self).__init__()
        vgg_features = models.vgg19(pretrained=True).features
        modules = [m for m in vgg_features]
        if conv_index == '22':
            self.vgg = nn.Sequential(*modules[:8])
        elif conv_index == '54':
            self.vgg = nn.Sequential(*modules[:35])

        self.vgg.requires_grad = False

    def forward(self, input, target):
        def _forward(x):
            x = 
        nInput = self.normalizer(input)
        nTarget = self.normalizer(target)

        inputFeat = self.vgg54(nInput)
        targetFeat = self.vgg54(nTarget)
        targetFeat = targetFeat.detach()
        targetFeat.requires_grad = False
        loss = self.mse(inputFeat, targetFeat)

        return loss
