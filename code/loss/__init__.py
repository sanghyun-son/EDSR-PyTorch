from importlib import import_module

import torch
import torch.nn as nn
import torch.nn.functional as F

class loss:
    def __init__(self, args):
        self.args = args

    def getLoss(self):
        myLoss = []
        lossList = self.args.loss.split('+')
        for loss in lossList:
            weight, lossType = loss.split('*')
            if lossType == 'MSE':
                lossFunction = nn.MSELoss()
            elif lossType == 'L1':
                lossFunction = nn.L1Loss()
            elif lossType == 'VGG':
                vggModule = import_module('loss.vgg')
                lossFunction = getattr(vggModule, 'vgg')(self.args, lossType)
            
            if self.args.precision != 'Single':
                if self.args.precision == 'Double':
                    lossFunction = lossFunction.double()
                elif self.args.precision == 'Half':
                    lossFunction = lossFunction.half()

            myLoss.append({
                'type': lossType,
                'weight': float(weight),
                'function': lossFunction})
            if self.args.cuda:
                myLoss[-1]['function'].cuda()
        if len(lossList) > 1:
            myLoss.append({
                'type': 'Total',
                'weight': 0,
                'function': None})

        print('Prepare loss function...')
        print(myLoss)

        return myLoss
