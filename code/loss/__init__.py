from importlib import import_module

import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss(nn.modules.loss._Loss):
    def __init__(self, args):
        super(Loss, self).__init__()
        print('Preparing loss function...')

        self.loss = []
        self.losslist = nn.ModuleList()
        for loss in args.loss.split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'MSE':
                loss_function = nn.MSELoss()
            elif loss_type == 'L1':
                loss_function = nn.L1Loss()
            elif loss_type.find('VGG') >= 0:
                module = import_module('loss.vgg')
                loss_function = getattr(module, 'VGG')(
                    loss_type[3:],
                    args.rgb_range
                )
           
            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function}
            )

        if len(self.loss) > 1:
            self.loss.append({
                'type': 'Total',
                'weight': 0,
                'function': None}
            )

        print('Loss:')
        for l in self.loss:
            print('{:.3f} * {}'.format(l['weight'], l['type']))
            if l['function'] is not None:
                self.losslist.append(l['function'])

        self.log = torch.Tensor(len(self.loss))

    def __len__(self):
        return len(self.loss)

    def forward(self, sr, hr):
        losses = []
        self.log.fill_(0)
        for i, l in enumerate(self.loss):
            if l['function'] is not None:
                if isinstance(sr, list):
                    if isinstance(hr, list):
                        loss = l['function'](sr[i], hr[i])
                    else:
                        loss = l['function'](sr[i], hr)
                else:
                    loss = l['function'](sr, hr)

                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                self.log[i] = effective_loss.data[0]

        loss_sum = sum(losses)
        self.log[-1] = loss_sum.data[0]

        return loss_sum

    def get_types(self):
        return [l['type'] for l in self.loss]
