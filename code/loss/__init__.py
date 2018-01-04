from importlib import import_module

import torch
import torch.nn as nn
import torch.nn.functional as F

class loss:
    def __init__(self, args):
        self.args = args

    def get_loss(self):
        print('Preparing loss function...')

        my_loss = []
        losslist = self.args.loss.split('+')
        for loss in losslist:
            weight, loss_type = loss.split('*')
            if loss_type == 'MSE':
                loss_function = nn.MSELoss()
            elif loss_type == 'L1':
                loss_function = nn.L1Loss()
           
            my_loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function})

        if len(losslist) > 1:
            my_loss.append({
                'type': 'Total',
                'weight': 0,
                'function': None})

        print(my_loss)

        return my_loss
