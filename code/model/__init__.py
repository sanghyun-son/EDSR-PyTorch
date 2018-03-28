from importlib import import_module

import torch
import torch.nn as nn
import torch.nn.functional as F

class model:
    def __init__(self, args):
        self.module = import_module('model.' + args.model)
        self.args = args

    def get_model(self):
        print('Making model...')
        my_model = self.module.make_model(self.args)
        if self.args.pre_train != '.':
            print('Loading model from {}...'.format(self.args.pre_train))
            my_model.load_state_dict(torch.load(self.args.pre_train))

        if not self.args.no_cuda:
            print('\tCUDA is ready!')
            torch.cuda.manual_seed(self.args.seed)
            my_model.cuda()

            if self.args.precision == 'half':
                my_model.half()

            if self.args.n_GPUs > 1:
                my_model = nn.DataParallel(my_model, range(0, self.args.n_GPUs))

        if self.args.print_model:
            print(my_model)

        return my_model

