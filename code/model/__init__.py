from importlib import import_module

import torch
import torch.nn as nn
import torch.nn.functional as F

class model(object):
    def __init__(self):
        pass

    def get_model(self, args):
        print('Making model...')
        module = import_module('model.' + args.model.lower())
        my_model = module.make_model(args)
        if args.pre_train != '.':
            print('Loading model from {}...'.format(args.pre_train))
            my_model.load_state_dict(torch.load(args.pre_train))

        if not args.no_cuda:
            print('\tCUDA is ready!')
            torch.cuda.manual_seed(args.seed)
            my_model.cuda()

            if args.precision == 'half':
                my_model.half()

            if args.n_GPUs > 1:
                my_model = nn.DataParallel(my_model, range(0, args.n_GPUs))

        if args.print_model:
            print(my_model)

        return my_model

