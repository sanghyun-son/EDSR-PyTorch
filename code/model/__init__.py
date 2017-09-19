from importlib import import_module

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class model:
    def __init__(self, args):
        self.module = import_module('model.' + args.model)
        self.args = args

    def getModel(self):
        if self.args.preTrained == '.':
            print('Making model...')
            myModel = getattr(self.module, self.args.model)(self.args)
        else:
            print('Loading model from {}...'.format(self.args.preTrained))
            myModel = torch.load(self.args.preTrained)        

        if self.args.cuda:
            print('\tCUDA is ready!')
            torch.cuda.manual_seed(self.args.seed)
            myModel.cuda()
            if self.args.precision == 'double':
                myModel = myModel.double()
            elif self.args.precision == 'half':
                myModel = myModel.half()

            if self.args.nGPUs > 1:
                gpuList = range(0, self.args.nGPUs)
                myModel = nn.DataParallel(myModel, gpuList)

        if self.args.printModel:
            print(myModel)
            #torch.save(myModel, 'check_size.pt')
            #input()

        return myModel

