from model import common
from model import EDSR

import torch.nn as nn

class EDSR_scale(EDSR.EDSR):
    def __init__(self, args):
        super(EDSR_scale, self).__init__(args)
        nResBlock = args.nResBlock
        nFeat = args.nFeat

        # Main branch
        modules = [
            common.ResBlock_scale(nFeat, scale=0.1) for _ in range(nResBlock)]
        modules.append(common.conv3x3(nFeat, nFeat))
        self.body = nn.Sequential(*modules)

