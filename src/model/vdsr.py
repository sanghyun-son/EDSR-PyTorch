from model import common

import torch.nn as nn

url = {
    'r20f64': ''
}

def make_model(args, parent=False):
    return VDSR(args)

class VDSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(VDSR, self).__init__()

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3 
        self.url = url['r{}f{}'.format(n_resblocks, n_feats)]
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # define body module
        m_body = []
        m_body.append(common.BasicBlock(
            conv, args.n_colors, n_feats, kernel_size, bn=False
        ))
        for _ in range(n_resblocks - 2):
            m_body.append(common.BasicBlock(
                conv, n_feats, n_feats, kernel_size, bn=False
            ))
        m_body.append(common.BasicBlock(
            conv, n_feats, args.n_colors, kernel_size, bn=False, act=None
        ))
        self.body = nn.Sequential(*m_body)

    def forward(self, x):
        x = self.sub_mean(x)
        res = self.body(x)
        res += x
        x = self.add_mean(res)

        return x 

