from model import common

import torch.nn as nn
import torch.nn.init as init

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

        def basic_block(in_channels, out_channels, act):
            return common.BasicBlock(
                conv, in_channels, out_channels, kernel_size,
                bias=True, bn=False, act=act
            )

        # define body module
        m_body = []
        m_body.append(basic_block(args.n_colors, n_feats, nn.ReLU(True)))
        for _ in range(n_resblocks - 2):
            m_body.append(basic_block(n_feats, n_feats, nn.ReLU(True)))
        m_body.append(basic_block(n_feats, args.n_colors, None))

        self.body = nn.Sequential(*m_body)

    def forward(self, x):
        x = self.sub_mean(x)
        res = self.body(x)
        res += x
        x = self.add_mean(res)

        return x 

