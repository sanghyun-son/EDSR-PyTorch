from model import common

import torch.nn as nn

def make_model(args, parent=False):
    return SRCNN(args)

class SRCNN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(SRCNN, self).__init__()

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)
        batch_norm = args.batch_norm
        """
        url_name = 'r{}f{}x{}'.format(n_resblocks, n_feats, scale)
        if url_name in url:
            self.url = url[url_name]
        else:
            self.url = None
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)
        """

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)] # should I have an activation function here too?

        # define body module
        m_body = [
            common.BasicBlock(
                conv, n_feats, n_feats, kernel_size, act=act, bn = batch_norm
            ) for _ in range(n_resblocks-2)
        ]

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        return x