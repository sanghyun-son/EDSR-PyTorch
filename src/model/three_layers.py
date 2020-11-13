from model import common
import torch.nn as nn

def make_model(args, parent=False):
    return three_layers(args)


class three_layers(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(three_layers, self).__init__()

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)

        self.url = None

        def basic_block(in_channels, out_channels, act):
            return common.BasicBlock(
                conv, in_channels, out_channels, kernel_size,
                bias=False, bn=False, act=act
            )

        #define body module
        m_body = []
        m_body.append(basic_block(args.n_colors, n_feats, act))
        for _ in range(n_resblocks - 2):
            m_body.append(basic_block(n_feats, n_feats, act))
        m_body.append((n_feats, args.n_colors, None))

        self.body = nn.Sequential(*m_body)

    def forward(self, x):
        x = self.body(x)

        return x











