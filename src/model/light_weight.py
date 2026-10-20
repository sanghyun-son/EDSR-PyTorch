import torch.nn as nn

from . import common


def make_model(args, parent=False):
    return LightweightEDSR(args)


class LightweightEDSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super().__init__()

        n_resblocks = getattr(args, "n_resblocks", 8)
        n_feats = getattr(args, "n_feats", 32)
        n_colors = getattr(args, "n_colors", 3)
        rgb_range = getattr(args, "rgb_range", 255)
        res_scale = getattr(args, "res_scale", 1.0)

        scale = args.scale[0] if isinstance(args.scale, (list, tuple)) else args.scale

        kernel_size = 3
        act = nn.ReLU(True)

        self.sub_mean = common.MeanShift(rgb_range)
        self.add_mean = common.MeanShift(rgb_range, sign=1)

        # Head: RGB input -> feature representation
        self.head = nn.Sequential(
            conv(n_colors, n_feats, kernel_size)
        )

        # Body: lightweight residual feature extraction
        body = [
            common.ResBlock(
                conv,
                n_feats,
                kernel_size,
                act=act,
                res_scale=res_scale,
            )
            for _ in range(n_resblocks)
        ]

        body.append(conv(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*body)

        # Tail: upsample features and reconstruct RGB image
        self.tail = nn.Sequential(
            common.Upsampler(
                conv,
                scale,
                n_feats,
                act=False,
            ),
            conv(n_feats, n_colors, kernel_size),
        )

    def forward(self, x):
        x = self.sub_mean(x)
        head = self.head(x)

        res = self.body(head)
        res = res + head

        x = self.tail(res)
        x = self.add_mean(x)

        return x
