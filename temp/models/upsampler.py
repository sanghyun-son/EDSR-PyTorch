import torch
from torch import nn

class UpSampler(nn.Module):
    def __init__(
            self,
            scale: int = 2,
            n_feats: int = 64,
            n_colors: int = 3,
            ):

        super().__init__()

        # TODO 1: Create a convolution that produces
        # n_colors * scale**2 output channels.
        self.conv = nn.Conv2d(in_channels=n_feats, out_channels=n_colors*scale**2, kernel_size=3, stride=1, padding=1)
        # TODO 2: Create PixelShuffle with the given scale.
        self.pixel_shuffle = nn.PixelShuffle(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor :
        res = self.conv(x)
        res = self.pixel_shuffle(res)
        return res
