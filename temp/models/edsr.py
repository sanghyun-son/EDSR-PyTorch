import torch
from torch import nn

from .blocks import ResidualBlock
from .upsampler import UpSampler

class EDSR(nn.Module):

    def __init__(
            self,
            scale: int = 2,
            n_resblocks: int = 16,
            n_feats: int = 64,
            n_colors: int = 3,
            res_scale: float = 0.1,
            ):
        super().__init__()

        # TODO 1: Initial convolution
        # Input: n_colors
        # Output: n_feats
        self.head = nn.Conv2d(in_channels=n_colors, out_channels=n_feats, kernel_size=3, stride=1, padding=1)
        # TODO 2: Create n_resblocks ResidualBlock modules
        # Hint: nn.Sequential and a generator expression may help.
        self.body = nn.Sequential(*[ResidualBlock(n_feats=n_feats, res_scale=res_scale) for _ in range(n_resblocks)])


        # TODO 3: Convolution after the residual blocks
        # Input and output channels: n_feats
        self.body_conv = nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=3, stride=1, padding=1)

        # TODO 4: Create your upsampler
        self.upsampler = UpSampler(scale=scale, n_feats=n_feats, n_colors=n_colors)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO 5: Compute the initial features
        head = self.head(x)
        # TODO 6: Pass them through the residual body
        res = self.body(head)
        # TODO 7: Apply body_conv and add the initial features
        res = self.body_conv(res) + head 
        # TODO 8: Upsample and return
        res = self.upsampler(res)
        return res
