import torch
from torch import nn

class L1Loss(nn.Module):
    def __init__( self):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(
            self,
            sr: torch.Tensor,
            hr: torch.Tensor,
            ) -> torch.Tensor:
        return self.loss(sr, hr)
