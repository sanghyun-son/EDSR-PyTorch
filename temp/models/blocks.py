import torch
from torch import nn
class ResidualBlock(nn.Module):
    def __init__(
            self,
            n_feats: int = 64,
            res_scale: float = 0.1,
            ):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=3, stride=1, padding=1)
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor):
        Fx = self.conv1(x)
        Fx = self.relu(Fx)
        Fx = self.conv2(Fx)
        Fx *= self.res_scale
        res = Fx + x
        return res 

