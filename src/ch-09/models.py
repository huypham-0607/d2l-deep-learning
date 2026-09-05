import torch

from utils import get_device
from torch import nn

device = get_device()

class LinearRegression(nn.Module):
    def __init__(self, num_features:int, out_features:int) -> None:
        self.lin = nn.Linear(
            in_features = num_features,
            out_features = 1,
            bias = True,
            device = device
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        output = self.lin(X)
        return output.squeeze(-1)
