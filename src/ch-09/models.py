import torch

from utils import get_device
from torch import nn

device = get_device()

class LinearRegression(nn.Module):
    def __init__(self, num_features:int, out_features:int) -> None:
        super().__init__()
        self.lin = nn.Linear(
            in_features = num_features,
            out_features = 1,
            bias = True,
            device = device
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        output = self.lin(X)
        return output.squeeze(-1)

class RNN(nn.Module):
    def __init__(self, num_inputs: int, num_hiddens: int, sigma: float = 0.01):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.sigma = sigma

        self.W_xh = nn.Parameter(
            torch.randn(self.num_inputs, self.num_hiddens) * sigma
        )
        self.W_hh = nn.Parameter(
            torch.randn(self.num_hiddens, self.num_hiddens) * sigma
        )
        self.b_h = nn.Parameter(
            torch.zeros(self.num_hiddens)
        )

    def forward(self, inputs: torch.Tensor, state = None) -> tuple:
        """Perform a RNN forward pass for given input, updating H_t at each step.
        
        Args:
            - inputs: tensor of [num_steps, batch_size, num_inputs], where
                - num_steps: No of steps we are processing (100 steps means 100 states update)
                - batch_size: No of sequence we are processing (5 means processing 5 sequences,
                  each with num_steps timesteps)
                - num_inputs: Vector dimension for each token.
            - state: Last state for our hidden state. Initialize to torch.zeros() if N/A
        
        Returns:
            - List of updated hidden states after each timestep
            - Final hidden state after performing updates for all time steps

        """
        if (state == None):
            state = torch.zeros(inputs.shape[1], self.num_hiddens, device=inputs.device)

        outputs = []
        for X in inputs:
            state = torch.tanh(X @ self.W_xh + state @ self.W_hh + self.b_h)
            outputs.append(state)
        return outputs, state

