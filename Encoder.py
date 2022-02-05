from turtle import forward
from mdlstm import *
import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, height: int, width: int, input_channel: int, hidden_size: int):
        super(Encoder, self).__init__()
        self.model = MDLSTM(height, width, input_channel, hidden_size)

    def forward(self, x: torch.Tensor):
        return self.model(x)