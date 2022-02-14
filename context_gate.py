'''
calculate context gate
'''
import imp
import torch
from torch import nn
from zmq import device

class context_gate(nn.Module):
    def __init__(self, embed_size: int, hidden_size: int) -> None:
        '''
        params:
            @embed_size{int}
            @hidden_size{int}
        '''
        super(context_gate, self).__init__() 
        self.Ws = nn.parameter.Parameter(torch.empty(hidden_size, hidden_size))
        self.Wy = nn.parameter.Parameter(torch.empty(embed_size, hidden_size))
        self.Wc = nn.parameter.Parameter(torch.empty(hidden_size, hidden_size))

        self.initialize_weights()
    
    def initialize_weights(self):
        
        torch.nn.init.xavier_uniform_(self.Ws)
        torch.nn.init.xavier_uniform_(self.Wy)
        torch.nn.init.xavier_uniform_(self.Wc)

    def forward(self, St, Yt, Ct):
        '''
        params:
            @St{torch.tensor}: (batch, len) state in t-1 step;
            @Yt{torch.tensor}: (batch, len) y in in t-1 step;
            @Ct{torch.tensor}: (batch, len) context vector in in t step.
        '''
        
        context_gate = St @ self.Ws + torch.exp(Yt) @ self.Wy + Ct @ self.Wc

        return torch.sigmoid(context_gate)