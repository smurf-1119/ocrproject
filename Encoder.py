'''
encode the feature map
'''

from turtle import forward
from black import out
from mdlstm import *
import torch
from torch import nn

class Block(nn.Module):
    '''
    description: the block of encoder composed by mdlstm, batchnormalize and non_linear.
    '''
    def __init__(self, height: int, width: int, input_channel: int, hidden_size: int, non_linear):
        '''
        params:
            @height{int}: the height of input image;
            @width{int}: the width of input image;
            @input_channel{int}: the channels of input image;
            @hidden_size{int}: the channels of output image;
            @non_linear{func}: the non-linear function.
        '''
        super(Block, self).__init__()
        self.mdlstm = MDLSTM(height, width, input_channel, hidden_size)
        self.bn = nn.BatchNorm2d(hidden_size)
        self.non_linear = non_linear
    
    def forward(self, x: torch.Tensor):
        '''
        params:
            @x{tensor}: (batch, channels, width, height) feature map embeded by resnet50.
        return: 
            output{tensor}: (batch, hidden_size, width, height);
            state{tensor}: (batch, hidden_size, width, height);
            cell{tensor}: (batch, hidden_size, width, height).
        '''

        state, cell = self.mdlstm(x)
        output = self.non_linear(self.bn(state))
        return output, (state, cell)
        

class Encoder(nn.Module):
    def __init__(self, height: int, width: int, input_channel: int, hidden_size: int, num_layers: int, non_linear):
        '''
        params:
            @height{int}: the height of input image;
            @width{int}: the width of input image;
            @input_channel{int}: the channels of input image;
            @hidden_size{int}: the channels of output image;
            @num_layers{int}: the number of layers;
            @non_linear{func}: the non-linear function.
        '''
        super(Encoder, self).__init__()
        self.model = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.model.append(Block(height, width, input_channel, hidden_size, non_linear))
            else:
                self.model.append(Block(height, width, hidden_size, hidden_size, non_linear))

    def forward(self, x: torch.Tensor):
        '''
        params:
            @x{tensor}: (batch, channels, width, height) feature map embeded by resnet50.
        return: 
            output{tensor}: (batch, hidden_size, width, height);
            state_list{tensor}: (num_layers, batch, hidden_size, width, height);
            cell_list{tensor}: (num_layers, batch, hidden_size, width, height).
        '''

        state_list = []
        cell_list = []
        output = x

        for i, model in enumerate(self.model):
        # looping every layer
            output, (state, cell) = model(output)
            state_list.append(state)
            cell_list.append(cell)

        return output, (torch.stack(state_list, dim=0), torch.stack(cell_list, dim=0))

def main():
    model = Encoder(7, 7, 2048, 256, 3, torch.nn.Tanh())
    model.cuda()
    img = torch.zeros(1, 2048, 7, 7)
    img = img.cuda()
    output, (states, cells) = model(img)
    print(output.shape, states.shape, cells.shape)

if __name__ == "__main__":
    main()
