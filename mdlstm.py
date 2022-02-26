'''
mdlstm cell
'''

import torch
import torch.nn as nn
import numpy as np


import matplotlib.pyplot as plt
from utils.tensor_helper import cuda_available


def imshow(inp):
    inp = inp.numpy()[0]
    mean = 0.1307
    std = 0.3081
    inp = ((mean * inp) + std)
    plt.imshow(inp, cmap='gray')
    plt.show()

class MDLSTMCell(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        params:
           @in_channels{int}:  Size of the embedding.
           @out_channels{int}: Size of the output.
        """
        super(MDLSTMCell, self).__init__()
        self.out_channels = out_channels
        x_parameters_shape = (in_channels, out_channels * 5)
        h_parameters_shape = (out_channels, out_channels * 5)
        bias_shape = (out_channels * 5)
        self.w = nn.parameter.Parameter(torch.empty(x_parameters_shape))
        self.u0 = nn.parameter.Parameter(torch.empty(h_parameters_shape))
        self.u1 = nn.parameter.Parameter(torch.empty(h_parameters_shape))
        self.b = nn.parameter.Parameter(torch.zeros(bias_shape))

    def initialize_weights(self):
        
        torch.nn.init.xavier_uniform_(self.w)
        torch.nn.init.xavier_uniform_(self.u0)
        torch.nn.init.xavier_uniform_(self.u1)

    def forward(self, x, c_prev_dim0, h_prev_dim0, c_prev_dim1, h_prev_dim1):
        """
        - For each output channel apply the same weights to each input channel
        - Sum the results
        :param x: Entry of shape (batch_size, in_channels)
        :param c_prev_dim0:  previous state cell (batch_size, out_channels) along the 1st dimension
        :param h_prev_dim0:  previous hidden state (batch_size, out_channels) along the 1st dimension
        :param c_prev_dim1:  previous state cell (batch_size, out_channels) along the 2nd dimension
        :param h_prev_dim1:  previous hidden state (batch_size, out_channels) along the 2nd dimension
        :return: Tuple[c, h] each being of dimension (batch_size, out_channels) which are the current state and hidden
        state for the current inputs
        """
        """
        print(f"x's shape is {x.shape}")
        print(f"h_prev_dim0's shape is {h_prev_dim0.shape}")
        print(f"h_prev_dim0[i, :]'s shape is {h_prev_dim0[0,:].shape}")
        """
        """
        Computes the current activation of this cell as described in
        https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html?highlight=lstm#torch.nn.LSTM
        and
        https://link.springer.com/chapter/10.1007%2F978-3-540-74690-4_56
        """
        gates = x @ self.w + h_prev_dim0 @ self.u0 + h_prev_dim1 @ self.u1 + self.b
        it, ft, gt, ot, lt = gates.chunk(5, 1)
        it = torch.sigmoid(it)
        ft = torch.sigmoid(ft)
        gt = torch.tanh(gt)
        ot = torch.sigmoid(ot)
        lt = torch.sigmoid(lt)
        ct = ft * ((lt * c_prev_dim0) + ((1 - lt) * c_prev_dim1)) + it * gt
        ht = ot * torch.tanh(ct)

        return ct, ht

class MDLSTM(nn.Module):
    '''
    description: given tensor(batch, channels, height, width), output the encoded feature->
        output(batch,channels,height,width), 
        states(num_layers, batch, channels, height, width),
        cells(num_layers, batch, channels, height, width)
    '''
    def __init__(self, height: int, width: int, in_channels: int, out_channels: int):
        '''
        param:
            @height{int}:the height of input image;
            @width{int}:the width of input image;
            @in_channels{int}:the channels of input image;
            @out_channels{int}: the channels of output image.
        '''
        super(MDLSTM, self).__init__()
        self.out_channels = out_channels
        self.width = width
        self.height = height
        # One LSTM per direction
        self.lstm_lr_tb = MDLSTMCell(in_channels=in_channels, out_channels=out_channels)
        self.lstm_rl_tb = MDLSTMCell(in_channels=in_channels, out_channels=out_channels)
        self.lstm_lr_bt = MDLSTMCell(in_channels=in_channels, out_channels=out_channels)
        self.lstm_rl_bt = MDLSTMCell(in_channels=in_channels, out_channels=out_channels)
        self.params = [self.lstm_lr_tb, self.lstm_rl_tb, self.lstm_lr_bt, self.lstm_rl_bt]
        self.fold = torch.nn.Fold(output_size=(self.height, self.width), kernel_size=(1, 1))

        self.initialize_weights()

    def initialize_weights(self):
        # initial the params
        self.lstm_lr_tb.initialize_weights()
        self.lstm_rl_tb.initialize_weights()
        self.lstm_lr_bt.initialize_weights()
        self.lstm_rl_bt.initialize_weights()

    def flipped_image(self, x: torch.Tensor, direction: int):
        # do flip operation to get 4 different directions 
        if direction == 0: # LRTP
            return x
        elif direction == 1: # RLTB
            return torch.flip(x, (3,))
        elif direction == 2: # LRBT
            return torch.flip(x, (2,))
        elif direction == 3: # RLBT
            return torch.flip(x, (2, 3,))

    def forward(self, x: torch.Tensor):
        """
        param 
            @x: Tensor of size (batch_size, in_channels, height, width)
        return: 
            @output:Tensor of size (batch_size, out_channels, height, width);
            @states:Tensor of size (num_layers, batch_size, out_channels, height, width);
            @cells:Tensor of size (num_layers, batch_size, out_channels, height, width);
        """
        # For each direction we're going to compute hidden_states and their activations
        global_hidden_states = len(self.params) * [None]
        global_cell_states = len(self.params) * [None]
        streams = [torch.cuda.Stream() for _ in self.params] if cuda_available else []
        if cuda_available:
            torch.cuda.synchronize()
        
        # looping 4 direction 
        for i, lstm in enumerate(self.params):
            x_ordered = self.flipped_image(x, direction=i)
            #imshow(x_ordered[0])
            # run code sychronizedly
            if cuda_available:
                stream = streams[i]
                with torch.cuda.stream(stream):
                    hidden_states_direction, cell_states_direction = self.do_forward(x_ordered, lstm)
            else:
                hidden_states_direction, cell_states_direction = self.do_forward(x_ordered, lstm)
            global_hidden_states[i] = self.flipped_image(hidden_states_direction, direction=i)
            global_cell_states[i] = self.flipped_image(cell_states_direction, direction=i)
        if cuda_available:
            torch.cuda.synchronize()
        # Each element in global_hidden_states is of shape (batch, channel, height, width)
        # Needs to be transposed because we stacked by direction while we expect the first dimension to be batch
        # return torch.stack(global_hidden_states, dim=1) # (batch, 4, channel, height, width) = stacked.shape
        # print(torch.mean(global_hidden_states, dim=1).shape)
        return torch.mean(torch.stack(global_hidden_states, dim=1), dim = 1), torch.mean(torch.stack(global_cell_states, dim=1), dim = 1)

    def do_forward(self, x, lstm):
        batch_size, in_channels, height, width = x.shape
        hidden_states_direction = []
        cell_states_direction = []
        i = -1
        for y_height in range(height):
            for x_width in range(width):
                i += 1
                # If we're on the first row the previous element is the vector of the good shape with 0s
                if y_height == 0:
                    prev_0_c = torch.zeros((batch_size, self.out_channels), requires_grad=False).to(device=x.device)
                    prev_0_h = torch.zeros((batch_size, self.out_channels), requires_grad=False).to(device=x.device)
                else:
                    # Otherwise we get back the previously computed c and h for this direction and coordinates
                    # So the tensors are of the shape (batch_size, out_channels)
                    idx_to_prev = i - width
                    prev_0_c = cell_states_direction[idx_to_prev]
                    prev_0_h = hidden_states_direction[idx_to_prev]
                # If we're on the first column the previous element is the vector of the good shape with 0s
                if x_width == 0:
                    prev_1_c = torch.zeros((batch_size, self.out_channels), requires_grad=False).to(device=x.device)
                    prev_1_h = torch.zeros((batch_size, self.out_channels), requires_grad=False).to(device=x.device)
                else:
                    # Otherwise we get back the previously computed c and h for this direction and coordinates
                    # So the tensors are of the shape (batch_size, out_channels)
                    idx_to_prev = i - 1
                    prev_1_c = cell_states_direction[idx_to_prev]
                    prev_1_h = hidden_states_direction[idx_to_prev]
                # The current input is a tensor of shape (batch_size, input_channels) at coordinates (x,y)
                current_input = x[:, :, y_height, x_width]
                cs, hs = lstm(current_input, prev_0_c, prev_0_h, prev_1_c, prev_1_h)
                cell_states_direction.append(cs)
                hidden_states_direction.append(hs)
        return self.fold(torch.stack(hidden_states_direction, dim=2)), self.fold(torch.stack(cell_states_direction, dim=2))
