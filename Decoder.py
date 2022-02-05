from turtle import forward
import torch
from torch import nn
from attention import *

class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, attention_size, num_layers=1, drop_prob=0):
        super(Decoder, self).__init__()
        self.attention = attention_model(2*hidden_size, attention_size)
        # LSTM的输入包含attention输出的c和上一次, 所以尺寸是 hidden_size*2
        self.lstm = nn.LSTM(2*hidden_size, hidden_size, 
                          num_layers=num_layers, dropout=drop_prob)
        self.out = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, cur_input, state, enc_states):
        """
        cur_input shape: (batch, )
        state shape: (num_layers, batch, hiddens_size)
        """
        # 使用注意力机制计算背景向量
        c = attention_forward(self.attention, enc_states, state[0][-1])
        # 将嵌入后的输入和背景向量在特征维连结, (批量大小, num_hiddens+embed_size)
        input_and_c = torch.cat((cur_input, c), dim=1)
        # 为输入和背景向量的连结增加时间步维，时间步个数为1
        output, state = self.lstm(input_and_c.unsqueeze(0), state)
        # 移除时间步维，输出形状为(批量大小, 输出词典大小)
        output = self.out(output).squeeze(dim=0)
        return output, state
        
    def begin_state(self, enc_state):
        # 直接将编码器最终时间步的隐藏状态作为解码器的初始隐藏状态
        return enc_state
        