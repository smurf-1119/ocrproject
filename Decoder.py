from tkinter.messagebox import NO
import torch
from torch import nn
from attention import *
from context_gate import context_gate

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, en_hidden_size, de_hidden_size, attention_size, num_layers, drop_prob=0):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.context_gate = context_gate(embed_size, en_hidden_size, de_hidden_size)
        self.attention = attention_model(en_hidden_size + de_hidden_size, attention_size)
        # LSTM的输入包含attention输出的c和上一次, 所以尺寸是 hidden_size*2
        self.lstm = nn.LSTM(en_hidden_size + embed_size, de_hidden_size, 
                          num_layers=num_layers, dropout=drop_prob)
        self.out = nn.Linear(de_hidden_size, vocab_size)
        self.en_hidden_size = en_hidden_size
        self.de_hidden_size = de_hidden_size
        self.num_layers = num_layers
        # transforms hidden size
        if en_hidden_size != de_hidden_size:
            self.enc2dec_state = nn.Conv1d(en_hidden_size, de_hidden_size, kernel_size=1)
            self.enc2dec_cell = nn.Conv1d(en_hidden_size, de_hidden_size, kernel_size=1)
        else:
            self.W_enc2dec_state = None
            self.W_enc2dec_cell = None
    
    def forward(self, cur_input, state, enc_states, attention_mask):
        """
        params:
            @cur_input{tensor}: (batch, )
            @state{tupple}: (tensor (num_layers, batch, hiddens_size), 
                             tensor (num_layers, batch, hiddens_size))
            @enc_states{tensor}: (seq_len, batch, hidden_size)
            @attention_mask{tensor}: (batch, seq_len)
        """
        # 使用注意力机制计算背景向量
        c = attention_forward(self.attention, enc_states, state[0][-1], attention_mask)
        # embeding
        emb = self.embedding(cur_input)
        # calculate context gate
        c = self.context_gate(state[0][-1], emb, c) * c
        # 将嵌入后的输入和背景向量在特征维连结, (批量大小, num_hiddens+embed_size)
        input_and_c = torch.cat((emb, c), dim=1) 
        # 为输入和背景向量的连结增加时间步维，时间步个数为1
        output, state = self.lstm(input_and_c.unsqueeze(0), state)
        # 移除时间步维，输出形状为(批量大小, 输出词典大小)
        output = self.out(output).squeeze(dim=0)
        return output, state
        
    def begin_state(self, enc_state):
        # 直接将编码器最终时间步的隐藏状态作为解码器的初始隐藏状态
        if self.en_hidden_size != self.de_hidden_size:
            dec_state = self.enc2dec_state(enc_state[0].permute(1,2,0))
            dec_state = dec_state.permute(2,0,1).contiguous() #后续需要用view
            dec_cell = self.enc2dec_cell(enc_state[1].permute(1,2,0))
            dec_cell = dec_cell.permute(2,0,1).contiguous()
        
            enc_state = (dec_state, dec_cell)

        return enc_state
        