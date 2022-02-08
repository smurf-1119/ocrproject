import torch
from torch import nn
import torch.nn.functional as F

def attention_model(input_size, attention_size):
    '''
    description: construct the attention model.
    params: 
        @input_size{int}:     the lenhth of the input tensor.
        @attention_size{int}: the length of the output tensor.
    '''
    model = nn.Sequential(nn.Linear(input_size, 
                                    attention_size, bias=False),
                          nn.Tanh(),
                          nn.Linear(attention_size, 1, bias=False))
    return model

def attention_forward(model, enc_states, dec_state):
    '''
    description: implement the attention model.
    params:
        @model{function}: the attention model.
        @enc_states: the output of the encoder's last layer.
        @dec_state: the current hiddent state of the decoder's output. 
    '''
    # 将解码器隐藏状态广播到和编码器隐藏状态形状相同后进行连结
    dec_states = dec_state.unsqueeze(dim=0).expand_as(enc_states)
    enc_and_dec_states = torch.cat((enc_states, dec_states), dim=2)
    e = model(enc_and_dec_states)  # 形状为(时间步数, 批量大小, 1)
    alpha = F.softmax(e, dim=0)  # 在时间步维度做softmax运算
    return (alpha * enc_states).sum(dim=0)  # 返回背景变量

