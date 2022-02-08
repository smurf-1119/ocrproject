'''
recognition model
'''

from zmq import device
from utils.data_utilis import process_label, build_vocab
from Encoder import Encoder
from Decoder import Decoder
from resnet50 import resnet50
import torch
from torch import nn

class ANMT(nn.Module):
    def __init__(self, height: int, width: int, input_channel: int, embed_size: int, hidden_size: list, attention_size: int, vocab, num_layers=3, drop_prob=0, non_linear=torch.nn.Tanh(), device='cuda') -> None:
        super(ANMT, self).__init__()
        self.vocab_size = len(vocab)
        self.vocab = vocab
        self.backbone = resnet50()
        self.Encoder = Encoder(height, width, input_channel, hidden_size, num_layers, non_linear)
        self.Decoder = Decoder(self.vocab_size, embed_size, hidden_size, attention_size, num_layers, drop_prob)
        self.device = torch.device(device) 
    
    def forward(self, x: torch.Tensor, Y: torch.Tensor, mode='train'):
        PAD, BOS, EOS = '<pad>', '<bos>', '<eos>'

        feature_map = self.backbone(x)
        
        # enc_output (batch, hidden_size, width, height)
        # enc_states (batch, hidden_size, width, height)
        # enc_cells  (batch, hidden_size, width, height)
        enc_output, (enc_states, enc_cells) = self.Encoder(feature_map)

        # After reshape->(seq_len, batch, hidden_size)
        enc_output = enc_output.view(enc_output.shape[0], enc_output.shape[1], -1).permute(2,0,1)

        # After reshape->(num_layers, batch, hidden_size)
        enc_state = enc_states.view(enc_states.shape[0], enc_states.shape[1], enc_states.shape[2], -1).permute(3,0,1,2)[-1] # get the last of sequence elements

        # After reshape->(num_layers, batch, hidden_size)
        enc_cell = enc_cells.view(enc_cells.shape[0], enc_cells.shape[1], enc_cells.shape[2], -1).permute(3,0,1,2)[-1] # get the last of sequence elements


        dec_state = self.Decoder.begin_state((enc_state[:].contiguous(), enc_cell[:].contiguous()))
        
        dec_input = torch.tensor([self.vocab[BOS]] * enc_output.shape[1], device=self.device)

        for y in Y.permute(1, 0):# Y shape: (batch, seq_len)
            dec_output, dec_state = self.Decoder(dec_input, dec_state, enc_output)
            dec_input = y # 使用强制教学

        return dec_output



def main():
    # params
    embed_size = 20
    hidden_size = 256
    attention_size = 10
    max_seq = 9
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # build vocab
    vocab = build_vocab('./dict.txt')

    test_str = '5*8=40'

    # construct the label
    label = process_label(max_seq, vocab, test_str)
    
    # construct the model  
    model = ANMT(7, 7, 2048, embed_size, hidden_size, attention_size, vocab, device=device)
    img = torch.zeros((1,3,224,224))
    img = img.cuda()
    label = label.cuda()
    model.cuda()

    # forward
    output = model(img, label)
    print(output.shape)

if __name__ == '__main__':
    main()
