'''
recognition model
'''

from utils.data_utilis import process_label, build_vocab
from Encoder import Encoder
from Decoder import Decoder
from resnet50_pre import resnet50
import torch
from torch import nn

class ANMT(nn.Module):
    def __init__(self, height: int, width: int, input_channel: int, embed_size: int, en_hidden_size: int, de_hidden_size, attention_size: int, vocab,max_seq=None,num_layers=3, drop_prob=0, non_linear=torch.nn.Tanh(), device='cuda') -> None:
        super(ANMT, self).__init__()
        self.vocab_size = len(vocab)
        self.vocab = vocab
        self.backbone = resnet50()
        self.en_seq_len = height * width #the lenth of encoder states
        self.Encoder = Encoder(height, width, input_channel, en_hidden_size, num_layers, non_linear)
        self.Decoder = Decoder(self.vocab_size, embed_size, en_hidden_size, de_hidden_size, attention_size, num_layers, drop_prob)
        self.device = torch.device(device) 
        self.max_seq = max_seq
    def forward(self, x: torch.Tensor, Y=None, mode='train', loss=None):
        PAD, BOS, EOS = '<pad>', '<bos>', '<eos>'
        batch_size = x.shape[0]

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

        attention_mask = torch.ones(self.en_seq_len, batch_size, 1, device=self.device,requires_grad=False)
        if mode == 'train':
            mask, num_not_pad_tokens = torch.ones((batch_size,), device=self.device), 0
            l = torch.tensor([0.0], device=self.device)
            
            for y in Y.permute(1, 0):# Y shape: (batch, seq_len)
                dec_output, dec_state = self.Decoder(dec_input, dec_state, enc_output, attention_mask)
                l = l + (mask * loss(dec_output, y)).sum()
                dec_input = y # 使用强制教学
                num_not_pad_tokens += mask.sum().item()
                # EOS后面全是PAD. 下面一行保证一旦遇到EOS接下来的循环中mask就一直是0
                mask = mask * (y != self.vocab[EOS]).float()
            # return dec_output
            return l / num_not_pad_tokens
        elif mode == 'validation':
            mask, num_not_pad_tokens = torch.ones((batch_size,), device=self.device), 0
            correct = torch.tensor([0.0],device=self.device)
            for y in Y.permute(1,0):
                dec_output, dec_state = self.Decoder(dec_input, dec_state, enc_output, attention_mask)
                pred,index = torch.max(dec_output,dim=1) #取最大值的位置
                correct = correct+ (mask*(index==y).float()).sum()
                dec_input = index
                num_not_pad_tokens +=mask.sum().item()
                mask = mask * (y != self.vocab[EOS]).float()
            
            return correct/num_not_pad_tokens
        else:
            predict = []
            for _ in range(self.max_seq):
                
                dec_output,dec_state = self.Decoder(dec_input, dec_state, enc_output, attention_mask)
                _,index = torch.max(dec_output,dim=1)
                if int(index.item()) == self.vocab[EOS]:
                    return predict
                else:
                    dec_input = index
                    predict.append(int(index.item()))
            return predict

            
                
def main():
    # params
    embed_size = 128
    en_hidden_size = 256
    de_hidden_size = 512
    attention_size = 128
    num_layers = 3
    max_seq = 35
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loss = nn.CrossEntropyLoss(reduction='none')

    # build vocab
    vocab = build_vocab('./dict.txt')

    test_str = ['5*8=40', '5*8=40']

    # construct the label
    label = torch.cat((process_label(max_seq, vocab, test_str[0]), process_label(max_seq, vocab, test_str[0])), dim=0)
    
    # construct the model  
    model = ANMT(7, 7, 2048, embed_size, en_hidden_size, de_hidden_size, attention_size, vocab, device=device, num_layers=num_layers)
    img = torch.zeros((2,3,224,224))
    img = img.cuda()
    label = label.cuda()
    model.cuda()

    # forward
    l = model(img, label, mode='train', loss=loss)
    print(l)

if __name__ == '__main__':
    main()