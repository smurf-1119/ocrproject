from Encoder import *
from Decoder import *

class ANMT(nn.Module):
    def __init__(self, height: int, width: int, input_channel: int, vocab_size: int, hidden_size: int, attention_size: int, num_layers=1, drop_prob=0) -> None:
        super(ANMT, self).__init__()
        self.Encoder = Encoder(height, width, input_channel, hidden_size)
        self.Decoder = Decoder(vocab_size, hidden_size, attention_size, num_layers, drop_prob)
    
    def forward(self, x: torch.Tensor):
        embbeding = self.Encoder(x)
        embbeding = embbeding.view(embbeding.shape[0], embbeding.shape[1], -1).permute(2,0,1).unsqueeze(0)
        state = self.Decoder.begin_state(embbeding[:,-1,:,:]).contiguous()
        cell = torch.zeros_like(state, device=torch.device('cuda' if cuda_available else 'cpu'))
        state = (state, cell)

        for inp in embbeding[-1]:
            output, state = self.Decoder(inp, state, embbeding[-1])

        return output

model = ANMT(7, 7, 2048, 20, 300, 10)
img = torch.zeros(1, 2048, 7, 7)
img = img.cuda()
model.cuda()
output = model(img)
print(output.shape)