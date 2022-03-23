'''
get model params
'''

class config():
    def __init__(self) -> None:
        # params
        self.embed_size = 128 # 
        self.en_hidden_size = 256
        self.de_hidden_size = 512
        self.attention_size = 128
        self.num_layers = 3

        # shape of resnet50 output
        self.height = 7
        self.width = 7
        self.feature_size = 2048