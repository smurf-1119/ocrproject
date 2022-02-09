'''
get model params
'''

class config():
    def __init__(self) -> None:
        # params
        self.embed_size = 20 # 
        self.hidden_size = 256
        self.attention_size = 10

        # shape of resnet50 output
        self.height = 7
        self.width = 7
        self.feature_size = 2048