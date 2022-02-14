'''
train code
'''

from zmq import device
from ANMT import ANMT
from recognitiondataset import getDataLoader
from utils.data_utilis import build_vocab
import torch
from torch import nn
from tqdm import tqdm
import argparse
import os
from configs import config
import numpy as np

def set_interact_args():
    """
    Sets up the training arguments.
    """
    parser = argparse.ArgumentParser()

    # 数据路径
    parser.add_argument('--vocab_path', default='./dict.txt', type=str, required=False, help='输入字典路径')
    parser.add_argument('--data_dir', default='./AEC_recognition', type=str, required=False, help='输入数据路径')

    # 模式
    parser.add_argument('--mode', default='train', type=str, required=False, help='模式')

    # 训练参数
    parser.add_argument('--lr', default=0.01, type=float, required=False, help='学习率')
    parser.add_argument('--batch_size', default=10, type=int, required=False, help='batch')
    parser.add_argument('--num_epoch', default=5, type=int, required=False, help='epoch')
    parser.add_argument('--max_seq', default=35, type=int, required=False, help='最大标签长度')
    parser.add_argument('--opt', default='sgd', type=str, required=False, help='训练方式')
    parser.add_argument('--device', default='cuda', type=str, required=False, help='运行设备')
    parser.add_argument('--model_path',default='./model',type=str,required=False,help='模型位置')
    
    return parser.parse_args()

def train(model, DataLoader, max_seqence, lr, batch_size, num_epochs, loss, opt, device, vocab_path='./dict.txt',data_dir='./AEC_recognition',model_path='./model'):
    '''
    description: train model
    params:
        @model{torch.nn.module}: defaut ANMT;
        @DataLoader{torch.nn.DataLoader}: getDataloader;
        @max_seqence{int}: the max length of label;
        @lr{float}: learning rate;
        @batch_size{int}: batch;
        @num_epochs{int}: epoch;
        @loss: loss;
        @opt: adam or sgd;
        @device{torch.device}: cuda or cpu;
        @vocab_path{str};
        @data_dir{str};
        @model_path{str}
    return
        None
    '''
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) if opt == 'adam' else torch.optim.SGD(model.parameters(), lr=lr)
    if device == 'cuda':
        model.cuda()
    model.train()
    data_iter = DataLoader(mode='train', batch_size=batch_size, max_sequence=max_seqence, vocab_path=vocab_path,data_dir=data_dir)
    for epoch in tqdm(range(num_epochs), total=num_epochs):
        l_sum = 0.0
        for X, Y in tqdm(data_iter, total=len(data_iter)):

            if device == 'cuda':
                X = X.to(torch.device(device=device))
                Y = Y.to(torch.device(device=device))

            optimizer.zero_grad()
            l = model(X, Y, mode='train', loss=loss)
            l.backward()
            optimizer.step()
            l_sum += l.item()

        if (epoch + 1) % 10 == 0:
            print("epoch %d, loss %.3f" % (epoch + 1, l_sum / len(data_iter)))
        if (epoch + 1) % 100 == 0:
            evaluate(model,DataLoader=getDataLoader,device=device,max_sequence=max_seqence,batch_size=batch_size,vocab_path=vocab_path,data_dir=data_dir)
            model_name = f'recognition_{num_epochs}_{epoch}'
            torch.save(model.state_dict(),os.path.join(model_path,model_name))
            model.train()
            
def evaluate(model,DataLoader,device,max_sequence,batch_size,vocab_path,data_dir):
    '''
    description: make the evaluation of the model
    params:
        @model{torch.nn.module}: defaut ANMT;
        @DataLoader{torch.nn.DataLoader}: getDataloader;
        @max_seqence{int}: the max length of label;
        @batch_size{int}: batch;
        @device{torch.device}: cuda or cpu;
        @vocab_path{str};
        @data_dir{str};
        @model_path{str}
    return:
        None
    '''  
    model.eval()
    if device == 'cuda':
        model.cuda()
    accuracy_list = []
    valDataloader = DataLoader(mode='validation',batch_size=batch_size,max_sequence=max_sequence,vocab_path=vocab_path,data_dir=data_dir)
    for X,Y in tqdm(valDataloader,total=len(valDataloader)):
        if device == 'cuda':
            X = X.to(torch.device(device=device))
            Y = Y.to(torch.device(device=device))
        with torch.no_grad():
            accuracy = model(X,Y,mode='validation')
            accuracy_list.append(accuracy)
    print(np.mean(accuracy_list))

def main():
    # train parameters
    args = set_interact_args() 

    # model parameters
    configs = config() 

    # params
    embed_size = configs.embed_size
    hidden_size = configs.hidden_size
    attention_size = configs.attention_size
    height = configs.height
    width = configs.width
    feature_size = configs.feature_size

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr = args.lr
    batch_size = args.batch_size
    num_epochs = args.num_epoch
    max_seq = args.max_seq
    opt = args.opt
    device = args.device
    loss = nn.CrossEntropyLoss(reduction='none')

    mode = args.mode # train or validation

    # data path
    vocab_path = args.vocab_path
    data_dir = args.data_dir

    # build vocab
    vocab = build_vocab(vocab_path)

    # construct the model  
    model = ANMT(height, width, feature_size, embed_size, hidden_size, attention_size, vocab,device=device)

    #the model save path
    model_path = args.model_path

    # train
    train(model=model, DataLoader=getDataLoader, max_seqence=max_seq, lr=lr, batch_size=batch_size,num_epochs=num_epochs, loss=loss, opt=opt, device=device, vocab_path=vocab_path,data_dir=data_dir,model_path=model_path)
        
if __name__ == '__main__':
    main()
