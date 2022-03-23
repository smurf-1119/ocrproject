'''
train code
'''
from tabnanny import check
import pandas as pd
from ANMT import ANMT
from recognitiondataset import getDataLoader,getDataLoader_multigpu
from utils.data_utilis import build_vocab,adjustLr,build_reverse_vocab,bleu
import torch
from torch import nn
from tqdm import tqdm
import argparse
import os
from configs import config
import numpy as np
import sys

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
    parser.add_argument('--drop_prob', default=0, type=float, required=False, help='失活概率')
    parser.add_argument('--batch_size', default=10, type=int, required=False, help='batch')
    parser.add_argument('--num_epoch', default=5, type=int, required=False, help='epoch')
    parser.add_argument('--max_seq', default=35, type=int, required=False, help='最大标签长度')
    parser.add_argument('--opt', default='sgd', type=str, required=False, help='训练方式')
    parser.add_argument('--device', default='cuda', type=str, required=False, help='运行设备')
    parser.add_argument('--model_path',default='./model',type=str,required=False,help='模型位置')
    parser.add_argument('--multi_gpu',default='True',type=str,required=False,help='多gpu训练')
    parser.add_argument('--local_rank',type=int)
    parser.add_argument('--model_name',type=str,required=False,help='预模型名称')
    parser.add_argument('--world_size',type=int,required=False,help='全局训练进程数')
    return parser.parse_args()

def train(model, DataLoader, max_seqence,batch_size, num_epochs, loss, optimizer,acc,device, model_name, args, ngpus_per_node, iteration, reversed_vocab,start_epoch=0,vocab_path='./dict.txt',data_dir='./AEC_recognition',model_path='./model'):
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
    model.train()
    data_iter = DataLoader(mode='train', batch_size=batch_size, max_sequence=max_seqence, vocab_path=vocab_path,data_dir=data_dir)
    iteration = iteration
    new_iteration = 0
    best_acc = acc
    if os.path.exists('./loss_epoch.csv'):
        loss_epoch_df = pd.read_csv('./loss_epoch.csv')
        acc_df = pd.read_csv('./acc.csv')
        train_acc_df = pd.read_csv('./train_acc.csv')
    else:
        loss_epoch_df = pd.DataFrame(columns = ['epoch','loss'])
        acc_df = pd.DataFrame(columns = ['epoch','accuracy'])
        train_acc_df = pd.DataFrame(columns=['epoch','accuracy'])

    if os.path.exists('./bleu.csv'):
        bleu_epoch_df = pd.read_csv('./bleu.csv')
    else:
        bleu_epoch_df = pd.DataFrame(columns=['epoch','bleu'])

    for epoch in tqdm(range(num_epochs), total=num_epochs,position=0,ncols=80):
        l_sum = 0.0
        for X, Y in tqdm(data_iter,total=len(data_iter),position=1,ncols=80):
            X = X.to(device)
            Y = Y.to(device)

            optimizer.zero_grad()
            l = model(X, Y, mode='train', loss=loss)
            l.backward()
            optimizer.step()
            l_sum += l.item()
            

            if iteration < 300000:
                iteration += 1
            else:
                new_iteration += 1

            # test print
            # if iteration % 10 == 0:
            #     print('loss is {}'.format(l_sum / iteration))
            #     pd.DataFrame(ls_list_iter).to_csv('./loss_iter.csv')
            #     sys.stdout.flush()

        loss_temp = pd.DataFrame({'epoch':[start_epoch+epoch+1],'loss':[l_sum / len(data_iter)]})
        loss_epoch_df = pd.concat([loss_epoch_df,loss_temp],axis=0).reset_index(drop=True)

        if iteration == 300000: #300k先降低
            optimizer = adjustLr(optimizer)
            iteration += 1
        if new_iteration == 100000: #每100k之后再降低
            optimizer = adjustLr(optimizer)
            new_iteration = 0
    
        if (epoch + 1) % 1 == 0:
            if args.local_rank % ngpus_per_node == 0:
                print("epoch %d, loss %.3f" % (epoch + 1, l_sum / len(data_iter)))
            loss_epoch_df.to_csv('./loss_epoch.csv',index=False)

        if (epoch+1) % 20 == 0:
            train_acc = evaluate(model,mode='train',DataLoader=getDataLoader,device=device,max_sequence=max_seqence,batch_size=batch_size,vocab_path=vocab_path,data_dir=data_dir)
            train_acc_temp = pd.DataFrame({'epoch':[start_epoch+epoch+1],'accuracy':[train_acc]})
            train_acc_df = pd.concat([train_acc_df,train_acc_temp],axis=0).reset_index(drop=True)
            train_acc_df.to_csv('train_acc.csv',index=False)
            model.train()

        if (epoch+1) % 20 == 0:
            score = evaluate_bleu(model,mode='validation',DataLoader=getDataLoader,device=device,max_sequence=max_seqence,batch_size=batch_size,vocab_path=vocab_path,data_dir=data_dir,reversed_vocab=reversed_vocab)
            bleu_score_temp = pd.DataFrame({'epoch':[start_epoch+epoch+1],'bleu':[score]})
            bleu_epoch_df = pd.concat([bleu_epoch_df,bleu_score_temp],axis=0).reset_index(drop=True)
            bleu_epoch_df.to_csv('./bleu.csv')
            model.train()

        if (epoch + 1) % 1 == 0: #每10个epoch保存一次模型, 并测试一次
            acc = evaluate(model,mode='validation',DataLoader=getDataLoader,device=device,max_sequence=max_seqence,batch_size=batch_size,vocab_path=vocab_path,data_dir=data_dir)
            if args.local_rank % ngpus_per_node == 0:
                print(f'accuracy: {acc:.2f}')
            acc_temp = pd.DataFrame({'epoch':[start_epoch+epoch+1],'accuracy':[acc]})
            acc_df = pd.concat([acc_df,acc_temp],axis=0).reset_index(drop=True)
            if acc > best_acc:
                if args.local_rank % ngpus_per_node == 0:
                    if model_name != None:
                        model_name = f'recognition_{start_epoch+num_epochs}_{start_epoch+epoch+1}.pth.tar'
                        state = {
                            'epoch':start_epoch + epoch + 1,
                            'state_dict':model.state_dict(),
                            'accuracy':acc,
                            'iteration':iteration,
                            'optimizer':optimizer.state_dict()
                        }
                        torch.save(state,os.path.join(model_path,model_name))
                    else:
                        model_name = f'recognition_{start_epoch+num_epochs}_{start_epoch+epoch+1}.pth.tar'
                        state = {
                            'epoch':epoch+1,
                            'state_dict':model.state_dict(),
                            'accuracy':acc,
                            'iteration':iteration,
                            'optimizer':optimizer.state_dict()
                        }
                        torch.save(state,os.path.join(model_path,model_name))
                    best_acc = acc
            acc_df.to_csv('./acc.csv',index=False)
            model.train()
            
def evaluate(model,mode,DataLoader,device,max_sequence,batch_size,vocab_path,data_dir):
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
    accuracy_list = []
    valDataloader = DataLoader(mode=mode,batch_size=batch_size,max_sequence=max_sequence,vocab_path=vocab_path,data_dir=data_dir,val_mode='accuracy')

    for X,Y in valDataloader:
        X = X.to(device)
        Y = Y.to(device)
    with torch.no_grad():
        accuracy = model(X,Y,mode='validation')
        accuracy_list.append(accuracy.item())

    acc = sum(accuracy_list) / len(accuracy_list)
    
    return acc


def evaluate_bleu(model,mode,DataLoader,device,max_sequence,batch_size,vocab_path,data_dir,reversed_vocab):
    model.eval()
    scores = []
    valDataloader = DataLoader(mode=mode,batch_size=batch_size,max_sequence=max_sequence,vocab_path=vocab_path,data_dir=data_dir,val_mode='bleu')
    for X,Y in valDataloader:
        X = X.to(device)
    with torch.no_grad():
        pre_label = ''
        predict_tokens = model(X,mode='predict')
        for token in predict_tokens:
            pre_label = pre_label + reversed_vocab[token]
        labels = Y[0].tolist()
        label = ''
        for lab in labels:
            if reversed_vocab[lab] in ['<pad>', '<bos>', '<eos>']:
                continue
            else:
                label = label + reversed_vocab[lab]
        score = bleu(pre_label,label,k=2)
        scores.append(score)
    bleu_score = sum(scores) / len(scores)
    return bleu_score


def load_model(model,device,multi_gpu,args):
    '''
    description: multi_gpu or single gpu
    params:
        @model{torch.nn.module}: defaut ANMT;
        @device: cuda or cpu
        @multi_gpu: more than one gpu or not;
    return:
        model
    '''  
    if device == 'cuda':
        if multi_gpu and torch.cuda.device_count() > 1:
            ngpus_per_node = torch.cuda.device_count()
            # use multi gpu training
            if args.local_rank % ngpus_per_node == 0:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
            torch.distributed.init_process_group(backend="nccl")
            local_rank = torch.distributed.get_rank()
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            model.to(device)
            model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[local_rank])
        else:
            device = torch.device("cuda")
            model.to(device)
    else:
        device = torch.device("cpu")
        model.to(device)

    return model,device



def main():
    # train parameters
    args = set_interact_args() 

    # model parameters
    configs = config() 

    # params
    embed_size = configs.embed_size
    en_hidden_size = configs.en_hidden_size
    de_hidden_size = configs.de_hidden_size
    attention_size = configs.attention_size
    height = configs.height
    width = configs.width
    feature_size = configs.feature_size
    num_layers = configs.num_layers

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr = args.lr
    batch_size = args.batch_size
    num_epochs = args.num_epoch
    drop_prob = args.drop_prob
    max_seq = args.max_seq
    opt = args.opt
    device = args.device
    ngpus_per_node = torch.cuda.device_count()

    multi_gpu = args.multi_gpu
    if multi_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
        DataLoader = getDataLoader_multigpu
    else:
        DataLoader = getDataLoader

    loss = nn.CrossEntropyLoss(reduction='none')

    # data path
    vocab_path = args.vocab_path
    data_dir = args.data_dir

    # build vocab
    vocab = build_vocab(vocab_path)


    reversed_vocab = build_reverse_vocab(vocab_path)

    # construct the model  
    model = ANMT(height=height, width=width, input_channel=feature_size, embed_size=embed_size, en_hidden_size=en_hidden_size, de_hidden_size=de_hidden_size, attention_size=attention_size, vocab=vocab, max_seq=max_seq, num_layers=num_layers, drop_prob=drop_prob, device=device)
    #model.backbone.load_state_dict(torch.load('./model/resnet50_new.pth', map_location='gpu'))
    ###########################################
    #pretrain_path = './model/resnet50_new.pth'
    #model.backbone,device = load_model(model.backbone,device,multi_gpu,args) 
    #state_dict = torch.load(pretrain_path)  
    #model.load_state_dict(state_dict)  
    #print('load success!')
    ###########################################
    
    #construct the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) if opt == 'adam' else torch.optim.SGD(model.parameters(), lr=lr)
    
    #the model save path
    model_path = args.model_path

    model_name = args.model_name
    if model_name != None:
        pretrain_path = os.path.join(model_path,model_name)
        model,device = load_model(model,device,multi_gpu,args) 
        checkpoint = torch.load(pretrain_path)
        epoch = checkpoint['epoch']
        state_dict = checkpoint['state_dict']
        acc = checkpoint['accuracy']
        iteration = checkpoint['iteration']
        optimizer.load_state_dict(checkpoint['optimizer'])   
        model.load_state_dict(state_dict)
        if args.local_rank % ngpus_per_node == 0:  
            print('load success!')
    else:
        model,device = load_model(model,device,multi_gpu,args)
        model.module.backbone.load_state_dict(torch.load('./model/resnet18_new.pth'))
        acc = 0.0
        epoch = 0
        iteration = 0

    # train
    train(model=model, DataLoader=DataLoader, max_seqence=max_seq,optimizer=optimizer,acc=acc,batch_size=batch_size,num_epochs=num_epochs,iteration=iteration,loss=loss,device=device,model_name=model_name, args=args, ngpus_per_node=ngpus_per_node, start_epoch=epoch,reversed_vocab=reversed_vocab,vocab_path=vocab_path,data_dir=data_dir,model_path=model_path)
        
if __name__ == '__main__':
    main()