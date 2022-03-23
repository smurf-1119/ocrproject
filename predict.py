
import torch
from ANMT import ANMT
import argparse
from configs import config
import os
from utils.data_utilis import build_vocab,prep_image,build_reverse_vocab
import cv2
import numpy as np
from tqdm import tqdm
import json

def set_interact_args():
    """
    Sets up the training arguments.
    """
    parser = argparse.ArgumentParser()

    # 数据路径
    parser.add_argument('--vocab_path', default='./dict.txt', type=str, required=False, help='输入字典路径')
    parser.add_argument('--data_dir', default='./AEC_recognition', type=str, required=False, help='输入测试数据路径')

    # 训练参数
    parser.add_argument('--max_seq', default=35, type=int, required=False, help='最大标签长度')
    parser.add_argument('--device', default='cuda', type=str, required=False, help='运行设备')
    parser.add_argument('--model_path',default='./model',type=str,required=False,help='模型位置')
    # parser.add_argument('--multi_gpu',default='True',type=str,required=False,help='多gpu训练')
    parser.add_argument('--local_rank',type=int)
    parser.add_argument('--model_name',default='recognition_442_392.pth.tar',type=str,required=False,help='预模型名称')
    parser.add_argument('--output_path',type=str,default='./result')
    # parser.add_argument('--world_size',type=int,required=False,help='全局训练进程数')
    return parser.parse_args()

def load_model(model,device):
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
        if torch.cuda.device_count() > 1:
            torch.distributed.init_process_group(backend="nccl")
            local_rank = torch.distributed.get_rank()
            torch.cuda.set_device(local_rank)
            local_rank = 0
            device = torch.device("cuda", local_rank)
            model.to(device)
            # model = torch.nn.DataParallel(model)
            model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[local_rank])
        else:
            device = torch.device("cuda")
            model.to(device)
    else:
        device = torch.device("cpu")
        model.to(device)

    return model,device

def predict(model,images,reversed_vocab,device):
    model.eval()
    input_dim = 224
    output = {}
    for image in tqdm(images, total=len(images)):
        image_name = image.split('\\')[-1]
        temp = ''
        image = np.array(cv2.imread(image))
        image = prep_image(image,input_dim)
        image = image.to(device)
        predict_tokens = model(image,mode='predict')
        for token in predict_tokens:
            temp = temp + reversed_vocab[token]
        output[image_name] = temp[:]

    return output


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
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

    ngpus_per_node = torch.cuda.device_count()

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    max_seq = args.max_seq
    device = args.device

    # data path
    vocab_path = args.vocab_path
    data_dir = args.data_dir

    #construct the images path
    image_data_dir = os.path.join(data_dir,'test','img')
    images = [os.path.join(image_data_dir,image) for image in os.listdir(image_data_dir)]
    
    # build vocab
    vocab = build_vocab(vocab_path)
    
    #build reversed vocab
    reversed_vocab = build_reverse_vocab(vocab_path)

    # construct the model  
    model = ANMT(height=height, width=width, input_channel=feature_size, embed_size=embed_size, en_hidden_size=en_hidden_size, de_hidden_size=de_hidden_size, attention_size=attention_size, vocab=vocab, max_seq=max_seq, num_layers=num_layers, device=device)
            
    #the model save path
    model_path = args.model_path

    #the model name
    model_name = args.model_name
    # model_name = None

    output_path = os.path.join(data_dir,'test',args.output_path)

    if model_name != None:
        pretrain_path = os.path.join(model_path,model_name)
        model,device = load_model(model,device) 
        checkpoint = torch.load(pretrain_path)
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)
        if args.local_rank % ngpus_per_node == 0:
            print('load success!')
    else:
        model,device = load_model(model,device)

    # predict
    output = predict(model=model,images=images,reversed_vocab=reversed_vocab,device=device)
    if args.local_rank % ngpus_per_node == 0:
        with open(os.path.join(output_path,'result.json'),'w') as f:
            json.dump(output,f,indent=2)
        print('finish')

if __name__ == '__main__':
    main()

