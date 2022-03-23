"""
help construct the dataeset
"""

import torch
import cv2
import numpy as np
import math
import collections

PAD, BOS, EOS = '<pad>', '<bos>', '<eos>'

def process_label(max_seq_len: int, vocab: dict, label: str):
    '''
    description: given vocab, map strings of labels to integers array.
    params:
        @max_seq_len{int}: the max length of label seqence;
        @vocab{dict}: the dictionary to map the strings to integers;
        @labels{list}: (batch, )
    return:
        out_label{tensor}
    '''
    out_label = []

    label = [vocab[ch] for ch in label]
    if len(label) + 1 > max_seq_len:
    # if the length of label is longer than the max seqence length, we truncate it.
        label = np.array(label)
        label = (label[:max_seq_len-1]).tolist() + [vocab[EOS]]
    else:
        label += [vocab[EOS]] + [vocab[PAD]] * (max_seq_len - len(label) - 1)

    out_label.append(label)
    
    return torch.tensor(out_label)

def build_vocab(dict_name: str):
    PAD, BOS, EOS = '<pad>', '<bos>', '<eos>'

    with open(dict_name, encoding='utf-8') as f:
        dic = f.read().split('\n')
    dic = [PAD, BOS, EOS] + dic

    vocab = {key:value for value, key in enumerate(dic)}
    return vocab

def build_reverse_vocab(dict_name:str):
    PAD, BOS, EOS = '<pad>', '<bos>', '<eos>'

    with open(dict_name, encoding='utf-8') as f:
        dic = f.read().split('\n')
    dic = [PAD, BOS, EOS] + dic

    reversed_vocab = {value:key for value, key in enumerate(dic)}
    return reversed_vocab

def letterbox_image(img, inp_dim:int):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)   
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)
    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w, :] = resized_image
    return canvas

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """
    img = letterbox_image(img, (inp_dim, inp_dim))
    img = img[:,:,::-1].transpose((2,0,1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img

def adjustLr(optimizer):
    current_lr = optimizer.param_groups[0]['lr']
    lr = current_lr * 0.5
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def bleu(pred_tokens, label_tokens, k):
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[''.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[''.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[''.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score