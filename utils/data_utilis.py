'''
help construct data
'''

from black import out
import torchtext.vocab as Vocab
import collections
import torch

PAD, BOS, EOS = '<pad>', '<bos>', '<eos>'

def process_label(max_seq_len: int, vocab: dict, label: str):
    '''
    description: given vocab, map strings of labels to integers array.
    params:
        @max_seq_len{int}: the max length of label seqence;
        @vocab{dict}: the dictionary to map the strings to integers;
        @label{str}: 
    return:
        out_label{tensor}
    '''
    out_label = []

    label = [vocab[ch] for ch in label]
    if len(label) + 1 > max_seq_len:
    # if the length of label is longer than the max seqence length, we truncate it.
        label[-1] = vocab[EOS]
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


