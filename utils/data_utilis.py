import torchtext.vocab as Vocab
import collections
import torch

PAD, BOS, EOS = '<pad>', '<bos>', '<eos>'
# 将一个序列中所有的词记录在all_tokens中以便之后构造词典，然后在该序列后面添加PAD直到序列
# 长度变为max_seq_len，然后将序列保存在all_seqs中
def process_one_seq(seq_tokens, all_tokens, all_seqs, max_seq_len):
    all_tokens.extend(seq_tokens)
    seq_tokens += [EOS] + [PAD] * (max_seq_len - len(seq_tokens) - 1)
    all_seqs.append(seq_tokens)

# 使用所有的词来构造词典。并将所有序列中的词变换为词索引后构造Tensor
def build_label(all_tokens, all_seqs):
    vocab = Vocab.Vocab(collections.Counter(all_tokens), specials=[PAD, BOS, EOS])
    indices = [[vocab.stoi[w] for w in seq] for seq in all_seqs]
    return vocab, torch.tensor(indices)

def read_label(max_seq_len: int, labels: list):
    out_tokens, out_seqs = [], []
    for label in labels:
        out_seq = label
        out_seq_tokens = [ch for ch in out_seq]
        # if max(len(in_seq_tokens), len(out_seq_tokens)) > max_seq_len - 1:
        #     continue  # 如果加上EOS后长于max_seq_len，则忽略掉此样本
        process_one_seq(out_seq_tokens, out_tokens, out_seqs, max_seq_len)
        
    out_vocab, out_label = build_label(out_tokens, out_seqs)

    return out_vocab, out_label