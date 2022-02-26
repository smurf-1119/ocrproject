<<<<<<< HEAD
'''
dataset
'''

import torch
from torch.utils.data import DataLoader,Dataset
from utils.data_utilis import *
import json
import cv2
import os
import platform
import matplotlib.pyplot as plt
from torch.utils.data.distributed import DistributedSampler

class recognitionDataset(Dataset):
    def __init__(self, mode, input_dim, max_sequence, vocab_path,data_dir) -> None:
        super(recognitionDataset,self).__init__()
        self.mode = mode
        self.max_sequence = max_sequence
        self.input_dim = input_dim
        self.vocab_path = vocab_path
        self.vocab  = build_vocab(self.vocab_path)
        img_path = os.path.join(data_dir, self.mode, 'img')
        label_path = os.path.join(data_dir, self.mode, 'labels.json')
        self.img_base_file = img_path
        label_file = label_path
        with open(label_file,'r') as f:
            self.labels = json.load(f)['labels']
    def __getitem__(self, index):
        data = self.labels[index]
        image_path = data['image_path']
        image = np.array(cv2.imread(os.path.join(self.img_base_file,image_path)))
        resize_image = prep_image(image,self.input_dim)
        label = data['label']
        label = process_label(self.max_sequence,self.vocab,label)
        return resize_image,label
    def __len__(self):
        return len(self.labels) 

def collate_fn(batch):
    write = 0
    images = None
    labels = []
    for img, label in batch:
        if write == 0:
            images = img
            write += 1
        else:
            images = torch.cat([images, img], 0)
        labels.append(label)
    
    return [images, torch.cat(labels, dim=0)]



def getDataLoader(mode,batch_size,max_sequence, vocab_path='./dict.txt',data_dir='./AEC_recognition'):
    """"
    descriptions:generate the dataloader for the following operations
    params:
    @mode{str}:train,validation or test
    @batch_size{int}:the variable for the dataset
    @max_sequence{int}: make sure that the label is in a appropirate length

    @vocab_path{str}: './dict.txt'
    @img_path{str}
        "train":'./AEC_recognition/train/img'
        "validation":'./AEC_recognition/validation/img'
    @label_path{str}
        "train":'./AEC_recognition/train/labels.json'
        "validation":'./AEC_recognition/validation/labels.json'
    """
    dataset = recognitionDataset(mode,224,max_sequence, vocab_path,data_dir)
    if mode == 'test':
        iter = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers= 4 if platform.system() == "linux" else 0, collate_fn=collate_fn,sampler=DistributedSampler(dataset))
    else:
        iter = DataLoader(dataset,batch_size=batch_size,num_workers= 4 if platform.system() == "linux" else 0,collate_fn=collate_fn,sampler=DistributedSampler(dataset))
    return iter

def main():
    trainLoader = getDataLoader('train',2,35)
    for image,label in trainLoader:
        print(image.size())
        print(label)
        break
    # dataset = recognitionDataset('train',224)
    # img = dataset[0][0]
    # label = dataset[0][1]
    # print(img.shape)
    # print(label)
    # plt.imshow(img)
    # plt.show()

if __name__ == "__main__":
    main()

=======
'''
dataset
'''

import torch
from torch.utils.data import DataLoader,Dataset
from utils.data_utilis import *
import json
import cv2
import os
import platform
import matplotlib.pyplot as plt

class recognitionDataset(Dataset):
    def __init__(self, mode, input_dim, max_sequence, vocab_path,data_dir) -> None:
        super(recognitionDataset,self).__init__()
        self.mode = mode
        self.max_sequence = max_sequence
        self.input_dim = input_dim
        self.vocab_path = vocab_path
        self.vocab  = build_vocab(self.vocab_path)
        img_path = os.path.join(data_dir, self.mode, 'img')
        label_path = os.path.join(data_dir, self.mode, 'labels.json')
        self.img_base_file = img_path
        label_file = label_path
        with open(label_file,'r') as f:
            self.labels = json.load(f)['labels']
    def __getitem__(self, index):
        data = self.labels[index]
        image_path = data['image_path']
        image = np.array(cv2.imread(os.path.join(self.img_base_file,image_path)))
        resize_image = prep_image(image,self.input_dim)
        label = data['label']
        label = process_label(self.max_sequence,self.vocab,label)
        return resize_image,label
    def __len__(self):
        return len(self.labels) 

def collate_fn(batch):
    write = 0
    images = None
    labels = []
    for img, label in batch:
        if write == 0:
            images = img
            write += 1
        else:
            images = torch.cat([images, img], 0)
        labels.append(label)
    
    return [images, torch.cat(labels, dim=0)]



def getDataLoader(mode,batch_size,max_sequence, vocab_path='./dict.txt',data_dir='./AEC_recognition'):
    """"
    descriptions:generate the dataloader for the following operations
    params:
    @mode{str}:train,validation or test
    @batch_size{int}:the variable for the dataset
    @max_sequence{int}: make sure that the label is in a appropirate length

    @vocab_path{str}: './dict.txt'
    @img_path{str}
        "train":'./AEC_recognition/train/img'
        "validation":'./AEC_recognition/validation/img'
    @label_path{str}
        "train":'./AEC_recognition/train/labels.json'
        "validation":'./AEC_recognition/validation/labels.json'
    """
    dataset = recognitionDataset(mode,224,max_sequence, vocab_path,data_dir)
    if mode == 'test':
        iter = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers= 4 if platform.system() == "linux" else 0, collate_fn=collate_fn)
    else:
        iter = DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers= 4 if platform.system() == "linux" else 0,collate_fn=collate_fn)
    return iter

def main():
    trainLoader = getDataLoader('train',2,35)
    for image,label in trainLoader:
        print(image.size())
        print(label)
        break
    # dataset = recognitionDataset('train',224)
    # img = dataset[0][0]
    # label = dataset[0][1]
    # print(img.shape)
    # print(label)
    # plt.imshow(img)
    # plt.show()

if __name__ == "__main__":
    main()

>>>>>>> d133b2ab7694edf677283a5796d5f42503e2384c
