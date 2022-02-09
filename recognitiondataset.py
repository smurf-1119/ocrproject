import torch
from torch.utils.data import DataLoader,Dataset
from utils.data_utilis import *
import json
import cv2
import os
import platform
import matplotlib.pyplot as plt

class recognitionDataset(Dataset):
    def __init__(self,mode,input_dim,max_sequence) -> None:
        super(recognitionDataset,self).__init__()
        self.max_sequence = max_sequence
        self.input_dim = input_dim
        self.vocab_path = './dict.txt'
        self.vocab = build_vocab(self.vocab_path)
        if mode == 'train':
            self.img_base_file = './AEC_recognition/train/img'
            label_file = './AEC_recognition/train/labels.json'
            with open(label_file,'r') as f:
                self.labels = json.load(f)['labels']
        elif mode =='validation':
            self.img_base_file = './AEC_recognition/validation/img'
            label_file = './AEC_recognition/validation/labels.json'
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
    images = None
    labels = []
    for img, label in batch:
        if images == None:
            images = img
        else:
            images = torch.cat([images, img], 0)
        labels.append(label)
    return [images, labels]



def getDataLoader(mode,batch_size,max_sequence):
    """"
    descriptions:generate the dataloader for the following operations
    params:
    @mode{str}:train,validation or test
    @batch_size{int}:the variable for the dataset
    @max_sequence{int}: make sure that the label is in a appropirate length
    """
    dataset = recognitionDataset(mode,416,max_sequence)
    if mode == 'test':
        iter = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers= 4 if platform.system() == "linux" else 0, collate_fn=collate_fn)
    else:
        iter = DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers= 4 if platform.system() == "linux" else 0,collate_fn=collate_fn)
    return iter

def main():
    trainLoader = getDataLoader('train',2,7)
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


