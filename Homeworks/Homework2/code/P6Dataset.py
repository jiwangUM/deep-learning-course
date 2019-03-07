#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 13:54:30 2019

@author: tradingmachine
"""
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import numpy as np

class Sentence2SentimentDataset(Dataset):
    def __init__(self, data_file, transform=None, mode = 'BoW', word_to_idx = {}, labeled = True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_file = open(data_file, 'r')
        self.sent_data = self.data_file.readlines()
        if labeled is not True:
            for i in range(len(self.sent_data)):
                self.sent_data[i] = "0 " + self.sent_data[i]
        
        self.transform = transform
        self.mode = mode
        self.word_to_idx = word_to_idx
    
    def collate_fn(self, data):
        x = None
        y = []
        #print(data)
        for sent, label in data:
            if self.mode is 'BoW':
                sent_tensor = torch.zeros(len(self.word_to_idx))
                for word in sent:
                    if word in self.word_to_idx:
                        sent_tensor[self.word_to_idx[word]] += 1
                        
            x = torch.unsqueeze(sent_tensor,0) if x is None else torch.cat((x, torch.unsqueeze(sent_tensor, 0)), 0)
            #print(x.size())
            y.append(label)
        
        return x, torch.tensor(y)

    def __len__(self):
        return len(self.sent_data)

    def __getitem__(self, idx):
        line = self.sent_data[idx].split()
        sample = (line[1:], int(line[0]))

        return sample
    

class BOW_Dataset(Dataset):
    """Sentence Sentiment Classification dataset."""

    def __init__(self, txt_file, vocab = None):
        """
        Args:
            txt_file (string): Path to the txt file with annotations.
        """
        self.to_tensor = transforms.ToTensor()
        self.label = []
        self.words = []
        self.dict = []
        file = open(txt_file, 'r')
        for line in file:
             words = line.split(' ')
             #BUG: words[0] is a string, have to convert it to int
             self.label.append(int(words[0]))
             self.words.append(words[1:]) #append becomes 2D
             self.dict += words[1:] #still 1D
        
        if vocab is None:
            self.dict = dict(enumerate(set(self.dict)))
            self.dict = {v:k for k, v in self.dict.items()} #dict must add .items to be iterable
        else:
            self.dict = vocab
        
        print(len(self.words))
             
    def __len__(self):
        return len(self.dict)

    def __getitem__(self, idx):
        data = np.zeros(len(self.dict), dtype=float)
        label = np.zeros(1, dtype=float)
        for word in self.words[idx]:
            if word in self.dict:
                data[self.dict[word]] += 1
                
        label[0] = self.label[idx]
        return (data, label)