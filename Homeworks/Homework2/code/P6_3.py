import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

import math
from collections import OrderedDict
import matplotlib.pyplot as plt

from torch.utils.data.dataset import Dataset
import numpy as np
class Sentence2SentimentDataset(Dataset):
    def __init__(self, data_file, max_length, word_to_idx = {}, labeled = True):
        """
        Args:
            data_file (string): Path to the txt file
            word_to_idx (string): Directory with all the words.
            max_length (int): max number of words in a sentence, if less pad with len(word_to_idx)
            labeled (Bool): if False, append the unlabelled sentences with label '0' at the beginning
        """
        self.data_file = open(data_file, 'r')
        self.sent_data = self.data_file.readlines()
        if labeled is False:
            for i in range(len(self.sent_data)):
                self.sent_data[i] = "0 " + self.sent_data[i]
        
        self.word_to_idx = word_to_idx
        self.max_length = max_length
    
    def collate_fn(self, data):
        x = None
        y = []
        #print(data)
        for sent, label in data:
            sent_tensor = torch.empty(self.max_length)
            for i in range(self.max_length):
                if i < len(sent):
                    if sent[i] in self.word_to_idx:
                        sent_tensor[i] = self.word_to_idx[sent[i]]
                    else:
                        sent_tensor[i] = len(self.word_to_idx)
                else:
                    sent_tensor[i] = len(self.word_to_idx)
                    
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

class WordEmbedding(nn.Module):
    def __init__(self, max_length, GloVe, pool_size=16):
        super(WordEmbedding, self).__init__()
        self.train_acc_history = []
        self.test_acc_history = []
        self.dev_acc_history = []
        
        self.embed = nn.Sequential(
                nn.Embedding.from_pretrained(GloVe, freeze=False),
                nn.AvgPool2d((1, pool_size)),
        )
        self.fc = nn.Sequential(
            nn.Linear(int(max_length * GloVe.size(1)/pool_size), 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.embed(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def get_vocab(train_file):
    word_to_idx = {}
    
    # pre-process train_file
    train_data = []
    file = open(train_file, 'r')
    for line in file:
        line = line.split(' ')
        train_data.append(line[1:])
            
    for words in train_data:
        for word in words:
            if word not in word_to_idx:
                word_to_idx[word] = len(word_to_idx)
            
    return word_to_idx

def get_GloVe(word_to_idx):
    GloVe_dict = {}
    with open('./glove.6B.50d.txt', 'r', encoding = "utf8") as f:
        for line in f:
            #first ele of line is the word, there rest are embeddings
            eles = line.split(' ')
            embed = np.array([float(ele) for ele in eles[1:]])
            GloVe_dict[eles[0]] = embed
        
#        GloVe = torch.zeros(len(word_to_idx), 50)
        #len(word_to_idx)+1 to reserver the last row of the matrix for UNK words in test
        GloVe = torch.zeros(len(word_to_idx)+1, 50)
        for word, idx in word_to_idx.items():
            if word in GloVe_dict:
                GloVe[idx] = torch.from_numpy(GloVe_dict[word])
    return GloVe


def train(trainloader, net, criterion, optimizer, device):
    for epoch in range(7):  # loop over the dataset multiple times
        start = time.time()
        running_loss = 0.0
        for i, (images, labels) in enumerate(trainloader):
            images = images.type(torch.LongTensor).to(device)
            labels = labels.type(torch.FloatTensor).to(device)
            
            # TODO: zero the parameter gradients
            optimizer.zero_grad()
            
            # TODO: forward pass
            outputs = net(images)
            
            # TODO: backward pass
            loss = criterion(outputs, labels)
            loss.backward()
            
            # TODO: optimize the network
            optimizer.step()
            
            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 2000 mini-batches
                end = time.time()
                print('[epoch %d, iter %5d] loss: %.3f eplased time %.3f' %
                      (epoch + 1, i + 1, running_loss / 100, end-start))
                start = time.time()
                running_loss = 0.0
        net.train_acc_history.append(test(trainloader, net, device))
        net.dev_acc_history.append(test(devloader, net, device))
        net.test_acc_history.append(test(testloader, net, device))

    print('Finished Training')


def test(testloader, net, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.type(torch.LongTensor).to(device)
            labels = labels.type(torch.FloatTensor).to(device)
            outputs = net(images)
            
#            _, predicted = torch.max(outputs.data, 1)
            predicted = (outputs > 0.5)
            total += labels.size(0)
            #BUG: predicted is [100,1], labels is [100], (predicted.float() == labels) returns [100, 100]
            correct += (predicted.float().squeeze() == labels.squeeze()).sum().item()
            
    acc = 100 * correct / total
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        acc))
    return acc

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#    device = torch.device('cpu')
    print(device)

    word_to_idx = get_vocab('./data/train.txt')
    GloVe = get_GloVe(word_to_idx)
    max_length = 20 #Max number of word in a sentence
    
    trainset = Sentence2SentimentDataset(data_file='./data/train.txt', word_to_idx = word_to_idx, max_length = max_length)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, collate_fn = trainset.collate_fn)
    devset = Sentence2SentimentDataset(data_file='./data/dev.txt', word_to_idx = word_to_idx, max_length = max_length)
    devloader = torch.utils.data.DataLoader(devset, batch_size=100, shuffle=True, collate_fn = trainset.collate_fn)
    testset = Sentence2SentimentDataset(data_file='./data/test.txt', word_to_idx = word_to_idx, max_length = max_length)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, collate_fn = trainset.collate_fn)
    
    
    net = WordEmbedding(max_length = max_length, GloVe=GloVe, pool_size=1).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    train(trainloader, net, criterion, optimizer, device)
    test(trainloader, net, device)
    test(devloader, net, device)
    test(testloader, net, device)
    
    plt.plot(net.dev_acc_history, label="val_acc")
    plt.plot(net.train_acc_history, label="train_acc")
    plt.legend(loc="lower right")
    
    
    unlabelledset = Sentence2SentimentDataset(data_file='./data/unlabelled.txt', word_to_idx = word_to_idx, max_length = max_length, labeled=False)
    unlabelledloader = torch.utils.data.DataLoader(unlabelledset, batch_size=100, shuffle=False, collate_fn = trainset.collate_fn)
    with open('predictions_q3.txt', 'w') as result_file:
        with torch.no_grad():
            for data in unlabelledloader:
                images, labels = data
                images = images.type(torch.LongTensor).to(device)
                labels = labels.type(torch.FloatTensor).to(device)
                outputs = net(images)
                predicted = outputs > 0.5
                for i in predicted:
                    result_file.write("%s\n" % i.item())

if __name__== "__main__":
    main()
   
