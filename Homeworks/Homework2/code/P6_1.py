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
    
class BOW(nn.Module):
    def __init__(self):
        super(BOW, self).__init__()
        self.train_acc_history = []
        self.test_acc_history = []
        self.dev_acc_history = []
        
        self.fc = nn.Sequential(
            nn.Linear(7647,1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        return x



def train(trainloader, net, criterion, optimizer, device):
    for epoch in range(30):  # loop over the dataset multiple times
        start = time.time()
        running_loss = 0.0
        for i, (images, labels) in enumerate(trainloader):
#            images = images.float().to(device)
#            labels = labels.float().to(device)
            images = images.type(torch.FloatTensor).to(device)
            labels = labels.type(torch.FloatTensor).to(device)
#            print(labels)
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
#        for i, (images, labels) in enumerate(testloader):
#            images = images.float().to(device)
#            labels = labels.float().to(device)
            images = images.type(torch.FloatTensor).to(device)
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

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
#    device = torch.device('cpu')

    word_to_idx = get_vocab('./data/train.txt')
    
    #BUG: unknow reason, if use BOW_Dataset, len(trainloader) == 77
    #Besides this dataset will make label and predicts all equal to 0, which generates 100% on all the training result
#    trainset = BOW_Dataset(txt_file='./data/train.txt', vocab = word_to_idx)
#    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True)
#    devset = BOW_Dataset(txt_file='./data/dev.txt', vocab = word_to_idx)
#    devloader = torch.utils.data.DataLoader(devset, batch_size=100, shuffle=True)
#    testset = BOW_Dataset(txt_file='./data/test.txt', vocab = word_to_idx)
#    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)
    
    trainset = Sentence2SentimentDataset(data_file='./data/train.txt', word_to_idx = word_to_idx)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, collate_fn = trainset.collate_fn)
    devset = Sentence2SentimentDataset(data_file='./data/dev.txt', word_to_idx = word_to_idx)
    devloader = torch.utils.data.DataLoader(devset, batch_size=100, shuffle=True, collate_fn = trainset.collate_fn)
    testset = Sentence2SentimentDataset(data_file='./data/test.txt', word_to_idx = word_to_idx)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, collate_fn = trainset.collate_fn)
    
    
    net = BOW().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    train(trainloader, net, criterion, optimizer, device)
    test(trainloader, net, device)
    test(devloader, net, device)
    test(testloader, net, device)
    
    plt.plot(net.dev_acc_history, label="val_acc")
    plt.plot(net.train_acc_history, label="train_acc")
    plt.legend(loc="lower right")
    
    
    unlabelledset = Sentence2SentimentDataset(data_file='./data/unlabelled.txt', word_to_idx = word_to_idx, labeled=False)
    unlabelledloader = torch.utils.data.DataLoader(unlabelledset, batch_size=100, shuffle=False, collate_fn = trainset.collate_fn)
    with open('predictions_q1.txt', 'w') as result_file:
        with torch.no_grad():
            for data in unlabelledloader:
                images, labels = data
                images = images.type(torch.FloatTensor).to(device)
                labels = labels.type(torch.FloatTensor).to(device)
                outputs = net(images)
                predicted = outputs > 0.5
                for i in predicted:
                    result_file.write("%s\n" % i.item())

if __name__== "__main__":
    main()
   
