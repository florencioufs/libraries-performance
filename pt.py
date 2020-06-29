#!/usr/bin/env python
# coding: utf-8

# In[1]:



import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as dsets
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# In[2]:


#LeNet-5

class CNN(nn.Module):

    # network structure
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        '''
        One forward pass through the network.
        
        Args:
            x: input
        '''
        x = F.avg_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.avg_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        '''
        Get the number of features in a batch of tensors `x`.
        '''
        size = x.size()[1:]
        return np.prod(size)


# In[3]:


def fit (model, train_loader, database_size, num_epochs=10):
    learning_rate = 0.001
    # loss function and optimizer
    criterion = torch.nn.functional.cross_entropy
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    start = time.time()
    for epoch in range(num_epochs):
        total_loss = 0
        for i, (images, labels) in enumerate(train_loader):

            # Forward + Backward + Optimize
            outputs = model(images.cuda())
            loss = criterion(outputs,labels.cuda())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    end = time.time()
    print(f'{end-start:.6f}')
    return model


# In[4]:


def predict (model, test_loader):
    model.eval()
    correct = 0
    total = 0
    start = time.time()
    for images, labels in test_loader:
        outputs = model(images.cuda())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels).sum()
    end = time.time()
    print(f'{end-start:.6f}')

# In[5]:


def main():
    
    batch_size = 100
    transform = torchvision.transforms.Compose ([
        torchvision.transforms.ToTensor(),
    ])
    train_dataset = dsets.MNIST(root='./MNIST/',
                                       download=True,
                                       transform=transform)
    test_dataset = dsets.MNIST(root='./MNIST/',
                                      train=False,
                                      download=True,
                                      transform=transform)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size, shuffle=True)
    # show_img(train_dataset)

    model = CNN().cuda()
    for i in range (101):
        model = fit(model,train_loader,len(train_dataset),10)
    for i in range (101):
        predict(model,test_loader)
    torch.save (model.state_dict(),'cnn_model.th')


# In[6]:


if __name__ == '__main__':
    main()


# In[ ]:




