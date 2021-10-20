"""
Author: Jamal Toutouh (www.jamal.es)

02-introduction-to-pytorch contains the code to create a basic neuron.
"""


import torch
import torch.nn as nn


from pandas import read_csv
import numpy as np
from sklearn.model_selection import train_test_split

data_df = read_csv('https://raw.githubusercontent.com/jamaltoutouh/CursoGAN-URep-2021/main/Tema%203/pima-indians-diabetes.csv', header=None)

data = data_df.to_numpy()
np.shape(data)

# Creating dataset
X_data = data[:,0:8]          
print('X_data:',np.shape(X_data))
y_data = data[:,8]
print('Y_data:',np.shape(y_data))


# Split data into X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.25, random_state= 0)


# Check the dimension of the sets
print('X_train:',np.shape(X_train))
print('y_train:',np.shape(y_train))
print('X_test:',np.shape(X_test))
print('y_test:',np.shape(y_test))

from torch.utils.data import Dataset
# Class to define the train dataset
class trainData(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)


train_data = trainData(torch.FloatTensor(X_train),  torch.FloatTensor(y_train))

# Class to define the test dataset
class testData(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)
    

test_data = testData(torch.FloatTensor(X_test),
                     torch.FloatTensor(y_test))

# Creating Neural Networks using Sequential()
class MyNeuralNetwork(nn.Module):
    """
    Class that defines a Neural Network with: 
    - an input layer of size 8, 
    - a hidden layer of size 10, 
    - and an output layer of size 1.
    """
    def __init__(self):
        super(MyNeuralNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(8, 10),
            nn.LeakyReLU(0.2),
            nn.Linear(10, 10),
            nn.LeakyReLU(0.2),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )
       
    def forward(self, x):
        x = self.net(x)
        return x

myNN = MyNeuralNetwork()

# Selecting a Loss function
myNN_loss = nn.BCELoss() # For a binary classification problem
myNN_optimizer = torch.optim.SGD(myNN.parameters(), lr=0.01, momentum=0.5) # SGD optimizer

EPOCHS = 100
BATCH_SIZE = 50

from torch.utils.data import Dataset, DataLoader

train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=1)

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc

myNN.train()
for e in range(1, EPOCHS+1):
    epoch_loss = 0
    epoch_acc = 0
    for X_batch, y_batch in train_loader:
        #X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        myNN_optimizer.zero_grad()
        
        y_pred = myNN(X_batch)
        
        loss = myNN_loss(y_pred, y_batch.unsqueeze(1))
        acc = binary_acc(y_pred, y_batch.unsqueeze(1))
        
        loss.backward()
        myNN_optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        

    print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')

myNN.eval()

test_loss = 0
test_acc = 0

for X_batch, y_batch in test_loader:
     
        y_pred = myNN(X_batch)
        
        loss = myNN_loss(y_pred, y_batch.unsqueeze(1))
        acc = binary_acc(y_pred, y_batch.unsqueeze(1))
        
        loss.backward()
        myNN_optimizer.step()
        
        test_loss += loss.item()
        test_acc += acc.item()
print(f'Epoch {e+0:03}: | Loss: {test_loss/len(test_loader):.5f} | Acc: {test_acc/len(test_loader):.3f}')