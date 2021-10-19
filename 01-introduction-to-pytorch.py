"""
Author: Jamal Toutouh (www.jamal.es)

01-introduction-to-pytorch contains the code to create a basic neuron.
"""

# Import libraries
import numpy as np
import torch

from torch import nn
from torch.nn import functional as F

import matplotlib.pyplot as plt

# Function to show the loss during the training
def show_output(loss):
    y = np.array(loss)
    fig, ax = plt.subplots()
    ax.plot(y)
    ax.set(xlabel='training epochs (x50)', ylabel='loss',
           title='Loss evolution')
    ax.grid()
    plt.show()

def show_data(train_x, train_y, test_x, test_y=None):
  fig, ax = plt.subplots()
  
  x_researcher = train_x[:, 0][train_y == 0]
  y_researcher = train_x[:, 1][train_y == 0]

  x_industry = train_x[:, 0][train_y == 1]
  y_industry = train_x[:, 1][train_y == 1]

  plt.scatter(x_researcher, y_researcher, marker='o', c='k')
  plt.scatter(x_industry, y_industry, marker='^', c='k')

  print(test_x)
  print(test_y)
  
  if not test_y is None:
    x_test_researcher = test_x[:, 0][test_y == 0]
    y_test_researcher = test_x[:, 1][test_y == 0]

    x_test_industry = test_x[:, 0][test_y == 1]
    y_test_industry = test_x[:, 1][test_y == 1]
  else:
    x_test = test_x[:, 0]
    y_test = test_x[:, 1]

  if not test_y is None:
    plt.scatter(x_test_researcher, y_test_researcher, marker='o', c='r')
    plt.scatter(x_test_industry, y_test_industry, marker='^', c='r')
  else:
    plt.scatter(x_test, y_test, marker='*', c='b')
  
  plt.show()

# -------------------------------------------------

class Net(nn.Module):
  """It encapsulates a two layer neural neuron with:
  - 2 inputs
  - a hidden layer with 2 neurons (h1, h2)
  - an output layer with 1 neuron (o1)
  """
  def __init__(self):
    super().__init__()
    self.hid = nn.Linear(2, 2)  
    self.out = nn.Linear(2, 1)

  def forward(self, x):
    x = torch.sigmoid(self.hid(x))
    x = torch.sigmoid(self.out(x)) 
    return x

# -------------------------------------------------

# Define dataset

# Real data
train_x = np.array([
    [12, 6],  # Michael (12 hours/week, 6 papers)
    [32, 4],  # Shash (32 hours/week, 4 papers)
    [11, 3],  # Hannah (11 hours/week,  3 papers)
    [29, 11],  # Lisa (29 hours/week, 11 papers)
])

# jamal = np.array([-0.426254443, -0.647635041])  # 14 hours/week, 3 papers
# mina = np.array([1.112853239, -1.490562272])  # 31 hours/week, 0 papers

# Normalized data
train_x = np.array([
  [-0.607325935, 0.195292189],
  [1.203388985, -0.366659298],
  [-0.697861681, -0.647635041],
  [0.931781747, 1.600170906]
])

test_x = np.array([
                   [-0.426254443, -0.647635041],
                   [1.112853239, -1.490562272]              
])


train_y = np.array([
    0,  # Michael
    1,  # Shash
    0,  # Hannah
    1,  # Lisa
])

show_data(train_x, train_y, test_x)

# Define tensors to train the network using PyTorch
train_x = torch.tensor(train_x, dtype=torch.float32)
train_y = torch.tensor(train_y, dtype=torch.float32)

# 2. create network
net = Net()

# 3. train model
epochs = 1000
lrn_rate = 0.001
loss_fun = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=lrn_rate)

show = True
loss_to_show = []
print("\nStarting training ")
net.train()

for epoch in range(0, epochs):
  for i in range(len(train_x)):
    X = train_x[i].reshape(1,2)
    Y = train_y[i].flatten()

    output = net(X)

    loss_obj = loss_fun(output, Y.reshape(-1,1))
    optimizer.zero_grad()
    loss_obj.backward()
    optimizer.step()
    # (monitor error)

# For logging purposes
  if epoch % 50 == 0:
      output = net(train_x)
      loss = loss_fun(output, train_y.reshape(-1,1))
      loss_to_show.append(loss)
      print("Epoch %d loss: %.3f" % (epoch, loss))

if show: show_output(loss_to_show)
print("Done training ")

# 4. use model to make a prediction
net.eval()

jamal = np.array([-0.426254443, -0.647635041])  # 14 hours/week, 3 papers
mina = np.array([1.112853239, -1.490562272])  # 31 hours/week, 0 papers

jamal = torch.tensor(jamal, dtype=torch.float32)
mina = torch.tensor(mina, dtype=torch.float32)

output_jamal = net(jamal)
prediction_jamal = output_jamal.reshape(-1).detach().numpy().round()
print('Jamal: ' + str(prediction_jamal))

output_mina = net(mina)
prediction_mina = output_mina.reshape(-1).detach().numpy().round()
print('Mina: ' + str(prediction_mina))

test_y = np.array([prediction_jamal[0], prediction_mina[0]])

# Show data

show_data(train_x, train_y, test_x, test_y)

