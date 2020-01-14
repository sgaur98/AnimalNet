# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
"""
This is the main entry point for MP6. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset, DataLoader


class NeuralNet(torch.nn.Module):
    def __init__(self, lrate,loss_fn,in_size,out_size):
        """
        Initialize the layers of your neural network

        @param lrate: The learning rate for the model.
        @param loss_fn: A loss function defined in the following way:
            @param yhat - an (N,out_size) tensor
            @param y - an (N,) tensor
            @return l(x,y) an () tensor that is the mean loss
        @param in_size: Dimension of input
        @param out_size: Dimension of output

        For Part 1 the network should have the following architecture (in terms of hidden units):

        in_size -> 32 ->  out_size

        """
        super(NeuralNet, self).__init__()
        self.input_size = in_size
        #print(in_size)
        #self.hidden_size = 32
        self.hidden1_size  = 64
        self.hidden2_size = 32
        self.out_size = out_size

        #self.hidden = torch.nn.Linear(self.input_size, self.hidden_size)
        self.hidden1 = torch.nn.Linear(self.input_size, self.hidden1_size)
        self.hidden2 = torch.nn.Linear(self.hidden1_size, self.hidden2_size)
        self.output = torch.nn.Linear(self.hidden2_size, out_size)
        #self.output = torch.nn.Linear(self.hidden_size, self.out_size)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        self.loss_fn = loss_fn
        self.optimizer = optim.Adam(self.parameters(), lr=lrate)

    def set_parameters(self, params):
        """ Set the parameters of your network
        @param params: a list of tensors containing all parameters of the network
        """
        self.params = params

    def get_parameters(self):
        """ Get the parameters of your network
        @return params: a list of tensors containing all parameters of the network
        """
        return self.params


    def forward(self, x):
        """ A forward pass of your neural net (evaluates f(x)).

        @param x: an (N, in_size) torch tensor

        @return y: an (N, out_size) torch tensor of output from the network
        """
        x = self.hidden1(x)
        #x = self.sigmoid(x)
        x = self.relu(x)
        x = self.hidden2(x)
        #x = self.sigmoid(x)
        x = self.relu(x)

        x = self.output(x)
        #x = self.sigmoid(x)
        #x = self.relu(x)

        return x

    def step(self, x,y):
        """
        Performs one gradient step through a batch of data x with labels y
        @param x: an (N, in_size) torch tensor
        @param y: an (N,) torch tensor
        @return L: total empirical risk (mean of losses) at this time step as a float
        """

        #self.train()
        self.optimizer.zero_grad()

        yhat = self(x)
        #_, label = torch.max(yhat.data, 1)
        # print(y)
        # print(yhat)
        # print(label)
        loss = self.loss_fn(yhat, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()



def fit(train_set,train_labels,dev_set,n_iter,batch_size=100):
    """ Make NeuralNet object 'net' and use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, in_size) torch tensor
    @param train_labels: an (N,) torch tensor
    @param dev_set: an (M,) torch tensor
    @param n_iter: int, the number of epochs of training
    @param batch_size: The size of each batch to train on. (default 100)

    # return all of these:

    @return losses: Array of total loss at the beginning and after each iteration. Ensure len(losses) == n_iter
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: A NeuralNet object

    # NOTE: This must work for arbitrary M and N
    """

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)

    (N, in_size) = tuple(train_set.size())
    learn_rate = 0.000095
    #learn_rate = 0.0001
    out_size = 2
    n_iter = 25

    net = NeuralNet(learn_rate, torch.nn.CrossEntropyLoss(), in_size, out_size)
    params = list(net.parameters())
    net.set_parameters(params)


    training_losses = []
    development_losses = []
    dev_labels = None
    train_data = TensorDataset(train_set, train_labels)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=2)

    #dev_data = TensorDataset(dev_set)
    #dev_loader = DataLoader(dataset=dev_data, batch_size=batch_size, shuffle=False, num_workers=2)

    for epoch in range(0, n_iter):
        batch_losses = []
        for x_batch, y_batch in train_loader:
            loss = net.step(x_batch, y_batch)
            batch_losses.append(loss)
        training_loss = np.mean(batch_losses)
        #print("Epoch %d: %f" % (epoch, training_loss))
        training_losses.append(training_loss)

    print(training_losses)

    with torch.no_grad():
        #print(dev_set)
        #print(tuple(data.size()))
        net.eval()
        outputs = net(dev_set)
        _, predicted = torch.max(outputs.data, 1)
        #print(predicted)
        dev_labels = predicted


    torch.save(net, "net.model")
    return training_losses, dev_labels.numpy(), net
