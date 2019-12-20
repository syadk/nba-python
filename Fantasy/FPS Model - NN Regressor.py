import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, TensorDataset, DataLoader
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
from itertools import product
from collections import OrderedDict
from collections import namedtuple
import numpy as np
import os
import pandas as pd
import wandb
import argparse



os.chdir('C:\\GitHub\\nba-python\\Fantasy\\FPS Input Data for Model')
dfview = pd.read_pickle('2017-2018 DFS Input test.pkl')


########################## Hyper-parameters 
#num_epochs = 50
#learning_rate = 0.001
#weight_decay = 0.001
#layers_size = 28
#num_layers = 2
#input_size = 12
output_size = 1


############## INPUT DATA AND CONVERT TO BINARY
def load_data(file):
    os.chdir('C:\\GitHub\\nba-python\\Fantasy\\FPS Input Data for Model')
    df = pd.read_pickle(file)
    df = df.dropna()
    df.drop(columns=['Player', 'day'],inplace=True)
#    df = df[['actual', 'Player_FPS']]
    df.reset_index(drop=True,inplace=True)

    dftrain, dftest = train_test_split(df, test_size=0.15)
    dftrain.reset_index(drop=True,inplace=True)
    trainy = pd.to_numeric(dftrain['actual'])
    temp=(dftrain.drop(['actual'], axis=1))
    scaler = StandardScaler()
    temp.reset_index(drop=True,inplace=True)
    trainx = scaler.fit_transform(temp)
    testy = pd.to_numeric(dftest['actual'])
    temp=dftest.drop(['actual'], axis=1)
    testx = scaler.transform(temp)
    input_size = temp.shape[1]
    del df, dftrain, dftest, temp
    #turn the pandas series into numpy arrays
    testy = testy.values
    trainy = trainy.values
    #turn the numpy arrays into tensors, with x data as type float
    trainx = torch.from_numpy(trainx).float()
    trainy = torch.from_numpy(trainy).float()
    testx = torch.from_numpy(testx).float()
    testy = torch.from_numpy(testy).float()
    ##############################################################################
    
    class CustomDataset(Dataset):
        def __init__(self, x_tensor, y_tensor):
            self.x = x_tensor
            self.y = y_tensor
            
        def __getitem__(self, index):
            return (self.x[index], self.y[index])
    
        def __len__(self):
            return len(self.x)
    
    train_data = CustomDataset(trainx, trainy)
    train_loader = DataLoader(dataset=train_data, batch_size=trainx.shape[0], shuffle=True)
    
    test_data = CustomDataset(testx, testy)
    test_loader = DataLoader(dataset=test_data, batch_size=testx.shape[0], shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    return(train_loader, test_loader, device, input_size)


#def train(layers_size, num_layers, num_epochs, weight_decay, train_loader, test_loader, device):
class NeuralNet(nn.Module):
    def __init__(self, input_size, layers_size, num_layers, output_size):
        super(NeuralNet, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(input_size, layers_size)])
        for i in range(1, num_layers - 1):
            self.linears.extend([nn.Linear(layers_size, layers_size)])
        self.linears.append(nn.Linear(layers_size, output_size))
        self.sigmoid = nn.Sigmoid()

        
    
    def forward(self, x):
        for i in range(len(self.linears)-1):
            x = F.relu(self.linears[i](x))
        x = self.linears[i+1](x)
        return x
    
    
def train(model, num_epochs, learning_rate, weight_decay, train_loader, device, comment):  

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Train the model
    for epoch in range(num_epochs):
        for i, (features, labels) in enumerate(train_loader):  
            # Move tensors to the configured device
            features = features.reshape(-1, features.shape[1]).to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs.squeeze(), labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(epoch, loss.item())
    return(model, loss.item())

    
def test(model, test_loader):
    for i, (features, labels) in enumerate(test_loader):
        testx = features
        testy = labels
    criterion = nn.MSELoss()
    #test data
    y_pred = model(testx)
    test_error = criterion(y_pred.squeeze(), testy)
    print(test_error.item())
    return(test_error.item(), y_pred, testy)  

#train(input_size, layers_size, num_layers, output_size, train_loader, test_loader, device)

def main(file, layers_size, num_layers, num_epochs, weight_decay, comment):

    train_loader, test_loader, device, input_size = load_data(file)
    model = NeuralNet(input_size, layers_size, num_layers, output_size)
    model_trained, train_error = train(model, num_epochs, learning_rate, weight_decay, train_loader, device, comment)
    test_error, y_pred, testy = test(model_trained, test_loader)

    
    return(test_error, train_error, y_pred, testy)
#combine(test_file, input_size, layers_size, num_layers, output_size)

learning_rate = 0.01
file = '2017-2018 DFS Input test.pkl'
#

    


params = OrderedDict(epochs = [200],
                     regularization = [0.001],
                     layers_size = [5],
                     num_layers = [5])

class RunBuilder():
    @staticmethod
    def get_runs(params):

        Run = namedtuple('Run', params.keys())

        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs

df = pd.DataFrame(columns=['File', 'Epochs','Regularization','Layer Size', 'Number of Layers',
                           'Test Error', 'Train Error'])

for run in RunBuilder.get_runs(params):
    
    comment = f'-{run}'
    print(comment)
    test_error, train_error, y_pred, testy = main(file, run.layers_size, run.num_layers, run.epochs, run.regularization, comment)
        
    df = df.append({'File':file, 'Epochs':run.epochs,'Regularization':run.regularization,
                                    'Layer Size':run.layers_size, 'Number of Layers':run.num_layers, 
                                    'Test Error':test_error, 'Train Error':train_error}, ignore_index=True)
    
#quick test
df_view = pd.DataFrame(data=[y_pred.detach().squeeze().numpy(), testy.numpy()]).transpose()

