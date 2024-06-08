import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tqdm
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
torch.set_default_device(device)
torch.set_default_dtype(torch.float64)
print("Setting default device to: ", device)

class MLP(nn.Module):
    def __init__(self, hidden_layer_sizes=(100,), 
                 input_size=1, 
                 output_size=1, 
                 activation=nn.GELU(), 
                 last_activation=nn.Identity(),):
        super(MLP, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_sizes = hidden_layer_sizes

        self.layers = nn.ModuleList()
        self.activation = activation
        self.layers.append(nn.Linear(input_size, hidden_layer_sizes[0]))
        for i in range(1, len(hidden_layer_sizes)):
            self.layers.append(nn.Linear(hidden_layer_sizes[i-1], hidden_layer_sizes[i]))
        self.layers.append(nn.Linear(hidden_layer_sizes[-1], output_size))
        self.last_activation = last_activation

    def forward(self, x):
        x = self.layers[0](x)
        x = self.activation(x)
        for layer in self.layers[1:-1]:
            x = layer(x)
            x = self.activation(x)
        x = self.layers[-1](x)
        x = self.last_activation(x)
        return x
    
class LearnableVariable(nn.Module):

    def __init__(self, shape=(1,), activation=nn.Identity(), init_value=None):
        super(LearnableVariable, self).__init__()
        self.shape = shape
        if init_value is not None:
            self.variable = nn.Parameter(init_value * torch.ones(shape))
        else:
            self.variable = nn.Parameter(torch.randn(shape))
        self.activation = activation   

    def forward(self, x=None):
        return self.activation(self.variable) * torch.ones((x.shape[0], 1))
    
class ConstantVariable(nn.Module):

    def __init__(self, value):
        super(ConstantVariable, self).__init__()
        self.constant_variable = nn.Parameter(torch.tensor(value), requires_grad=False)

    def forward(self, x=None):
        return self.constant_variable * torch.ones((x.shape[0], 1))

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
        
    def forward(self, x):
        return self.lambd(x)