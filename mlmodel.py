import collections
import os

import numpy as np
import matplotlib.pyplot as plt

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
torch.set_default_tensor_type('torch.DoubleTensor')
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split

Model = collections.namedtuple('Model', 'phi h options')


class Phi_Net(nn.Module):
    def __init__(self, options):
        super(Phi_Net, self).__init__()

        self.fc1 = nn.Linear(options['dim_x'], 50)
        self.fc2 = nn.Linear(50, 60)
        self.fc3 = nn.Linear(60, 50)
        # One of the NN outputs is a constant bias term, which is append below
        self.fc4 = nn.Linear(50, options['dim_a']-1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        if len(x.shape) == 1:
            # single input
            return torch.cat([x, torch.ones(1)])
        else:
            # batch input for training
            return torch.cat([x, torch.ones([x.shape[0], 1])], dim=-1)
    
# Cross-entropy loss
class H_Net_CrossEntropy(nn.Module):
    def __init__(self, options):
        super(H_Net_CrossEntropy, self).__init__()
        self.fc1 = nn.Linear(options['dim_a'], 20)
        self.fc2 = nn.Linear(20, options['num_c'])
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def save_model(*, phi_net, h_net, modelname, options):
    if not os.path.isdir('./models/'):
        os.makedirs('./models/')
    if h_net is not None:
        torch.save({
            'phi_net_state_dict': phi_net.state_dict(),
            'h_net_state_dict': h_net.state_dict(),
            'options': dict(options)
        }, './models/' + modelname + '.pth')
    else:
        torch.save({
            'phi_net_state_dict': phi_net.state_dict(),
            'h_net_state_dict': None,
            'options': dict(options)
        }, './models/' + modelname + '.pth')

def load_model(modelname, modelfolder='./models/'):
    model = torch.load(modelfolder + modelname + '.pth')
    options = model['options']

    phi_net = Phi_Net(options=options)
    # h_net = H_Net_CrossEntropy(options)
    h_net = None

    phi_net.load_state_dict(model['phi_net_state_dict'])
    # h_net.load_state_dict(model['h_net_state_dict'])

    phi_net.eval()
    # h_net.eval()

    return Model(phi_net, h_net, options)


class MyDataset(Dataset):

    def __init__(self, inputs, outputs, c):
        self.inputs = inputs
        self.outputs = outputs
        self.c = c

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        Input = self.inputs[idx,]
        output = self.outputs[idx,]
        sample = {'input': Input, 'output': output, 'c': self.c}

        return sample


_softmax = nn.Softmax(dim=1)

## Training Helper Functions

def validation(phi_net, h_net, adaptinput: np.ndarray, adaptlabel: np.ndarray, valinput: np.ndarray, options, lam=0):
    """
    Helper function for compute the output given a sequence of data for adaptation (adaptinput)
    and validation (valinput)

    adaptinput: K x dim_x numpy, adaptlabel: K x dim_y numpy, valinput: B x dim_x numpy
    output: K x dim_y numpy, B x dim_y numpy, dim_a x dim_y numpy, B x dim_c numpy
    """
    with torch.no_grad():       
        # Perform least squares on the adaptation set to get a
        X = torch.from_numpy(adaptinput) # K x dim_x
        Y = torch.from_numpy(adaptlabel) # K x dim_y
        Phi = phi_net(X) # K x dim_a
        Phi_T = Phi.transpose(0, 1) # dim_a x K
        A = torch.inverse(torch.mm(Phi_T, Phi) + lam*torch.eye(options['dim_a'])) # dim_a x dim_a
        a = torch.mm(torch.mm(A, Phi_T), Y) # dim_a x dim_y
        
        # Compute NN prediction for the validation and adaptation sets
        inputs = torch.from_numpy(valinput) # B x dim_x
        val_prediction = torch.mm(phi_net(inputs), a) # B x dim_y
        adapt_prediction = torch.mm(phi_net(X), a) # K x dim_y
        
        # Compute adversarial network prediction
        temp = phi_net(inputs)
        if h_net is None:
            h_output = None
        else:
            h_output = h_net(temp) # B x num_of_c (CrossEntropy-loss) or B x dim_c (c-loss) or B x (dim_y*dim_a) (a-loss) 
            if options['loss_type'] == 'crossentropy-loss':
                # Cross-Entropy
                h_output = _softmax(h_output)
            h_output = h_output.numpy()
    
    return adapt_prediction.numpy(), val_prediction.numpy(), a.numpy(), h_output

def vis_validation(*, t, x, y, phi_net, h_net, idx_adapt_start, idx_adapt_end, idx_val_start, idx_val_end, c, options, lam=0):
    """
    Visualize performance with adaptation on x[idx_adapt_start:idx_adapt_end] and validation on x[idx_val_start:idx_val_end]
    """
    adaptinput = x[idx_adapt_start:idx_adapt_end, :]
    valinput = x[idx_val_start:idx_val_end, :]
    adaptlabel = y[idx_adapt_start:idx_adapt_end, :]
    y_adapt, y_val, a, h_output = validation(phi_net, h_net, adaptinput, adaptlabel, valinput, options, lam=lam)
    print(f'a = {a}')
    print(f"|a| = {np.linalg.norm(a,'fro')}")

    idx_min = min(idx_adapt_start, idx_val_start)
    idx_max = max(idx_adapt_end, idx_val_end)

    plt.figure(figsize=(15, 3))

    for i in range(3):
        plt.subplot(1, 4, i+1)
        plt.plot(t[idx_min:idx_max], y[idx_min:idx_max, i], 'k', alpha=0.3, label='gt')
        plt.plot(t[idx_val_start:idx_val_end], y_val[:, i], label='val')
        plt.plot(t[idx_adapt_start:idx_adapt_end], y_adapt[:, i], label='adapt')
        plt.legend()
        plt.title(r'$F_{s,' + 'xyz'[i] + '}$')

    if h_output is not None:
        plt.subplot(1, 4, 4)
        if options['loss_type'] == 'c-loss':
            colors = ['red', 'blue', 'green']
            for i in range(options['dim_c']):
                plt.plot(h_output[:, i], color=colors[i], label='c'+str(i))
                plt.hlines(c[i], xmin=0, xmax=len(h_output), linestyles='--', color=colors[i], label='c'+str(i)+' gt')
            plt.legend()
            plt.title('c prediction')
        if options['loss_type'] == 'crossentropy-loss':
            plt.plot(h_output)
            plt.title('c prediction (after Softmax)')
        if options['loss_type'] == 'a-loss':
            a_gt = a.reshape(1, options['dim_a'] * options['dim_y'])
            plt.plot(h_output - np.repeat(a_gt, h_output.shape[0], axis=0))
            # plt.hlines(a_gt, xmin=Data['time'][idx_val_start], xmax=Data['time'][idx_val_end]-1, linestyles='--')
            plt.title('a prediction')
    plt.show()

def error_statistics(data_input: np.ndarray, data_output: np.ndarray, phi_net, h_net, options):
    ''' Computes error statistics on given data.
        error1 is the loss without any learning 
        error2 is the loss when the prediction is the average output
        error3 is the loss using the NN where a is adapted on the entire dataset
     '''
    criterion = nn.MSELoss()

    with torch.no_grad():
        error_1 = criterion(torch.from_numpy(data_output), 0.0*torch.from_numpy(data_output)).item()
        error_2 = criterion(torch.from_numpy(data_output), torch.from_numpy(np.ones((len(data_output), 1))) * np.mean(data_output, axis=0)[np.newaxis, :])

        _, prediction, _, _ = validation(phi_net, h_net, data_input, data_output, data_input, options=options)
        error_3 = criterion(torch.from_numpy(data_output), torch.from_numpy(prediction)).item()
        
        return error_1, error_2, error_3