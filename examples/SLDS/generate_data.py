import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
torch.set_default_dtype(torch.float64)

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
args = parser.parse_args()

from torchdiffeq import odeint

# from torchdiffeq import odeint
from torchdiffeq import odeint_event
from torchdiffeq import odeint_adjoint as odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
print(device)

# Random init_point (train: 100개. val, test: 25개)
true_y0 = torch.tensor([[3., 0.]]).to(device)

### FIXED PARAMETERS ###
# Uniform linspace and sort. (train: 0~4 50. test: 0~12 150)
train_t = torch.linspace(0., 4., 50).to(device)
test_val_t = torch.linspace(0., 12., 50).to(device)
# Dynamics Parameters
true_A = torch.tensor([[0., 1.], [-1., 0.]]).to(device)

c1 = torch.tensor([[0., 2.]]).to(device)
c2 = torch.tensor([[-1., -1.]]).to(device)
c3 = torch.tensor([[1., -1.]]).to(device)


class Lambda(nn.Module):

    def forward(self, t, y):

        if y[0, 0] >= 2:
            y0 = torch.mm(y, true_A) + c1
        elif y[0, 0] < 2 and y[0, 1] >= 0:
            y0 = c2
        elif y[0, 0] < 2 and y[0, 1] < 0:
            y0 = c3
        else:
            y0 = torch.mm(y, true_A)
        return y0


if __name__ == '__main__':
    # make train_set
    dyn = Lambda()
    trainset = []
    testset = []
    valset = []
    with torch.no_grad():
        for _ in tqdm(range(100)):
            # MAKE TRAIN
            r = torch.rand([1])*0.5 + 2.5 # 2.5 ~ 3.5
            theta = torch.rand([1]) * 3.141592 * 2
            y0 = torch.tensor([2 + r*torch.cos(theta), r*torch.sin(theta)]).to(device).unsqueeze(0)
            y = odeint(dyn, y0, train_t, method='rk4', options={'step_size': 0.01}).squeeze()
            trainset.append(y)
        trainset = torch.stack(trainset).cpu().numpy()
        print(trainset.shape)
        np.save('train.npy', trainset)
        for _ in tqdm(range(25)):
            # MAKE VAL
            r = torch.rand([1])*0.5 + 2.5  # 0.5 ~ 1.5
            theta = torch.rand([1]) * 3.141592 * 2
            y0 = torch.tensor([2 + r * torch.cos(theta), r * torch.sin(theta)]).to(device).unsqueeze(0)
            y = odeint(dyn, y0, test_val_t, method='rk4', options={'step_size': 0.01}).squeeze()
            valset.append(y)
        valset = torch.stack(valset).cpu().numpy()
        print(valset.shape)
        np.save('val.npy', valset)
        for _ in tqdm(range(25)):
            # MAKE TEST
            r = torch.rand([1])*0.5 + 2.5  # 0.5 ~ 1.5
            theta = torch.rand([1]) * 3.141592 * 2
            y0 = torch.tensor([2 + r * torch.cos(theta), r * torch.sin(theta)]).to(device).unsqueeze(0)
            y = odeint(dyn, y0, test_val_t, method='rk4', options={'step_size': 0.01}).squeeze()
            testset.append(y)
        testset = torch.stack(testset).cpu().numpy()
        print(testset.shape)
        np.save('test.npy', testset)
