#!/usr/bin/env python3
import argparse
import matplotlib.pyplot as plt
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

import numpy as np
from torchdiffeq import odeint, odeint_adjoint
from torchdiffeq import odeint_event

from tqdm import tqdm

import random
random.seed(1234, version=2)

torch.set_default_dtype(torch.float64)

# Load .npy files.
    # input: (x, y, x, y, 0, 0, 0, 0)
    # state: (x, y, x, y, vx, vy, vx, vy)
# ODE Module Specify
    # Given initial points --> unroll 100.
    # ODESolveEvent: dynamics:f, event_fn: g
    # Instant update h.
# Train / Val roop.

class BouncingBall(Dataset):

    def __init__(self, root='train.npy', seq_len=25, subseq=True):
        super().__init__()
        self.data = np.load(root)
        self.subseq = subseq
        self.seq_len = seq_len

    def __getitem__(self, item):
        # Parse 25-length subsequence.
        # Get Velocity. [pos, vel]
        pos = self.data[item]
        velocity = pos[1:] - pos[:-1]
        velocity = np.concatenate([np.zeros([1, 4]), velocity], 0)

        if self.subseq:
            start = random.randint(0, 74)
        else:
            start = 0
        end = start+self.seq_len
        state = np.concatenate([pos, velocity], -1)
        return state[start:end] # --> [25, 8]

    def __len__(self):
        return len(self.data)

class DynsSolver(nn.Module):

    def __init__(self, dt=1./30., seq_len=25, layer_norm=False, adjoint=True):
        super().__init__()
        # self.event_fn = EventFn(hidden_dim=128, layer_norm=layer_norm)
        self.dynamics = GravityDyns(dt, layer_norm=layer_norm)
        # self.inst_func = InstFn(hidden_dim=512, n_hidden=3, layer_norm=layer_norm)
        self.seq_len = seq_len

    def forward(self, state):

        sample_ts = [float(t) for t in range(0, self.seq_len)]
        sample_ts = torch.tensor(sample_ts, requires_grad=True).cuda()
        y0 = state

        results = odeint(self.dynamics, y0, sample_ts,
                                 method='rk4', options={'step_size': 0.1})[1:]
        # try:
        #     results = odeint(self.dynamics, y0, sample_ts, method='dopri5')[1:]
        # except AssertionError:
        #     results = odeint(self.dynamics, y0, sample_ts,
        #                              method='rk4', options={'step_size': 0.1})[1:]
        return None, results


class GravityDyns(nn.Module):
    def __init__(self, dt, gravity=9.8, hidden_dim=256, layer_norm=False):
        super().__init__()
        gravity = gravity * dt * dt
        self.gravity = torch.tensor([0., gravity, 0., gravity], requires_grad=True).cuda()
        if layer_norm:
            self.mlp = nn.Sequential(
                nn.Linear(8, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 4)
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(8, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 4)
            )

    def forward(self, t, state):
        dx = state[4:]
        dv = self.mlp(state)

        # dv --> constant for event ODE, MLP for NODE.
        return torch.cat([dx, dv], dim=-1)


def train(args):
    writer = SummaryWriter(os.path.join('logs', 'orig_ode', args.name))
    trainset = BouncingBall('train.npy', args.seq_len, not args.no_subseq)
    valset = BouncingBall('val.npy')
    testset = BouncingBall('test.npy')

    # batchsize == 1
    train_loader = DataLoader(trainset, shuffle=True, num_workers=0)
    # val_loader = DataLoader(valset, shuffle=False)
    # test_loader = DataLoader(testset, shuffle=False)

    model = DynsSolver(dt=args.dt, seq_len=args.seq_len, layer_norm=args.layer_norm, )
    model = model.cuda()

    # Param_group
    params = [
        {'params': model.parameters(), 'lr': args.lr},
        ]
    optimizer = optim.Adam(params)

    os.makedirs(os.path.join('checkpoints', 'orig_ode', args.name), exist_ok=True)
    for epoch in range(100):
        epoch_loss = 0
        for i, batch in tqdm(enumerate(train_loader), total=1000):
            # squeeze unnecessary batch dimension
            batch = batch.squeeze(0).cuda()

            # Split batch to input, target
            input_state = batch[0]
            target_pos = batch[1:, :4]

            pred_t, pred_state = model(input_state)
            pred_pos = pred_state[:, :4]
            # for stability
            # pred_pos = torch.clamp(pred_pos, min=0, max=5)

            # MSE Loss
            loss = F.mse_loss(pred_pos, target_pos)

            try:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                optimizer.step()
            except AssertionError:
                pass

            writer.add_scalar('loss', loss.item(), epoch * len(trainset) + i)
            epoch_loss += loss.item()
        writer.add_scalar('epoch_loss', epoch_loss / len(trainset))
        torch.save(model.state_dict(), os.path.join('checkpoints', args.name, f'{epoch}.pth'))

'''
TODO:
- tensorboard logging.
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default='default')
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--clip_grad", default=5.0, type=float)
    parser.add_argument("--dt", default=1./30., type=float)
    parser.add_argument("--seq_len", default=25, type=int)
    parser.add_argument("--no_subseq", action='store_true')

    parser.add_argument("--layer_norm", action='store_true')
    args = parser.parse_args()
    train(args)
